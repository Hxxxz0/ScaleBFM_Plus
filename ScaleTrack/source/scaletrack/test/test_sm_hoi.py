"""HOI object heads, state and policy (no simulator required)."""

import importlib.util
import math
import os
import unittest
from pathlib import Path

import torch

from sm_offline import load

heads = load("object_heads")
state = load("state")
rot = load("rotations")

SUPERTRACKER_REWARD = Path(os.environ.get("SUPERTRACKER_ROOT", "")) / "source/supertracker/hoi_reward.py"


def random_quat(n, generator=None):
    return rot.quat_normalize(torch.randn(n, 4, generator=generator, dtype=torch.float64))


class ObjectRewardTest(unittest.TestCase):
    def test_matches_supertracker_kernel(self):
        """tracking_components equals supertracker's kernel on random inputs and its counterfactuals."""
        if not SUPERTRACKER_REWARD.exists():
            self.skipTest("Supertracker not available")
        spec = importlib.util.spec_from_file_location("st_reward", SUPERTRACKER_REWARD)
        st = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(st)
        st.qualification("cpu")
        g = torch.Generator().manual_seed(0)
        args = [torch.randn(256, 3, generator=g) * 0.01 for _ in range(4)] + [torch.rand(256, generator=g) for _ in range(4)]
        args[0][:32] = 0.0  # resting object
        args[2][32:48] = 0.0  # resting reference
        args[3][32:48] = 0.0
        ref = st.tracking_components(*args)
        ours = heads.tracking_components(*args)
        for k in ("score", "progress", "position"):
            torch.testing.assert_close(ours[k], ref[k], rtol=0, atol=1e-7)

    def test_surrogate_equal_when_moving_and_has_gradient_at_rest(self):
        g = torch.Generator().manual_seed(1)
        dp = torch.randn(64, 3, generator=g) * 0.01
        dr = torch.randn(64, 3, generator=g) * 0.01
        target = [torch.randn(64, 3, generator=g) * 0.01 for _ in range(2)]
        err = [torch.rand(64, generator=g) for _ in range(4)]
        a = heads.tracking_components(dp, dr, *target, *err)["score"]
        b = heads.tracking_components(dp, dr, *target, *err, surrogate=True)["score"]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        rest = torch.zeros(64, 3, requires_grad=True)
        heads.tracking_components(rest, torch.zeros(64, 3), *target, *err)["score"].sum().backward()
        self.assertEqual(float(rest.grad.abs().sum()), 0.0)
        rest.grad = None
        heads.tracking_components(rest, torch.zeros(64, 3), *target, *err, surrogate=True)["score"].sum().backward()
        self.assertGreater(float(rest.grad.abs().sum()), 0.0)

    def test_rot6d_rows_matches_matrix(self):
        q = random_quat(32)
        w, x, y, z = q.unbind(-1)
        R = torch.stack(
            [
                torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
            ],
            -2,
        )
        torch.testing.assert_close(heads.rot6d_rows(q), R[..., :2].reshape(32, 6))

    def test_token_invalid_is_zero_and_frame_invariant(self):
        g = torch.Generator().manual_seed(2)
        n, T = 8, 6
        anchor = (torch.randn(n, 3, generator=g, dtype=torch.float64), random_quat(n, g),
                  torch.randn(n, 3, generator=g, dtype=torch.float64), torch.randn(n, 3, generator=g, dtype=torch.float64))
        obj = torch.cat([torch.randn(n, 3, generator=g, dtype=torch.float64), random_quat(n, g),
                         torch.randn(n, 6, generator=g, dtype=torch.float64)], -1)
        ref = torch.cat([torch.randn(n, T, 3, generator=g, dtype=torch.float64), rot.quat_normalize(torch.randn(n, T, 4, generator=g, dtype=torch.float64)),
                         torch.randn(n, T, 6, generator=g, dtype=torch.float64)], -1)
        dt = torch.rand(n, T, dtype=torch.float64)
        dims = torch.tensor([0.36, 0.33, 0.34], dtype=torch.float64)
        valid = torch.tensor([1, 0] * 4, dtype=torch.float64)
        tok = heads.object_token(anchor, obj, ref, dt, dims, valid)
        self.assertEqual(tok.shape, (n, 1, T, 35))
        self.assertTrue(bool((tok[1::2] == 0).all()))
        # Rotating and translating the whole world about z leaves the token unchanged.
        yaw = rot.quat_from_yaw(torch.full((n,), 0.7, dtype=torch.float64))
        shift = torch.tensor([1.0, -2.0, 0.0], dtype=torch.float64)

        def move(p, q, v, w):
            return rot.quat_apply(yaw, p) + shift, rot.quat_mul(yaw, q), rot.quat_apply(yaw, v), rot.quat_apply(yaw, w)

        anchor2 = move(*anchor)
        o = heads.split_object(obj)
        obj2 = torch.cat(move(*o), -1)
        r = heads.split_object(ref)
        yT = yaw[:, None].expand(n, T, 4)
        ref2 = torch.cat([rot.quat_apply(yT, r[0]) + shift, rot.quat_mul(yT, r[1]), rot.quat_apply(yT, r[2]), rot.quat_apply(yT, r[3])], -1)
        torch.testing.assert_close(heads.object_token(anchor2, obj2, ref2, dt, dims, valid), tok)

    def test_reward_terms_zero_for_resting_reference_and_invalid(self):
        n = 4
        obj = torch.zeros(n, 13)
        obj[:, 3] = 1.0
        moved = obj.clone()
        moved[:, 0] = 0.01
        bodies = torch.randn(n, 14, 3)
        valid = torch.tensor([1.0, 1.0, 0.0, 0.0])
        cfg = heads.ObjectHeadsCfg()
        hold = heads.object_reward_terms(cfg, obj, moved, obj, obj, bodies, bodies, valid)
        self.assertTrue(bool((hold["object_motion"] == 0).all()))
        follow = heads.object_reward_terms(cfg, obj, moved, obj, moved, bodies, bodies, valid)
        self.assertGreater(float(follow["object_motion"][0]), 0.99)
        self.assertTrue(bool((follow["object_motion"][2:] == 0).all()))
        self.assertGreater(float(follow["object_relation"][0]), 0.99)


class HOIStateTest(unittest.TestCase):
    def setUp(self):
        self.spec = state.HOIStateSpec(state.HOIStateSpecCfg(), "cpu")
        g = torch.Generator().manual_seed(3)
        n = 16
        robot = torch.randn(n, state.STATE_DIM, generator=g)
        robot[:, 3:7] = rot.quat_normalize(robot[:, 3:7])
        obj = torch.randn(n, 13, generator=g)
        obj[:, 3:7] = rot.quat_normalize(obj[:, 3:7])
        valid = torch.tensor([1.0, 0.0] * 8)[:, None]
        index = torch.arange(16.0)[:, None] % 3
        self.s = torch.cat([robot, obj, valid, index], -1)

    def test_dims_and_robot_part_unchanged(self):
        spec, s = self.spec, self.s
        self.assertEqual(s.shape[-1], state.HOI_STATE_DIM)
        robot_spec = state.StateSpec(state.StateSpecCfg(), "cpu")
        f = spec.features(s)
        self.assertEqual(f.shape[-1], spec.in_scale.numel())
        torch.testing.assert_close(f[:, : robot_spec.in_scale.numel()], robot_spec.features(s[:, : state.STATE_DIM]))
        self.assertTrue(bool((f[1::2, robot_spec.in_scale.numel() :] == 0).all()))
        delta = torch.zeros(16, spec.out_scale.numel())
        nxt = spec.integrate(s, delta)
        torch.testing.assert_close(nxt[:, : state.STATE_DIM], robot_spec.integrate(s[:, : state.STATE_DIM], delta[:, :70]))

    def test_zero_delta_is_constant_velocity(self):
        spec, s = self.spec, self.s
        nxt = spec.integrate(s, torch.zeros(16, spec.out_scale.numel()))
        p, q, v, w, valid = state.split_object_state(s)
        p2, q2, v2, w2, valid2 = state.split_object_state(nxt)
        torch.testing.assert_close(nxt[:, -1], s[:, -1])
        torch.testing.assert_close(p2, p + spec.dt * v)
        torch.testing.assert_close(v2, v)
        torch.testing.assert_close(w2, w)
        torch.testing.assert_close(valid2, valid)
        torch.testing.assert_close(q2, rot.quat_normalize(rot.quat_mul(rot.quat_from_rotvec(spec.dt * w), q)))

    def test_error_masks_invalid_object(self):
        spec, s = self.spec, self.s
        other = s.clone()
        other[:, state.STATE_DIM : state.STATE_DIM + 3] += 1.0
        e = spec.error(other, s)
        self.assertEqual(e.shape[-1], spec.err_scale.numel())
        self.assertTrue(bool((e[1::2, 70:] == 0).all()))
        self.assertTrue(bool((e[0::2, 70:73] != 0).all()))


class HandBPSTest(unittest.TestCase):
    def test_matches_brute_force_and_masks(self):
        g = torch.Generator().manual_seed(4)
        n, S = 6, 40
        hand_pos = torch.randn(n, 2, 3, generator=g) * 0.2
        hand_quat = rot.quat_normalize(torch.randn(n, 2, 4, generator=g))
        basis = heads.hand_bps_basis((0.12, 0.0, 0.0))
        obj = torch.cat([torch.randn(n, 3, generator=g) * 0.1, rot.quat_normalize(torch.randn(n, 4, generator=g)), torch.zeros(n, 6)], -1)
        surface = torch.randn(n, S, 3, generator=g) * 0.2
        valid = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        out = heads.hand_bps(hand_pos, hand_quat, basis, obj, surface, valid).reshape(n, 2, -1, 3)
        world_surf = obj[:, None, :3] + rot.quat_apply(obj[:, None, 3:7].expand(n, S, 4), surface)
        for e in range(n):
            for h in range(2):
                wb = hand_pos[e, h] + rot.quat_apply(hand_quat[e, h].expand(len(basis), 4), basis)
                d = world_surf[e][None] - wb[:, None]
                near = d[torch.arange(len(basis)), d.norm(dim=-1).argmin(-1)]
                local = rot.quat_apply_inverse(hand_quat[e, h].expand(len(basis), 4), near)
                local = local * (heads.HAND_BPS_CLIP / local.norm(dim=-1, keepdim=True).clamp_min(heads.HAND_BPS_CLIP))
                torch.testing.assert_close(out[e, h], local * valid[e], atol=1e-5, rtol=0)
        hand_pos.requires_grad_(True)
        heads.hand_bps(hand_pos, hand_quat, basis, obj, surface, valid).sum().backward()
        self.assertTrue(bool(torch.isfinite(hand_pos.grad).all()))
        self.assertEqual(float(hand_pos.grad[3].abs().sum()), 0.0)


class ColliderBottomTest(unittest.TestCase):
    def test_shared_and_per_environment_hulls(self):
        """Multi-object envs hold one hull each (N, P, 3, padded by repeating a point); single-object share (P, 3)."""
        g = torch.Generator().manual_seed(5)
        n, P = 7, 11
        hulls = torch.randn(n, P, 3, generator=g, dtype=torch.float64) * 0.3
        hulls[:, 8:] = hulls[:, :1]  # padding
        obj = torch.cat([torch.randn(n, 3, generator=g, dtype=torch.float64), random_quat(n, g), torch.zeros(n, 6, dtype=torch.float64)], -1)
        got = heads.collider_bottom(hulls, obj)
        for e in range(n):
            ref = (obj[e, :3] + rot.quat_apply(obj[e, 3:7].expand(P, 4), hulls[e]))[:, 2].min()
            torch.testing.assert_close(got[e], ref)
            torch.testing.assert_close(heads.collider_bottom(hulls[e], obj[e : e + 1])[0], ref)  # shared hull
        self.assertEqual(heads.collider_bottom(hulls[0], obj).shape, (n,))


class ContactMatchTest(unittest.TestCase):
    def test_matches_brute_force_and_has_gradient(self):
        g = torch.Generator().manual_seed(6)
        n, L, P = 5, 4, 50
        obj = torch.cat([torch.randn(n, 3, generator=g, dtype=torch.float64), random_quat(n, g), torch.zeros(n, 6, dtype=torch.float64)], -1)
        points = torch.randn(n, P, 3, generator=g, dtype=torch.float64) * 0.2
        world = obj[:, None, :3] + rot.quat_apply(obj[:, None, 3:7].expand(n, P, 4), points)
        links = world[:, :L] + torch.randn(n, L, 3, generator=g, dtype=torch.float64) * 0.03
        labels = torch.tensor([[1, -1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1], [-1, -1, 0, 0], [1, 0, -1, 0]], dtype=torch.float64)
        got = heads.contact_match(links, obj, points, labels)
        for e in range(n):
            d = (world[e][None] - links[e][:, None]).norm(dim=-1).min(-1).values
            s = torch.exp(-(((d - heads.CONTACT_MARGIN).clamp_min(0) / heads.CONTACT_SIGMA) ** 2))
            m = labels[e] != 0
            ref = (1 - ((labels[e] > 0).double() - s).abs())[m].mean() if m.any() else torch.tensor(0.0, dtype=torch.float64)
            torch.testing.assert_close(got[e], ref)
        links.requires_grad_(True)
        heads.contact_match(links, obj, points, labels).sum().backward()
        self.assertTrue(bool(torch.isfinite(links.grad).all()))
        self.assertEqual(float(links.grad[1].abs().sum()), 0.0)  # no labels: no gradient


class SoftTerminationTest(unittest.TestCase):
    def test_rotation_threshold_terminates_toppled_objects(self):
        n = 4
        ref = torch.cat([torch.zeros(n, 3), torch.tensor([[1.0, 0, 0, 0]]).expand(n, 4), torch.zeros(n, 6)], -1).double()
        obj = ref.clone()
        angle = torch.tensor([0.0, 0.5, 1.5, 0.0], dtype=torch.float64)
        obj[:, 3], obj[:, 5] = torch.cos(angle / 2), torch.sin(angle / 2)  # turned about y
        obj[3, 0] = 0.8  # far, not turned
        valid = torch.ones(n, dtype=torch.float64)
        off = heads.object_soft_termination(heads.ObjectHeadsCfg(), obj, ref, valid)
        on = heads.object_soft_termination(heads.ObjectHeadsCfg(term_rot_threshold=1.0), obj, ref, valid)
        self.assertLess(float(off[:3].max()), 1e-6)
        self.assertGreater(float(off[3]), 0.999)
        self.assertLess(float(on[:2].max()), 1e-3)
        self.assertGreater(float(on[2]), 0.999)
        self.assertGreater(float(on[3]), 0.999)
        torch.testing.assert_close(heads.object_soft_termination(heads.ObjectHeadsCfg(term_rot_threshold=1.0), obj, ref, valid * 0), torch.zeros(n, dtype=torch.float64))


class StackedSamplerTest(unittest.TestCase):
    def test_stacked_cdf_draws_each_table_with_its_probabilities(self):
        """hoi_command._start_frames: one searchsorted over per-table CDFs shifted by the table index."""
        torch.manual_seed(0)
        nb, K = 50, 3
        table = torch.randint(0, K + 1, (nb,))
        p = torch.rand(nb, dtype=torch.float64) * (torch.rand(nb) > 0.2)
        p[table == 1] = 0.0  # an empty table in the middle
        order = torch.argsort(table, stable=True)
        mass = torch.zeros(K + 1, dtype=torch.float64).index_add_(0, table, p)
        cdf = torch.cumsum((p / mass[table].clamp_min(1e-300))[order], 0)
        nonempty = (mass > 0).double()
        start = torch.cumsum(nonempty, 0) - nonempty
        n = 200000
        t = torch.randint(0, K + 1, (n,))
        t = torch.where(mass[t] > 0, t, int(torch.nonzero(mass > 0)[0]))
        u = torch.rand(n, dtype=torch.float64).clamp(1e-9, 1 - 1e-9)
        b = order[torch.searchsorted(cdf, start[t] + u).clamp_max(nb - 1)]
        self.assertTrue(bool((table[b] == t).all()))
        self.assertTrue(bool((p[b] > 0).all()))
        for k in range(K + 1):
            sel = t == k
            if mass[k] == 0 or sel.sum() == 0:
                continue
            freq = torch.bincount(b[sel], minlength=nb).double() / sel.sum()
            expect = torch.where(table == k, p / mass[k], torch.zeros_like(p))
            self.assertLess(float((freq - expect).abs().max()), 0.01)


class HOIPolicyTest(unittest.TestCase):
    def make(self, objects: bool):
        from tensordict import TensorDict

        from my_rsl_rl.modules import ActorCriticHumanoidTransformer

        torch.manual_seed(0)
        n = 4
        obs = {
            "policy": torch.randn(n, 3, 64), "policy_task": torch.randn(n, 6, 253), "critic": torch.randn(n, 3, 266),
            "critic_task": torch.randn(n, 7, 337), "action": torch.randn(n, 3, 29), "mode": torch.ones(n, 14),
            "mode_mapping": torch.ones(n, 253),
        }  # fmt: skip
        if objects:
            obs["object_actor"] = torch.randn(n, 1, 6, 35)
            obs["object_critic"] = torch.randn(n, 1, 7, 35)
        obs = TensorDict(obs, batch_size=[n])
        policy = ActorCriticHumanoidTransformer(obs, {"policy": ["policy"], "critic": ["critic"]}, 29, embedding_dim=64, ff_dim=64, num_layers=2)
        return policy, obs

    def test_invalid_objects_reproduce_robot_policy(self):
        robot, obs = self.make(False)
        hoi, obs_hoi = self.make(True)
        missing = hoi.load_state_dict(robot.state_dict(), strict=False)
        for k in obs.keys():
            obs_hoi[k] = obs[k]
        obs_hoi["object_actor"][..., :-1] = float("nan")
        obs_hoi["object_actor"][..., -1] = 0.0
        a = robot.act_inference(obs)
        b = hoi.act_inference(obs_hoi)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        parts = (obs["policy"], obs["policy_task"], obs["action"], obs["mode"], obs["mode_mapping"])
        a = robot.actor_mean_from_parts(*parts)
        b = hoi.actor_mean_from_parts(parts[0], (parts[1], obs_hoi["object_actor"]), *parts[2:])
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_object_branch_is_trained_and_observation_not_modified(self):
        hoi, obs = self.make(True)
        obs["object_actor"][..., -1] = 1.0
        names = {id(p) for p in hoi.actor_object_encoder.parameters()}
        self.assertTrue(names <= {id(p) for p in hoi.actor_parameters})
        before = obs["policy_task"].clone()
        obs["mode_mapping"].zero_()
        hoi.act(obs)
        torch.testing.assert_close(obs["policy_task"], before, rtol=0, atol=0)
        parts = (obs["policy"], (obs["policy_task"], obs["object_actor"]), obs["action"], obs["mode"], torch.ones(4, 253))
        with torch.no_grad():  # the output head starts at zero, which blocks every gradient upstream
            hoi.actor.projection_head.weight.normal_()
        hoi.actor_mean_from_parts(*parts).sum().backward()
        self.assertGreater(float(hoi.actor_object_encoder[0].weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

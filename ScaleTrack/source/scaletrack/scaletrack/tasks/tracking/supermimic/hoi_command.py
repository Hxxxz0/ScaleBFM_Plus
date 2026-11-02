"""Motion command of the HOI tracking task: one dynamic object per environment, reset with the robot.

Every environment holds the same rigid object. Motions with an object (``lib.object_valid``) place it at its
reference state on reset; the other motions park it out of reach, where it is masked out of observations, rewards and
terminations. On top of SuperMimicMotionCommand:

* episodes of object motions only start at the audited supported frames (``lib.start_ok``: the object rests on the
  floor and will move), and at the original start with probability ``original_start_prob``;
* environments only start motions of their own object (or motions without an object); ``hoi_mass`` (>= 0.5) fixes
  the share of starts with the object, as the object data is small next to the flat-ground data;
* the return-based reset sampling scores the robot rewards only (the object reward pays nothing while the reference
  object rests, so its attainable return changes along a motion);
* failure retries (HybridReset) rewind only motions without an object, unless ``object_retries`` (supertracker's
  failure-window mining: the object restarts at its reference state just before the failure);
* ``hoi_mass_range`` makes the share of object starts per object adaptive: objects whose bins return least get more
  of their environments' starts (``hoi_mass`` scaled by the object's difficulty over the mean, clipped to the range);
* evaluation starts object motions at their audited start and accumulates the full-clip object criteria.
"""

from __future__ import annotations

import math
import torch
from dataclasses import MISSING

from isaaclab.utils import configclass

from .command import SuperMimicMotionCommand, SuperMimicMotionCommandCfg
from .object_heads import ObjectHeadsCfg, collider_bottom, increment, motion_credit, object_errors

OBJECT_REWARD_TERMS = ("object_motion", "object_relation", "object_contact")


class _RobotOnlyRetries:
    """HybridResetSampler whose local retries are reverted for motions with an object."""

    def __init__(self, sampler, lib):
        self.sampler, self.lib = sampler, lib
        self.group_names = sampler.group_names

    def clear(self, env_ids=None):
        self.sampler.clear(env_ids)

    def drain_statistics(self):
        return self.sampler.drain_statistics()

    def select(self, global_frames, env_ids, context, active):
        frames = self.sampler.select(global_frames, env_ids, context, active)
        return torch.where(self.lib.frame_object_valid[frames] > 0.5, global_frames, frames)


class HOIMotionCommand(SuperMimicMotionCommand):
    cfg: HOIMotionCommandCfg

    def __init__(self, cfg: HOIMotionCommandCfg, env):
        super().__init__(cfg, env)
        if not self.train_lib.has_objects or (self.has_test_set and not self.test_lib.has_objects):
            raise ValueError("HOI tracking needs packed libraries with object fields (pack_hoi_library.py)")
        self.object = env.scene[cfg.object_name]
        self.obj_cfg = ObjectHeadsCfg()
        # Parked objects rest out of reach on their lowest hull point (identity orientation).
        self.park_height = -env.object_hull[..., 2].amin(-1) + 0.002
        lib = self.train_lib
        if not 0.5 <= cfg.hoi_mass <= 1.0:
            raise ValueError(f"hoi_mass must be in [0.5, 1] (motions with an object at least half of the starts), got {cfg.hoi_mass}")
        self.num_objects = len(env.objects)
        if len(lib.object_names) != self.num_objects:
            raise ValueError("library object_names differ from the environment's objects")
        # Bins grouped into tables 0..K-1 (one per object) and K (no object), in table order.
        self._bin_table = torch.where(lib.bin_object >= 0, lib.bin_object, self.num_objects)
        self._bin_order = torch.argsort(self._bin_table, stable=True)
        if self.reset_sampler is not None and not cfg.object_retries:
            self.reset_sampler = _RobotOnlyRetries(self.reset_sampler, lib)
        if cfg.hoi_mass_range is not None and not 0.5 <= cfg.hoi_mass_range[0] <= cfg.hoi_mass_range[1] <= 1.0:
            raise ValueError(f"hoi_mass_range must satisfy 0.5 <= low <= high <= 1, got {cfg.hoi_mass_range}")
        self._table_bins = torch.zeros(self.num_objects + 1, device=self.device).index_add_(
            0, self._bin_table, (lib.bin_ok_count > 0).float()
        )
        self.metrics["hoi_mass"] = torch.full((self.num_envs,), cfg.hoi_mass, device=self.device)
        self._reward_ids = None
        for k in ("error_object_pos", "error_object_rot"):
            self.metrics[k] = torch.zeros(self.num_envs, device=self.device)
        n = self.num_envs
        self.eval_acc = {
            k: torch.zeros(n, device=self.device)
            for k in ("max_pos", "max_rot", "credit", "distance", "lifted", "ref_lift_frames")
        }
        # Envs whose evaluation clip is still running (set by the runner before every evaluation step).
        self.eval_active = torch.zeros(n, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ object reference

    def object_reference(self, frames: torch.Tensor) -> torch.Tensor:
        """(..., 13) env-local reference object state of the active library at global frames."""
        lib = self.lib
        return torch.cat(
            [lib.object_pos_w[frames], lib.object_quat_w[frames], lib.object_lin_vel_w[frames], lib.object_ang_vel_w[frames]],
            -1,
        )

    @property
    def object_valid(self) -> torch.Tensor:
        return self.lib.object_valid[self.motion_ids]

    def previous_frame(self, frames: torch.Tensor) -> torch.Tensor:
        lib = self.lib
        return torch.maximum(frames - 1, lib.starts[lib.frame_motion[frames]])

    # ------------------------------------------------------------------ sampling

    def _bin_reward(self, env_ids):
        rm = self._env.reward_manager
        if self._reward_ids is None:
            self._reward_ids = [rm.active_terms.index(k) for k in OBJECT_REWARD_TERMS if k in rm.active_terms]
        obj = rm._step_reward[env_ids][:, self._reward_ids].sum(-1) * self._env.step_dt
        return self._env.reward_buf[env_ids] - obj

    def _evaluation_start(self, env_ids):
        return self.lib.eval_start[self.motion_ids[env_ids]]

    def _start_frames(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Start frames sampled per environment: with probability ``hoi_mass`` a motion of the environment's own object,
        otherwise a motion without an object. Within each group the return-based bin probabilities are kept; all
        environments are drawn at once from one stacked CDF (searchsorted, no host synchronisation)."""
        cfg, lib = self.cfg, self.train_lib
        n = len(env_ids)
        hard = torch.softmax(-cfg.softmax_temp * self.bin_return, dim=0)
        self.bin_memory = (1 - cfg.memory_ema) * self.bin_memory + cfg.memory_ema * hard
        p = (
            cfg.uniform_ratio / lib.num_bins
            + cfg.hard_ratio * hard
            + cfg.memory_ratio * self.bin_memory / self.bin_memory.sum()
        )
        p = (p * (lib.bin_ok_count > 0)).double()
        K = self.num_objects
        mass = torch.zeros(K + 1, dtype=p.dtype, device=self.device).index_add_(0, self._bin_table, p)
        # Every non-empty table contributes 1 to the running sum: table t spans (start[t], start[t] + 1].
        cdf = torch.cumsum((p / mass[self._bin_table].clamp_min(1e-300))[self._bin_order], 0)
        nonempty = (mass > 0).to(p.dtype)
        start = torch.cumsum(nonempty, 0) - nonempty
        obj = self._env.object_ids[env_ids]
        has_obj, has_flat = mass[obj] > 0, mass[K] > 0
        hoi_mass = self._object_mass()[obj]
        self.metrics["hoi_mass"][env_ids] = hoi_mass
        use_obj = has_obj & ((torch.rand(n, device=self.device) < hoi_mass) | ~has_flat)
        table = torch.where(use_obj, obj, K)
        u = torch.rand(n, device=self.device, dtype=p.dtype).clamp(1e-9, 1 - 1e-9)
        b = self._bin_order[torch.searchsorted(cdf, start[table] + u).clamp_max(lib.num_bins - 1)]
        k = (torch.rand(n, device=self.device) * lib.bin_ok_count[b]).long().clamp_max(lib.bin_ok_count[b] - 1)
        f = lib.ok_frames[lib.bin_ok_start[b] + k]
        motion = lib.frame_motion[f]
        original = lib.object_valid[motion] & (torch.rand(n, device=self.device) < cfg.original_start_prob)
        f = torch.where(original, lib.starts[motion] + lib.eval_start[motion], f)
        q = p / p.sum()
        self.metrics["sampling_entropy"][:] = -(q * (q + 1e-12).log()).sum() / math.log(max(lib.num_bins, 2))
        return f

    def _object_mass(self) -> torch.Tensor:
        """(K,) share of object starts per object: hoi_mass, or with hoi_mass_range hoi_mass times the object's
        difficulty (1 - mean bin return over its startable bins) relative to the mean over objects, clipped."""
        cfg, K = self.cfg, self.num_objects
        if cfg.hoi_mass_range is None:
            return torch.full((K,), cfg.hoi_mass, device=self.device)
        ok = (self.train_lib.bin_ok_count > 0).float()
        hard = torch.zeros(K + 1, device=self.device).index_add_(0, self._bin_table, (1.0 - self.bin_return) * ok)
        d = (hard / self._table_bins.clamp_min(1))[:K]
        present = (self._table_bins[:K] > 0).float()
        mean = (d * present).sum() / present.sum().clamp_min(1)
        scale = torch.where(mean > 1e-6, d / mean.clamp_min(1e-6), torch.ones_like(d))
        return (cfg.hoi_mass * scale).clamp(*cfg.hoi_mass_range)

    # ------------------------------------------------------------------ resets

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        super()._resample_command(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        frames = self._global_time_index()[env_ids]
        origins = self._env.scene.env_origins[env_ids]
        ref = self.object_reference(frames)
        valid = self.lib.object_valid[self.motion_ids[env_ids]]
        # Motions without an object: park it at rest, out of reach of the robot.
        park = torch.zeros_like(ref)
        park[:, 0:2] = self.body_pos_w[env_ids, 0, :2] - origins[:, :2]
        park[:, 1] += self.cfg.park_distance
        park[:, 2] = self.park_height[env_ids]
        park[:, 3] = 1.0
        state = torch.where(valid[:, None], ref, park)
        self.object.write_root_link_pose_to_sim(torch.cat([state[:, 0:3] + origins, state[:, 3:7]], -1), env_ids=env_ids)
        self.object.write_root_link_velocity_to_sim(state[:, 7:13], env_ids=env_ids)
        prev = getattr(self._env, "object_prev", None)
        if prev is not None:
            prev[env_ids] = state

    # ------------------------------------------------------------------ evaluation

    def _update_metrics(self):
        if not self.is_evaluating:
            return
        super()._update_metrics()
        env = self._env
        frames = self._global_time_index()
        obj = env.object_state()
        ref = self.object_reference(frames)
        valid = self.object_valid.float() * self.eval_active
        e = object_errors(obj, ref)
        self.metrics["error_object_pos"] = e["position"] * valid
        self.metrics["error_object_rot"] = e["rotation"] * valid
        a = self.eval_acc
        a["max_pos"] = torch.maximum(a["max_pos"], self.metrics["error_object_pos"])
        a["max_rot"] = torch.maximum(a["max_rot"], self.metrics["error_object_rot"])
        prev = env.object_prev
        ref_prev = self.object_reference(self.previous_frame(frames))
        dp, dr = increment(prev[:, 0:3], prev[:, 3:7], obj[:, 0:3], obj[:, 3:7])
        tdp, tdr = increment(ref_prev[:, 0:3], ref_prev[:, 3:7], ref[:, 0:3], ref[:, 3:7])
        credit, length = motion_credit(dp, dr, tdp, tdr)
        a["credit"] += credit * valid
        a["distance"] += length * valid
        h = self.obj_cfg.lift_height
        hull = env.object_hull
        a["lifted"] = torch.maximum(a["lifted"], (collider_bottom(hull, obj) > h).float() * valid)
        a["ref_lift_frames"] += (collider_bottom(hull, ref) > h).float() * valid

    def reset_object_evaluation(self) -> None:
        for v in self.eval_acc.values():
            v.zero_()
        self.eval_active.zero_()

    def evaluation_chunks(self, motion_ids: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Split evaluation motions into chunks of (positions in motion_ids, environment slots): a motion with an
        object runs in an environment holding that object. Unused environments track a compatible filler motion."""
        lib, env_obj = self.lib, self._env.object_ids.cpu()
        lib_obj = lib.object_id.cpu()
        obj = lib_obj[motion_ids.cpu()]
        pending = list(range(len(motion_ids)))
        filler = {}
        for k in range(self.num_objects):
            match = torch.nonzero(lib_obj == k).squeeze(-1)
            filler[k] = int(match[0]) if len(match) else None
        flat = torch.nonzero(lib_obj < 0).squeeze(-1)
        flat_filler = int(flat[0]) if len(flat) else None
        chunks = []
        while pending:
            free = {k: torch.nonzero(env_obj == k).squeeze(-1).tolist() for k in range(self.num_objects)}
            ids, slots, rest = [], [], []
            for i in pending:
                k = int(obj[i])
                pool = free.get(k) if k >= 0 else max(free.values(), key=len)
                if pool:
                    ids.append(i)
                    slots.append(pool.pop())
                else:
                    rest.append(i)
            if not ids:
                raise RuntimeError("no environment holds the object of a pending evaluation motion")
            chunks.append((torch.tensor(ids, device=self.device), torch.tensor(slots, device=self.device)))
            pending = rest
        self._eval_filler = torch.tensor(
            [flat_filler if flat_filler is not None else filler[int(k)] for k in env_obj], device=self.device
        )
        return chunks

    def fill_evaluation(self, slots: torch.Tensor) -> None:
        """Give every environment not in ``slots`` a filler motion compatible with its object."""
        rest = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        rest[slots] = False
        self.motion_ids[rest] = self._eval_filler[rest]

    def object_evaluation(self, slots: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Full-clip object failure of the evaluation envs ``slots`` (supertracker HOIRunner criteria) and per-env
        object metrics; envs without an object never fail here."""
        a, c = self.eval_acc, self.obj_cfg
        g = {k: v[slots] for k, v in a.items()}
        valid = self.lib.object_valid[self.motion_ids[slots]]
        lift_required = g["ref_lift_frames"] >= 10
        reasons = {
            "object_fail_pos": g["max_pos"] > c.eval_max_pos_error,
            "object_fail_rot": g["max_rot"] > c.eval_max_rot_error,
            "object_fail_coverage": g["credit"] < c.eval_min_motion_credit * g["distance"],
            "object_fail_lift": lift_required & (g["lifted"] < 0.5),
        }
        failed = (reasons["object_fail_pos"] | reasons["object_fail_rot"] | reasons["object_fail_coverage"]
                  | reasons["object_fail_lift"]) & valid  # fmt: skip
        metrics = {k: v.float() for k, v in reasons.items()} | {
            "object_max_pos_error": g["max_pos"],
            "object_max_rot_error": g["max_rot"],
            "object_motion_coverage": g["credit"] / g["distance"].clamp_min(1e-6),
        }
        return failed, metrics


@configclass
class HOIMotionCommandCfg(SuperMimicMotionCommandCfg):
    class_type: type = HOIMotionCommand

    object_name: str = "object"
    hoi_mass: float = 0.5
    # (low, high): adaptive share of object starts per object around hoi_mass (None: fixed hoi_mass).
    hoi_mass_range: tuple | None = None
    # Failure retries (failure_reset) also rewind motions with an object.
    object_retries: bool = False
    original_start_prob: float = 0.25
    park_distance: float = 20.0

"""Imagination interface of the HOI tracking task: the robot tracking task plus one dynamic object per environment.

The state appends the object and its asset index to the robot state (state.HOIStateSpec); per-object quantities
(dimensions, surface samples) are gathered by that index, so imagined steps need no environment ids. The actor and
critic additionally observe the 35-d object tokens of Supertracker's ``hoi_tracking_v1`` profile and, for robots with
hands, the hand-object BPS (also a world-model feature); the world model and critic condition on the object's static
descriptor (appended to xi). The rewards add object_motion and object_relation (object_heads.py); the object reward
uses the surrogate gate in imagination (see object_heads.tracking_components), the simulator the original one.
"""

from __future__ import annotations

import torch

from .kinematics import HumanoidKinematics
from .object_heads import ObjectHeadsCfg, contact_match, hand_bps, object_reward_terms, object_soft_termination, object_token
from .state import HOIStateSpec, HOIStateSpecCfg, object_index, read_object_state, read_state, split_state
from .task import ASSET_XML, TrackingImaginationTask


class HOIImaginationTask(TrackingImaginationTask):
    reward_term_names = TrackingImaginationTask.reward_term_names + ("object_motion", "object_relation")

    def __init__(self, env, object_cfg: ObjectHeadsCfg | None = None, **kwargs):
        super().__init__(env, **kwargs)
        nj = len(self.robot.joint_names)
        self.object_ids = env.object_ids
        self.dims_table = env.object_dims_table
        self.surface_table = env.object_surface_table
        self.use_hand_bps = bool(env.hand_link_ids)
        if self.use_hand_bps:
            # The hand links (wrists) are already among the tracked bodies: reuse bodies["pos"] / bodies["quat"]
            # instead of running a second FK pass every imagined step.
            body_names = list(self.command.cfg.body_names)
            self.hand_body_idx = torch.tensor([body_names.index(n) for n in env.cfg.hand_bps_links], dtype=torch.long, device=self.device)
            self.hand_basis = env.hand_bps_basis
        # Reference contact labels of the next frames (G1InspireSuperMimicHOIContactEnvCfg), actor and critic input.
        self.use_contact_labels = getattr(env.cfg.observations, "contact_ref", None) is not None
        if self.use_contact_labels:
            self.contact_offsets = torch.tensor(env.cfg.contact_label_steps, device=self.device)
            self.contact_num_bodies = len(self.lib.contact_body_names)
        # Reference contact matching reward of the contact links (G1InspireSuperMimicHOIContactRewardEnvCfg): the
        # simulator's term (weight, links) imagined on the predicted link and object poses.
        term = getattr(env.cfg.rewards, "object_contact", None)
        self.use_contact_reward = term is not None
        if self.use_contact_reward:
            self.contact_weight = float(term.weight)
            bodies, labels = list(self.command.cfg.body_names), self.lib.contact_body_names
            self.contact_body_idx = torch.tensor([bodies.index(n) for n in env.cfg.contact_links], device=self.device)
            self.contact_label_cols = torch.tensor([labels.index(n) for n in env.cfg.contact_links], device=self.device)
            self.contact_table = env.object_contact_table
            self.reward_term_names = self.reward_term_names + ("object_contact",)
        self.spec = HOIStateSpec(HOIStateSpecCfg(dt=env.step_dt), self.device, num_joints=nj)
        self.state_dim = self.spec.state_dim
        self.obj_slice = slice(self.spec.robot_dim, self.spec.robot_dim + 13)
        # The rotation termination of the simulator's object_pos term, softened like the position.
        rot = env.cfg.terminations.object_pos.params.get("rot_threshold", 0.0)
        self.obj_cfg = object_cfg or ObjectHeadsCfg(term_rot_threshold=rot)
        self.object = env.scene["object"]
        self.step_dt = env.step_dt
        # xi: the robot's simulation context followed by the static descriptor of the environment's object.
        self.xi_norm = torch.cat([self.xi_norm, env.object_desc_table[self.object_ids]], dim=-1)
        self.xi_dim = self.xi_norm.shape[-1]
        lib = self.lib
        if not lib.has_objects:
            raise ValueError("the HOI task needs a packed library with object fields")
        self.obj_ref = torch.cat([lib.object_pos_w, lib.object_quat_w, lib.object_lin_vel_w, lib.object_ang_vel_w], -1)

    # ------------------------------------------------------------------ simulator side

    def read_state(self) -> torch.Tensor:
        valid = self.command.lib.object_valid[self.command.motion_ids]
        robot = read_state(self.robot, self.env.scene.env_origins)
        obj = read_object_state(self.object, self.env.scene.env_origins, valid, object_index=self.object_ids)
        return torch.cat([robot, obj], dim=-1)

    def env_task_obs(self, obs):
        groups = ["policy_task", "object_actor"] + ["hand_bps"] * self.use_hand_bps + ["contact_ref"] * self.use_contact_labels
        return tuple(obs[g] for g in groups)

    def hand_bps_features(self, s: torch.Tensor, bodies: dict[str, torch.Tensor]) -> torch.Tensor:
        """(N, H*P*3) hand-object BPS from the already-computed tracked body poses and state s."""
        idx = self.hand_body_idx
        valid = s[:, self.spec.valid_index]
        surface = self.surface_table[object_index(s)]
        return hand_bps(bodies["pos"][:, idx], bodies["quat"][:, idx], self.hand_basis, s[:, self.obj_slice], surface, valid)

    # ------------------------------------------------------------------ imagination side

    def _anchor(self, bodies):
        return bodies["pos"][:, 0], bodies["quat"][:, 0], bodies["com_lin_vel"][:, 0], bodies["ang_vel"][:, 0]

    def _token(self, s, bodies, frame, offsets):
        ix = self.lib.future(frame, offsets)
        dt = (ix - frame[:, None]).to(s.dtype) * self.step_dt
        valid = s[:, self.spec.valid_index]
        dims = self.dims_table[object_index(s)]
        return object_token(self._anchor(bodies), s[:, self.obj_slice], self.obj_ref[ix], dt, dims, valid)

    def policy_task_obs(self, bodies, frame, roff, noise, s=None):
        robot = super().policy_task_obs(bodies, frame, roff, noise)
        offsets = torch.where(self.policy_offsets < 0, roff[:, None], self.policy_offsets[None, :])
        token = self._token(s, bodies, frame, offsets)
        out = [robot, token]
        if self.use_hand_bps:
            bps = torch.cat([self.hand_bps_features(s, bodies), s[:, self.spec.valid_index : self.spec.valid_index + 1]], -1)
            out.append(bps[:, None, None])
        if self.use_contact_labels:
            out.append(self.lib.contact_token(frame, self.contact_offsets, self.contact_num_bodies))
        return tuple(out)

    def previous_frame(self, frame: torch.Tensor) -> torch.Tensor:
        return torch.maximum(frame - 1, self.lib.starts[self.lib.frame_motion[frame]])

    def transition(self, s_next, bodies_next, action, prev_action, frame, s_prev=None):
        out = super().transition(s_next, bodies_next, action, prev_action, frame)
        valid = s_next[:, self.spec.valid_index]
        obj, obj_prev = s_next[:, self.obj_slice], s_prev[:, self.obj_slice]
        ref, ref_prev = self.obj_ref[frame], self.obj_ref[self.previous_frame(frame)]
        d_obj = object_soft_termination(self.obj_cfg, obj, ref, valid)
        done = 1.0 - (1.0 - out["done"]) * (1.0 - d_obj)
        # Survival pays for not terminating: recompute it with the combined termination.
        terms = dict(out["terms"])
        terms["survival"] = self.heads.cfg.w_alive * (1.0 - done) * self.heads.cfg.step_dt
        obj_terms = object_reward_terms(
            self.obj_cfg, obj_prev, obj, ref_prev, ref, bodies_next["pos"], self.lib.body_pos_w[frame], valid, surrogate=True
        )
        terms["object_motion"] = self.obj_cfg.w_motion * obj_terms["object_motion"] * self.heads.cfg.step_dt
        terms["object_relation"] = self.obj_cfg.w_relation * obj_terms["object_relation"] * self.heads.cfg.step_dt
        if self.use_contact_reward:
            terms["object_contact"] = self.contact_reward(s_next, bodies_next, frame)
        return {"reward": sum(terms.values()), "done": done, "truncated": out["truncated"], "terms": terms}

    def contact_reward(self, s, bodies, frame):
        """Weighted contact matching term of the batch of states s (objects looked up by the state's object index)."""
        labels = self.lib.contact_labels_at(frame, self.contact_label_cols) * s[:, self.spec.valid_index, None]
        points = self.contact_table[object_index(s)]
        score = contact_match(bodies["pos"][:, self.contact_body_idx], s[:, self.obj_slice], points, labels)
        return self.contact_weight * score * self.heads.cfg.step_dt

    def critic_features(self, s, bodies, frame, mode_ids, ctx) -> torch.Tensor:
        base = super().critic_features(s, bodies, frame, mode_ids, ctx)
        token = self._token(s, bodies, frame, self.critic_offsets.expand(frame.shape[0], -1))
        parts = [base, token.flatten(1)]
        if self.use_contact_labels:
            parts.append(self.lib.contact_token(frame, self.contact_offsets, self.contact_num_bodies).flatten(1))
        return torch.cat(parts, dim=-1)

    @property
    def critic_dim(self) -> int:
        contact = len(self.contact_offsets) * self.contact_num_bodies + 1 if self.use_contact_labels else 0
        return super().critic_dim + len(self.critic_offsets) * 35 + contact

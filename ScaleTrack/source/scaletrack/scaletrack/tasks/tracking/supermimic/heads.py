"""Analytic observation, reward and termination heads of the BFM tracking MDP.

These functions recompute, from a robot state and reference-motion tensors, exactly what the IsaacLab managers of
``G1-BFM-Transformer-Tracking`` compute in simulation (see tracking_env_cfg.py, mdp/observations.py,
mdp/rewards.py and mdp/terminations.py), so that the policy can be trained through imagined rollouts of a learned
world model. They are pure torch and differentiable.

Conventions:
    * states use the layout of state.py, joints in simulator order;
    * reference tensors hold the tracked bodies in command order (pelvis first), positions env-local;
    * the anchor body is the pelvis, i.e. the articulation root.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .kinematics import HumanoidKinematics
from .rotations import projected_gravity, quat_angle_sq, quat_apply_inverse, quat_conjugate, quat_mul, quat_to_tan_norm
from .state import split_state


@dataclass
class HeadsCfg:
    policy_future_idx: tuple[int, ...] = (0, 1, 2, 3, 4, -1)
    critic_future_idx: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)
    joint_vel_obs_scale: float = 0.05
    # Uniform noise half-widths of the policy observations (applied before scaling, as in IsaacLab).
    noise_gravity: float = 0.05
    noise_ang_vel: float = 0.2
    noise_joint_pos: float = 0.01
    noise_joint_vel: float = 0.5
    noise_task: float = 0.05
    # Reward weights and kernel widths (RewardsCfg); every term is multiplied by the control period.
    step_dt: float = 0.02
    w_height: float = 0.5
    w_pos: float = 1.0
    w_rot: float = 1.0
    w_lin_vel: float = 1.0
    w_ang_vel: float = 1.0
    w_action_rate: float = -0.1
    w_joint_limit: float = -10.0
    w_alive: float = 1.0
    std_height: float = 0.3
    std_pos: float = 0.3
    std_rot: float = 0.4
    std_lin_vel: float = 1.0
    std_ang_vel: float = 3.14
    # Termination: any tracked body farther than this from its reference.
    term_threshold: float = 0.5
    # Width of the sigmoid that softens the termination in imagination (m).
    term_temperature: float = 0.02
    task_feature_dims: tuple[int, ...] = field(default_factory=lambda: (3, 3, 6, 6))

    @property
    def max_step_reward(self) -> float:
        return (self.w_height + self.w_pos + self.w_rot + self.w_lin_vel + self.w_ang_vel + self.w_alive) * self.step_dt


@dataclass
class RefFrames:
    """Reference quantities of the tracked bodies, shape (..., num_bodies, 3/4)."""

    pos: torch.Tensor
    quat: torch.Tensor
    lin_vel: torch.Tensor | None = None
    ang_vel: torch.Tensor | None = None


class TrackingHeads:
    def __init__(
        self,
        cfg: HeadsCfg,
        kinematics: HumanoidKinematics,
        default_joint_pos: torch.Tensor,
        soft_joint_pos_limits: torch.Tensor,
        mode_table: torch.Tensor,
    ):
        """default_joint_pos (J,) nominal default pose; soft_joint_pos_limits (J, 2); mode_table (num_modes, num_bodies)."""
        self.cfg = cfg
        self.kin = kinematics
        self.q0 = default_joint_pos
        self.soft_limits = soft_joint_pos_limits
        self.mode_table = mode_table
        self.num_bodies = mode_table.shape[1]
        device = default_joint_pos.device
        nj = default_joint_pos.shape[0]
        self.prop_noise = torch.cat(
            [
                torch.full((3,), cfg.noise_gravity),
                torch.full((3,), cfg.noise_ang_vel),
                torch.full((nj,), cfg.noise_joint_pos),
                torch.full((nj,), cfg.noise_joint_vel * cfg.joint_vel_obs_scale),
            ]
        ).to(device)
        mappings = [mode_table[:, :, None].expand(-1, -1, d).reshape(mode_table.shape[0], -1) for d in cfg.task_feature_dims]
        mappings.append(torch.ones(mode_table.shape[0], 1, device=device))
        self.mode_mapping_table = torch.cat(mappings, dim=-1)
        self.task_noise = torch.cat(
            [torch.full((self.num_bodies * d,), cfg.noise_task) for d in cfg.task_feature_dims] + [torch.zeros(1)]
        ).to(device)

    # ------------------------------------------------------------------ kinematics

    def bodies(self, s: torch.Tensor, com_pos_b: torch.Tensor | None = None, with_vel: bool = True) -> dict[str, torch.Tensor]:
        p, q, qj, v, w, qd = split_state(s)
        if with_vel:
            return self.kin.forward(p, q, qj, v, w, qd, com_pos_b)
        return self.kin.forward(p, q, qj)

    # ------------------------------------------------------------------ observations

    def prop_obs(self, s: torch.Tensor, joint_offset: torch.Tensor) -> torch.Tensor:
        """Noise-free policy proprioception frame (..., 64): gravity, ang vel, joint pos rel, scaled joint vel."""
        _, q, qj, _, w, qd = split_state(s)
        return torch.cat(
            [projected_gravity(q), w, qj - (self.q0 + joint_offset), qd * self.cfg.joint_vel_obs_scale], dim=-1
        )

    def add_prop_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + (2.0 * torch.rand_like(obs) - 1.0) * self.prop_noise

    def task_obs(self, bodies: dict[str, torch.Tensor], ref: RefFrames, offsets: torch.Tensor) -> torch.Tensor:
        """Policy-task observation (..., F, 253) for reference frames ref (..., F, B, ·) at time offsets (..., F)."""
        a_pos = bodies["pos"][..., None, :1, :]  # anchor = pelvis, broadcast over frames and bodies
        a_quat = bodies["quat"][..., None, :1, :].expand_as(ref.quat)
        b_pos = bodies["pos"][..., None, :, :]
        b_quat = bodies["quat"][..., None, :, :].expand_as(ref.quat)
        pos = quat_apply_inverse(a_quat, ref.pos - a_pos)
        pos_rel = quat_apply_inverse(a_quat, ref.pos - b_pos)
        a_inv = quat_conjugate(a_quat)
        rot = quat_to_tan_norm(quat_mul(a_inv, ref.quat))
        rot_rel = quat_to_tan_norm(quat_mul(quat_mul(a_inv, quat_mul(ref.quat, quat_conjugate(b_quat))), a_quat))
        return torch.cat(
            [pos.flatten(-2), pos_rel.flatten(-2), rot.flatten(-2), rot_rel.flatten(-2), offsets[..., None].to(pos.dtype)],
            dim=-1,
        )

    def add_task_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + (2.0 * torch.rand_like(obs) - 1.0) * self.task_noise

    def critic_task_obs(self, bodies: dict[str, torch.Tensor], ref: RefFrames, offsets: torch.Tensor) -> torch.Tensor:
        """Privileged task observation (..., F, 337): task_obs terms plus body velocity errors in the anchor frame."""
        a_quat = bodies["quat"][..., None, :1, :].expand_as(ref.quat)
        vel_rel = quat_apply_inverse(a_quat, ref.lin_vel - bodies["com_lin_vel"][..., None, :, :])
        ang_vel_rel = quat_apply_inverse(a_quat, ref.ang_vel - bodies["ang_vel"][..., None, :, :])
        base = self.task_obs(bodies, ref, offsets)
        return torch.cat([base[..., :-1], vel_rel.flatten(-2), ang_vel_rel.flatten(-2), base[..., -1:]], dim=-1)

    # ------------------------------------------------------------------ reward / termination

    def body_errors(self, bodies: dict[str, torch.Tensor], ref: RefFrames) -> torch.Tensor:
        """Distance of every tracked body to its reference (..., B)."""
        return (ref.pos - bodies["pos"]).norm(dim=-1)

    def termination(self, bodies: dict[str, torch.Tensor], ref: RefFrames) -> torch.Tensor:
        return (self.body_errors(bodies, ref) > self.cfg.term_threshold).any(-1)

    def soft_termination(self, bodies: dict[str, torch.Tensor], ref: RefFrames) -> torch.Tensor:
        """Differentiable termination probability sigmoid((max_b e_b - threshold) / temperature).

        Uses the worst body like the hard termination; a product of per-body probabilities would terminate
        imagined episodes far more often than the simulator, as many bodies sit at moderate errors at once.
        """
        e = self.body_errors(bodies, ref).amax(-1)
        return torch.sigmoid((e - self.cfg.term_threshold) / self.cfg.term_temperature)

    def reward_terms(
        self,
        s_next: torch.Tensor,
        bodies: dict[str, torch.Tensor],
        action: torch.Tensor,
        prev_action: torch.Tensor,
        ref: RefFrames,
        done_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Reward r_{t+1} of the transition (s_t, a_t) -> s_{t+1} scored against the reference of step t."""
        c = self.cfg
        qj = split_state(s_next)[2]
        e_height = (ref.pos[..., 0, 2] - bodies["pos"][..., 0, 2]) ** 2
        e_pos = ((ref.pos - bodies["pos"]) ** 2).sum(-1).mean(-1)
        e_rot = quat_angle_sq(ref.quat, bodies["quat"]).mean(-1)
        e_lin = ((ref.lin_vel - bodies["com_lin_vel"]) ** 2).sum(-1).mean(-1)
        e_ang = ((ref.ang_vel - bodies["ang_vel"]) ** 2).sum(-1).mean(-1)
        lower, upper = self.soft_limits[:, 0], self.soft_limits[:, 1]
        out_of_limits = (lower - qj).clamp_min(0.0).sum(-1) + (qj - upper).clamp_min(0.0).sum(-1)
        terms = {
            "motion_body_height": c.w_height * torch.exp(-e_height / c.std_height**2),
            "motion_body_pos": c.w_pos * torch.exp(-e_pos / c.std_pos**2),
            "motion_body_rot": c.w_rot * torch.exp(-e_rot / c.std_rot**2),
            "motion_body_lin_vel": c.w_lin_vel * torch.exp(-e_lin / c.std_lin_vel**2),
            "motion_body_ang_vel": c.w_ang_vel * torch.exp(-e_ang / c.std_ang_vel**2),
            "action_rate_l2": c.w_action_rate * ((action - prev_action) ** 2).sum(-1),
            "joint_limit": c.w_joint_limit * out_of_limits,
            "survival": c.w_alive * (1.0 - done_prob),
        }
        return {k: v * c.step_dt for k, v in terms.items()}

    # ------------------------------------------------------------------ helpers

    def mode_mapping(self, mode_ids: torch.Tensor) -> torch.Tensor:
        return self.mode_mapping_table[mode_ids]

    def mode(self, mode_ids: torch.Tensor) -> torch.Tensor:
        return self.mode_table[mode_ids]

"""Robot state representation shared by the replay buffer and the world model.

Simulator state (71 = 3 + 4 + 29 + 3 + 3 + 29), joints in simulator order:

    s = [root_pos (env-local), root_quat (wxyz), joint_pos, root_lin_vel_w (link origin), root_ang_vel_b, joint_vel]

World-model input features (heading invariant, 71):

    [root_z, rot6d(heading^-1 * root_quat), root_lin_vel in heading frame, root_ang_vel_b, joint_pos, joint_vel]

The dynamics network outputs 70 scaled increments that are integrated explicitly (dt = control period):

    v' = v + R_h dv_h        w' = w + dw_b             qd' = qd + dqd
    p' = p + dt v' + R_h dp_h   q' = q * exp(dt w' + dth_b)   qj' = qj + dt qd' + dq

With a zero output the model extrapolates at constant velocity; this is the initialisation of the network.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_conjugate,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)

NUM_JOINTS = 29
STATE_DIM = 71
FEATURE_DIM = 71
DELTA_DIM = 70


def read_state(robot, env_origins: torch.Tensor, env_ids: torch.Tensor | slice = slice(None)) -> torch.Tensor:
    """Current state of an IsaacLab articulation as a (n, 71) tensor.

    The link-frame root velocity is derived here from the CoM velocity: ``Articulation.data.root_link_vel_w`` is
    cached per simulation step and still holds the pre-reset value right after ``write_root_state_to_sim``.
    """
    data = robot.data
    quat = data.root_link_quat_w[env_ids]
    com_vel = data.root_com_vel_w[env_ids]
    com_pos_w = quat_apply(quat, data.body_com_pos_b[env_ids, 0])
    lin_vel = com_vel[:, :3] - torch.cross(com_vel[:, 3:], com_pos_w, dim=-1)
    return torch.cat(
        [
            data.root_link_pos_w[env_ids] - env_origins[env_ids],
            quat,
            data.joint_pos[env_ids],
            lin_vel,
            quat_apply_inverse(quat, com_vel[:, 3:]),
            data.joint_vel[env_ids],
        ],
        dim=-1,
    )


def split_state(s: torch.Tensor):
    return s[..., 0:3], s[..., 3:7], s[..., 7:36], s[..., 36:39], s[..., 39:42], s[..., 42:71]


@dataclass
class StateSpecCfg:
    dt: float = 0.02
    # Nominal error scales for [root pos m, root rotvec rad, root lin vel m/s, root ang vel rad/s, joint pos, joint vel].
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # Output scales of the dynamics network for [dv_h, dw_b, dqd, dp_h, dth_b, dq].
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    # Input feature scales for [root_z, rot6d, lin_vel_h, ang_vel_b, joint_pos, joint_vel] (features are divided).
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)


class StateSpec:
    def __init__(self, cfg: StateSpecCfg, device, dtype=torch.float32):
        self.cfg = cfg
        self.dt = cfg.dt

        def rep(scales, sizes):
            return torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)]).to(device=device, dtype=dtype)

        nj = NUM_JOINTS
        self.err_scale = rep(cfg.err_scale, (3, 3, 3, 3, nj, nj))
        self.out_scale = rep(cfg.out_scale, (3, 3, nj, 3, 3, nj))
        self.in_scale = rep(cfg.in_scale, (1, 6, 3, 3, nj, nj))

    def features(self, s: torch.Tensor) -> torch.Tensor:
        p, q, qj, v, w, qd = split_state(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        q_local = quat_canonical(quat_mul(quat_conjugate(hq), q))
        v_h = quat_apply_inverse(hq, v)
        return torch.cat([p[..., 2:3], rot6d_from_quat(q_local), v_h, w, qj, qd], dim=-1) / self.in_scale

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        d = delta * self.out_scale
        dv_h, dw_b, dqd = d[..., 0:3], d[..., 3:6], d[..., 6:35]
        dp_h, dth_b, dq = d[..., 35:38], d[..., 38:41], d[..., 41:70]
        p, q, qj, v, w, qd = split_state(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        v_new = v + quat_apply(hq, dv_h)
        w_new = w + dw_b
        qd_new = qd + dqd
        p_new = p + self.dt * v_new + quat_apply(hq, dp_h)
        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(self.dt * w_new + dth_b)))
        qj_new = qj + self.dt * qd_new + dq
        return torch.cat([p_new, q_new, qj_new, v_new, w_new, qd_new], dim=-1)

    def error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Normalised 70-d error [dp, rotvec, dv, dw, dq, dqd] / nominal scale."""
        p1, q1, qj1, v1, w1, qd1 = split_state(s_pred)
        p2, q2, qj2, v2, w2, qd2 = split_state(s_true)
        rot = rotvec_from_quat(quat_mul(quat_conjugate(quat_normalize(q2)), quat_normalize(q1)))
        return torch.cat([p1 - p2, rot, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2], dim=-1) / self.err_scale

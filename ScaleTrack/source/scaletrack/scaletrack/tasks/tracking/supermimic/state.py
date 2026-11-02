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


def state_dim(num_joints: int = NUM_JOINTS) -> int:
    return 13 + 2 * num_joints


def split_state(s: torch.Tensor, num_joints: int = NUM_JOINTS):
    j = 7 + num_joints
    return s[..., 0:3], s[..., 3:7], s[..., 7:j], s[..., j : j + 3], s[..., j + 3 : j + 6], s[..., j + 6 : j + 6 + num_joints]


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
    def __init__(self, cfg: StateSpecCfg, device, dtype=torch.float32, num_joints: int = NUM_JOINTS):
        self.cfg = cfg
        self.dt = cfg.dt
        self.nj = num_joints
        self.state_dim = state_dim(num_joints)

        def rep(scales, sizes):
            return torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)]).to(device=device, dtype=dtype)

        nj = num_joints
        self.err_scale = rep(cfg.err_scale, (3, 3, 3, 3, nj, nj))
        self.out_scale = rep(cfg.out_scale, (3, 3, nj, 3, 3, nj))
        self.in_scale = rep(cfg.in_scale, (1, 6, 3, 3, nj, nj))

    def features(self, s: torch.Tensor) -> torch.Tensor:
        p, q, qj, v, w, qd = split_state(s, self.nj)
        q = quat_normalize(q)
        hq = heading_quat(q)
        q_local = quat_canonical(quat_mul(quat_conjugate(hq), q))
        v_h = quat_apply_inverse(hq, v)
        return torch.cat([p[..., 2:3], rot6d_from_quat(q_local), v_h, w, qj, qd], dim=-1) / self.in_scale

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        d = delta * self.out_scale
        nj = self.nj
        dv_h, dw_b, dqd = d[..., 0:3], d[..., 3:6], d[..., 6 : 6 + nj]
        dp_h, dth_b, dq = d[..., 6 + nj : 9 + nj], d[..., 9 + nj : 12 + nj], d[..., 12 + nj : 12 + 2 * nj]
        p, q, qj, v, w, qd = split_state(s, self.nj)
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
        p1, q1, qj1, v1, w1, qd1 = split_state(s_pred, self.nj)
        p2, q2, qj2, v2, w2, qd2 = split_state(s_true, self.nj)
        rot = rotvec_from_quat(quat_mul(quat_conjugate(quat_normalize(q2)), quat_normalize(q1)))
        return torch.cat([p1 - p2, rot, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2], dim=-1) / self.err_scale


# ---------------------------------------------------------------------- robot + one dynamic object (HOI)
#
# s = [robot state (71 for G1), object pos (env-local) 3, quat wxyz 4, lin_vel_w (origin) 3, ang_vel_w 3, valid 1,
#      object index 1] (86 for G1).
# The object integrates like the root, with increments in the robot heading frame and world angular velocity:
#     v' = v + R_h dv     w' = w + R_h dw     p' = p + dt v' + R_h dp     q' = exp(dt w' + R_h dth) * q
# ``valid`` (1 when the motion has an object) and the object index (which object asset the environment holds) are
# carried unchanged; invalid objects add no features and no error. Optional extra features (hand-object BPS) are
# computed from the state by a callable supplied by the task.

OBJ_SLICE = slice(STATE_DIM, STATE_DIM + 13)
HOI_STATE_DIM = STATE_DIM + 13 + 2


def read_object_state(
    obj, env_origins: torch.Tensor, valid: torch.Tensor, env_ids: torch.Tensor | slice = slice(None), object_index=None
):
    """(n, 15) object state of an IsaacLab RigidObject (link origin velocity from the CoM velocity, see read_state)."""
    data = obj.data
    quat = data.root_link_quat_w[env_ids]
    com_vel = data.root_com_vel_w[env_ids]
    com_pos_w = quat_apply(quat, data.body_com_pos_b[env_ids, 0])
    lin_vel = com_vel[:, :3] - torch.cross(com_vel[:, 3:], com_pos_w, dim=-1)
    return torch.cat(
        [
            data.root_link_pos_w[env_ids] - env_origins[env_ids], quat, lin_vel, com_vel[:, 3:],
            valid[env_ids, None].float(),
            torch.zeros_like(quat[:, :1]) if object_index is None else object_index[env_ids, None].float(),
        ],
        dim=-1,
    )  # fmt: skip


def split_object_state(s: torch.Tensor, robot_dim: int = STATE_DIM):
    o = s[..., robot_dim : robot_dim + 14]
    return o[..., 0:3], o[..., 3:7], o[..., 7:10], o[..., 10:13], o[..., 13:14]


def object_index(s: torch.Tensor) -> torch.Tensor:
    """Index of the object asset of every state (long)."""
    return s[..., -1].round().long()


@dataclass
class HOIStateSpecCfg(StateSpecCfg):
    # [object pos m, rotvec rad, lin vel m/s, ang vel rad/s]
    obj_err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0)
    # [dv, dw, dp, dth]
    obj_out_scale: tuple[float, ...] = (0.3, 0.5, 0.01, 0.02)
    # [relative pos, rot6d, height, lin vel, ang vel, valid]
    obj_in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 2.0, 1.0)


class HOIStateSpec:
    """StateSpec of the robot (delegated, unchanged) followed by one free rigid object."""

    def __init__(
        self, cfg: HOIStateSpecCfg, device, dtype=torch.float32, num_joints: int = NUM_JOINTS, extra_features=None,
        extra_dim: int = 0,
    ):  # fmt: skip
        self.cfg = cfg
        self.dt = cfg.dt
        self.robot = StateSpec(cfg, device, dtype, num_joints)
        self.robot_dim = self.robot.state_dim
        self.state_dim = self.robot_dim + 15
        self.valid_index = self.robot_dim + 13
        self.extra_features = extra_features

        def rep(scales, sizes):
            return torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)]).to(device=device, dtype=dtype)

        self.robot_out_dim = self.robot.out_scale.numel()
        self.err_scale = torch.cat([self.robot.err_scale, rep(cfg.obj_err_scale, (3, 3, 3, 3))])
        self.out_scale = torch.cat([self.robot.out_scale, rep(cfg.obj_out_scale, (3, 3, 3, 3))])
        self.in_scale = torch.cat(
            [self.robot.in_scale, rep(cfg.obj_in_scale, (3, 6, 1, 3, 3, 1)), torch.ones(extra_dim, device=device, dtype=dtype)]
        )
        self.obj_in_scale = self.in_scale[self.robot.in_scale.numel() : self.in_scale.numel() - extra_dim]
        self.obj_err_scale = self.err_scale[self.robot.err_scale.numel() :]

    def features(self, s: torch.Tensor) -> torch.Tensor:
        p, q = s[..., 0:3], quat_normalize(s[..., 3:7])
        hq = heading_quat(q)
        op, oq, ov, ow, valid = split_object_state(s, self.robot_dim)
        obj = torch.cat(
            [
                quat_apply_inverse(hq, op - p),
                rot6d_from_quat(quat_canonical(quat_mul(quat_conjugate(hq), quat_normalize(oq)))),
                op[..., 2:3],
                quat_apply_inverse(hq, ov),
                quat_apply_inverse(hq, ow),
            ],
            dim=-1,
        )
        obj = torch.cat([obj * valid, valid], dim=-1) / self.obj_in_scale
        parts = [self.robot.features(s[..., : self.robot_dim]), obj]
        if self.extra_features is not None:
            parts.append(self.extra_features(s))
        return torch.cat(parts, dim=-1)

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        robot = self.robot.integrate(s[..., : self.robot_dim], delta[..., : self.robot_out_dim])
        d = delta[..., self.robot_out_dim :] * self.out_scale[self.robot_out_dim :]
        hq = heading_quat(quat_normalize(s[..., 3:7]))
        op, oq, ov, ow, valid = split_object_state(s, self.robot_dim)
        v_new = ov + quat_apply(hq, d[..., 0:3])
        w_new = ow + quat_apply(hq, d[..., 3:6])
        p_new = op + self.dt * v_new + quat_apply(hq, d[..., 6:9])
        q_new = quat_normalize(quat_mul(quat_from_rotvec(self.dt * w_new + quat_apply(hq, d[..., 9:12])), quat_normalize(oq)))
        return torch.cat([robot, p_new, q_new, v_new, w_new, s[..., self.robot_dim + 13 :]], dim=-1)

    def error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Normalised 82-d error: robot (70) and object [dp, rotvec, dv, dw] (12), the object masked by validity."""
        robot = self.robot.error(s_pred[..., : self.robot_dim], s_true[..., : self.robot_dim])
        p1, q1, v1, w1, _ = split_object_state(s_pred, self.robot_dim)
        p2, q2, v2, w2, valid = split_object_state(s_true, self.robot_dim)
        rot = rotvec_from_quat(quat_mul(quat_conjugate(quat_normalize(q2)), quat_normalize(q1)))
        obj = torch.cat([p1 - p2, rot, v1 - v2, w1 - w2], dim=-1) * valid / self.obj_err_scale
        return torch.cat([robot, obj], dim=-1)

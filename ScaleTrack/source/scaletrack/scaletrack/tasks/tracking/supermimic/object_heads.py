"""Analytic object observations, rewards, termination and evaluation criteria of the HOI tracking task.

Ported from Supertracker's ``hoi_tracking_v1`` profile (supertracker/hoi_env.py, hoi_reward.py, hoi_local.py,
hoi_metrics.py) for one dynamic rigid object per environment. The functions are pure torch and differentiable, and
are shared by the simulator MDP terms (object_mdp.py) and the imagination task (task.HOIImaginationTask).

Object state (13): position (env-local), orientation wxyz, linear velocity of the body origin (world), angular
velocity (world). The reference trajectory uses the same convention (linear velocity = finite difference of the
origin position). ``valid`` (...,) is 1 for motions with an object and 0 otherwise; every output is exactly 0 where
it is 0.

Frame convention: rewards score the state after a control step against the reference frame of that step (the
``global_frame`` of the motion command), so the reference increment of a step is ref(frame) - ref(frame - 1), and the
physical increment is the object pose after minus before the step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rotations import quat_angle_sq, quat_apply, quat_apply_inverse, quat_conjugate, quat_mul, rotvec_from_quat

OBJECT_STATE_DIM = 13
OBJECT_TOKEN_DIM = 35
# Translation-equivalent length of a rotation (m / rad), shared by the reward and the evaluation credit.
ROTATION_LENGTH = 0.15


@dataclass
class ObjectHeadsCfg:
    w_motion: float = 2.0
    w_relation: float = 0.5
    # Termination: object farther than this from its reference (m), softened in imagination like the bodies.
    term_threshold: float = 0.5
    term_temperature: float = 0.02
    # Also terminate when the object turns farther than this from its reference (rad; 0 disables): a toppled pole
    # whose centre stays within term_threshold.
    term_rot_threshold: float = 0.0
    term_rot_temperature: float = 0.05
    # Full-clip completion (evaluation only).
    eval_max_pos_error: float = 0.2
    eval_max_rot_error: float = 0.8
    eval_min_motion_credit: float = 0.5
    lift_height: float = 0.08


def split_object(o: torch.Tensor):
    return o[..., 0:3], o[..., 3:7], o[..., 7:10], o[..., 10:13]


def rot6d_rows(q: torch.Tensor) -> torch.Tensor:
    """``matrix_from_quat(q)[..., :2].reshape(6)``: the first two columns of R flattened row by row."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
            2 * (x * z - w * y), 2 * (y * z + w * x),
        ],
        dim=-1,
    )  # fmt: skip


def robot_frame_object(anchor_pos, anchor_quat, anchor_lin_vel, anchor_ang_vel, p, q, v, w) -> torch.Tensor:
    """Object poses (..., T, ·) relative to the robot anchor (..., ·): pos 3, rot6 6, vel 3, ang vel 3 = 15.

    Same as supertracker.hoi_local.robot_frame_object; the anchor velocity is the pelvis CoM velocity
    (``body_lin_vel_w``).
    """
    aq = anchor_quat[..., None, :].expand(q.shape)
    return torch.cat(
        [
            quat_apply_inverse(aq, p - anchor_pos[..., None, :]),
            rot6d_rows(quat_mul(quat_conjugate(aq), q)),
            quat_apply_inverse(aq, v - anchor_lin_vel[..., None, :]),
            quat_apply_inverse(aq, w - anchor_ang_vel[..., None, :]),
        ],
        dim=-1,
    )


def object_token(anchor, obj: torch.Tensor, ref: torch.Tensor, dt: torch.Tensor, dims: torch.Tensor, valid: torch.Tensor):
    """Object token (N, 1, T, 35): actual 15 (repeated over T), reference 15, time offset 1 (s), dimensions 3, valid 1.

    anchor: (pos, quat, lin_vel, ang_vel) of the robot anchor, each (N, ·); obj (N, 13) current object state;
    ref (N, T, 13) reference object states at the future frames; dt (N, T) time offsets of these frames (s);
    dims (3,) or (N, 3) object dimensions; valid (N,).
    """
    T = ref.shape[-2]
    actual = robot_frame_object(*anchor, *(x[..., None, :] for x in split_object(obj)))
    target = robot_frame_object(*anchor, *split_object(ref))
    mask = valid[:, None, None].to(ref.dtype)
    token = torch.cat(
        [actual.expand(-1, T, -1), target, dt[..., None], dims[..., None, :].expand(*target.shape[:-1], 3) if dims.dim() == 2
         else dims.expand(*target.shape[:-1], 3), mask.expand(-1, T, 1)],
        -1,
    )
    return torch.where(mask > 0.5, token, torch.zeros_like(token))[:, None]


def increment(p_prev, q_prev, p, q):
    """Position increment and world-frame rotation vector of q * q_prev^-1."""
    return p - p_prev, rotvec_from_quat(quat_mul(q, quat_conjugate(q_prev)))


def tracking_components(dp, dr, target_dp, target_dr, pos_err, rot_err, vel_err, ang_vel_err, surrogate: bool = False):
    """supertracker.hoi_reward.tracking_components: object tracking paid only for physical motion along the reference.

    The simulator reward gates on both the reference and the object moving, so a resting object earns exactly 0,
    and so does its gradient. ``surrogate=True`` (imagination) keeps only the reference gate: the values agree
    wherever the object moves, while a resting object gets the gradient towards moving along the reference.
    """
    actual = torch.cat([dp, ROTATION_LENGTH * dr], -1)
    target = torch.cat([target_dp, ROTATION_LENGTH * target_dr], -1)
    energy = target.square().sum(-1)
    alignment = (actual * target).sum(-1)
    active = energy > 1e-8
    if not surrogate:
        active = active & (actual.square().sum(-1) > 1e-8)
    progress = torch.where(active, (alignment / energy.clamp_min(1e-12)).clamp(0, 1), torch.zeros_like(energy))
    position = 0.5 * torch.exp(-(pos_err / 0.4).square()) + 0.5 * torch.exp(-(pos_err / 0.15).square())
    rotation = torch.exp(-(rot_err / 0.4).square())
    velocity = torch.exp(-(vel_err / 1.0).square())
    angular = torch.exp(-(ang_vel_err / 3.14).square())
    score = progress * position * (0.5 + 0.25 * rotation + 0.125 * velocity + 0.125 * angular)
    return {"score": score, "progress": progress, "position": position}


def object_errors(obj: torch.Tensor, ref: torch.Tensor) -> dict[str, torch.Tensor]:
    p, q, v, w = split_object(obj)
    rp, rq, rv, rw = split_object(ref)
    return {
        "position": (p - rp).norm(dim=-1),
        "rotation": torch.sqrt(quat_angle_sq(q, rq) + 1e-12),
        "linear_velocity": (v - rv).norm(dim=-1),
        "angular_velocity": (w - rw).norm(dim=-1),
    }


def relation_error(body_pos, ref_body_pos, obj: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Weighted error of the tracked bodies (N, B, 3) in the object frame vs. the reference bodies in the reference
    object frame; bodies near the reference object weigh most (supertracker.hoi_env.object_relation_reward)."""
    p, q = obj[..., None, 0:3], obj[..., None, 3:7]
    rp, rq = ref[..., None, 0:3], ref[..., None, 3:7]
    actual = quat_apply_inverse(q.expand(*body_pos.shape[:-1], 4), body_pos - p)
    reference = quat_apply_inverse(rq.expand(*ref_body_pos.shape[:-1], 4), ref_body_pos - rp)
    weight = torch.exp(-reference.square().sum(-1) / 0.4**2)
    return ((actual - reference).square().sum(-1) * weight).sum(-1) / weight.sum(-1).clamp_min(1e-8)


def object_reward_terms(
    cfg: ObjectHeadsCfg,
    obj_prev: torch.Tensor,
    obj: torch.Tensor,
    ref_prev: torch.Tensor,
    ref: torch.Tensor,
    body_pos: torch.Tensor,
    ref_body_pos: torch.Tensor,
    valid: torch.Tensor,
    surrogate: bool = False,
) -> dict[str, torch.Tensor]:
    """Unweighted object_motion and object_relation rewards of the step obj_prev -> obj against ref_prev -> ref."""
    dp, dr = increment(obj_prev[..., 0:3], obj_prev[..., 3:7], obj[..., 0:3], obj[..., 3:7])
    tdp, tdr = increment(ref_prev[..., 0:3], ref_prev[..., 3:7], ref[..., 0:3], ref[..., 3:7])
    e = object_errors(obj, ref)
    c = tracking_components(dp, dr, tdp, tdr, e["position"], e["rotation"], e["linear_velocity"], e["angular_velocity"], surrogate)
    relation = c["progress"] * c["position"] * torch.exp(-relation_error(body_pos, ref_body_pos, obj, ref) / 0.15**2)
    zero = torch.zeros_like(c["score"])
    on = valid > 0.5
    return {"object_motion": torch.where(on, c["score"], zero), "object_relation": torch.where(on, relation, zero)}


def object_soft_termination(cfg: ObjectHeadsCfg, obj: torch.Tensor, ref: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    e = (obj[..., 0:3] - ref[..., 0:3]).norm(dim=-1)
    d = torch.sigmoid((e - cfg.term_threshold) / cfg.term_temperature)
    if cfg.term_rot_threshold > 0:
        r = torch.sqrt(quat_angle_sq(obj[..., 3:7], ref[..., 3:7]) + 1e-12)
        d = 1.0 - (1.0 - d) * (1.0 - torch.sigmoid((r - cfg.term_rot_threshold) / cfg.term_rot_temperature))
    return valid * d


# ---------------------------------------------------------------------- evaluation (supertracker.hoi_metrics)


def motion_credit(dp, dr, target_dp, target_dr):
    """Reference motion reproduced in its direction this step (capped at the reference step length), and that length."""
    actual = torch.cat([dp, ROTATION_LENGTH * dr], -1)
    target = torch.cat([target_dp, ROTATION_LENGTH * target_dr], -1)
    length = target.norm(dim=-1)
    moving = length > 1e-4
    projection = (actual * target).sum(-1) / length.clamp_min(1e-12)
    credit = torch.where(moving, torch.minimum(projection.clamp_min(0), length), torch.zeros_like(length))
    return credit, torch.where(moving, length, torch.zeros_like(length))


def collider_bottom(points: torch.Tensor, obj: torch.Tensor) -> torch.Tensor:
    """Lowest height of the collision hull points (object frame; (P, 3) shared or (..., P, 3) per state) at object
    states (..., 13)."""
    q = obj[..., None, 3:7].expand(*obj.shape[:-1], points.shape[-2], 4)
    return quat_apply(q, points.expand(*obj.shape[:-1], -1, -1))[..., 2].amin(-1) + obj[..., 2]


# ---------------------------------------------------------------------- hand-object perception (BPS)

HAND_BPS_POINTS = 32
HAND_BPS_RADIUS = 0.15  # m, basis ball around the palm
HAND_BPS_CLIP = 0.3  # m, vectors to farther objects are clipped


def hand_bps_basis(center, num_points: int = HAND_BPS_POINTS, radius: float = HAND_BPS_RADIUS, seed: int = 0):
    """Fixed basis points (num_points, 3) in a ball around ``center`` (hand-link frame)."""
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(num_points, 3, generator=g)
    d = d / d.norm(dim=-1, keepdim=True) * radius * torch.rand(num_points, 1, generator=g) ** (1.0 / 3.0)
    return d + torch.as_tensor(center, dtype=torch.float32)


# ---------------------------------------------------------------------- reference contact matching

CONTACT_MARGIN = 0.025  # m, link origin to the finger surface: closer counts as full contact
CONTACT_SIGMA = 0.02  # m, Gaussian fall-off of the contact strength beyond the margin


def contact_match(link_pos, obj, points, labels, margin: float = CONTACT_MARGIN, sigma: float = CONTACT_SIGMA):
    """HDMI's contact_match_score with a differentiable distance proxy for the contact force.

    link_pos (N, L, 3) world, obj (N, 13) object state, points (N, P, 3) object-frame surface samples, labels (N, L)
    in {+1 expected contact, -1 forbidden, 0 ignored}. The strength exp(-((d - margin)+ / sigma)^2) of the distance d
    from a link to the nearest surface point replaces clamp(force / saturate_force); the score is the mean over the
    labelled links of 1 - |[label > 0] - strength| (0 when no link is labelled)."""
    local = quat_apply_inverse(obj[:, None, 3:7].expand(-1, link_pos.shape[1], 4), link_pos - obj[:, None, :3])
    with torch.no_grad():
        nearest = ((local[:, :, None, :] - points[:, None, :, :]) ** 2).sum(-1).argmin(-1)
    surface = torch.gather(points, 1, nearest[..., None].expand(-1, -1, 3))
    d = (local - surface).norm(dim=-1)
    strength = torch.exp(-(((d - margin).clamp_min(0.0) / sigma) ** 2))
    mask = (labels != 0).float()
    score = 1.0 - ((labels > 0).float() - strength).abs()
    return (score * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)


def hand_bps(hand_pos, hand_quat, basis, obj, surface, valid, clip: float = HAND_BPS_CLIP):
    """Vectors from every hand basis point to its nearest object surface point, in the hand frame.

    hand_pos / hand_quat (N, H, 3/4) hand-link poses; basis (P, 3) basis points (hand frame); obj (N, 13) object state;
    surface (N, S, 3) object surface points (object frame); valid (N,). Returns (N, H*P*3), 0 where not valid, each
    vector rescaled to at most ``clip`` long. Distances are evaluated in the object frame so only H*P basis points are
    transformed and coordinates stay near zero; the nearest index is chosen without gradient.
    """
    N, H = hand_pos.shape[:2]
    P = basis.shape[0]
    world_basis = hand_pos[:, :, None] + quat_apply(hand_quat[:, :, None].expand(N, H, P, 4), basis.expand(N, H, P, 3))
    rel = (world_basis - obj[:, None, None, 0:3]).reshape(N, H * P, 3)
    oq = obj[:, None, 3:7].expand(N, H * P, 4)
    obj_basis = quat_apply_inverse(oq, rel)
    with torch.no_grad():
        d2 = (
            obj_basis.square().sum(-1, keepdim=True)
            - 2.0 * torch.bmm(obj_basis, surface.transpose(1, 2))
            + surface.square().sum(-1)[:, None]
        )
        idx = d2.argmin(-1)
    near_obj = torch.gather(surface, 1, idx[..., None].expand(N, H * P, 3))
    near_w = quat_apply(oq, near_obj - obj_basis).reshape(N, H, P, 3)
    local = quat_apply_inverse(hand_quat[:, :, None].expand(N, H, P, 4), near_w)
    norm = local.norm(dim=-1, keepdim=True)
    local = local * (clip / norm.clamp_min(clip))
    return torch.where(valid[:, None] > 0.5, local.flatten(1), torch.zeros_like(local.flatten(1)))

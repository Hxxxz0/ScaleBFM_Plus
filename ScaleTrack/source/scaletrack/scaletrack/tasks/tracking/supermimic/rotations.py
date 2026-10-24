"""Batched, differentiable rotation helpers (quaternions in wxyz, same convention as isaaclab.utils.math).

Unlike some IsaacLab helpers, every function here has finite gradients at the identity rotation, which matters
because the imagined rollouts back-propagate through rotation errors that are often exactly zero.
"""

from __future__ import annotations

import math

import torch

_EPS = 1.0e-8


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_canonical(q: torch.Tensor) -> torch.Tensor:
    """Flip the sign so that w >= 0."""
    return torch.where(q[..., :1] < 0, -q, q)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
    """Exponential map (rotation vector -> unit quaternion) with a series expansion near zero."""
    angle_sq = (v * v).sum(-1, keepdim=True)
    angle = torch.sqrt(angle_sq.clamp_min(1e-30))
    small = angle < 1e-4
    k = torch.where(small, 0.5 - angle_sq / 48.0, torch.sin(0.5 * angle) / angle)
    w = torch.where(small, 1.0 - angle_sq / 8.0, torch.cos(0.5 * angle))
    return torch.cat([w, v * k], dim=-1)


def rotvec_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Logarithmic map (unit quaternion -> rotation vector in (-pi, pi]) with a series expansion near zero."""
    q = quat_canonical(q)
    w, xyz = q[..., :1], q[..., 1:]
    s = torch.sqrt((xyz * xyz).sum(-1, keepdim=True).clamp_min(1e-30))
    angle = 2.0 * torch.atan2(s, w)
    small = s < 1e-6
    k = torch.where(small, 2.0 + angle * angle / 12.0, angle / torch.where(small, torch.ones_like(s), s))
    return xyz * k


def quat_angle_sq(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Squared rotation angle between q1 and q2 (same value as isaaclab quat_error_magnitude(q1, q2) ** 2)."""
    d = quat_mul(q1, quat_conjugate(q2))
    s = torch.sqrt((d[..., 1:] ** 2).sum(-1) + 1e-12)
    angle = 2.0 * torch.atan2(s, d[..., 0].abs())
    return angle * angle


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """axis (..., 3) unit vectors, angle (...,) -> wxyz quaternion."""
    half = 0.5 * angle
    return torch.cat([torch.cos(half)[..., None], axis * torch.sin(half)[..., None]], dim=-1)


def heading_yaw(q: torch.Tensor) -> torch.Tensor:
    """Yaw of the body x-axis projected on the ground (same as scaletrack.utils.torch_utils.calc_heading)."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def heading_quat(q: torch.Tensor) -> torch.Tensor:
    return quat_from_yaw(heading_yaw(q))


def rot6d_from_quat(q: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix, flattened (..., 6)."""
    w, x, y, z = q.unbind(-1)
    c0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1)
    c1 = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], dim=-1)
    return torch.cat([c0, c1], dim=-1)


def quat_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
    """Rotated x axis and rotated z axis (same as scaletrack.utils.torch_utils.quat_to_tan_norm)."""
    w, x, y, z = q.unbind(-1)
    tan = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1)
    norm = torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1)
    return torch.cat([tan, norm], dim=-1)


def projected_gravity(q: torch.Tensor) -> torch.Tensor:
    """Gravity direction (0, 0, -1) expressed in the frame q."""
    g = torch.zeros(*q.shape[:-1], 3, dtype=q.dtype, device=q.device)
    g[..., 2] = -1.0
    return quat_apply_inverse(q, g)


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

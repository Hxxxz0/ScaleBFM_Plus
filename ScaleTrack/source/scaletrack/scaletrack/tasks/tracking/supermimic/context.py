"""Simulation context xi: the per-environment physical parameters randomised at start-up.

The world model is deterministic and conditions on xi, so that its predictions can depend on the friction,
calibration offsets and mass properties of each simulated robot. All randomisation of the tracking task happens
at start-up, so xi is a fixed (num_envs, XI_DIM) table read once after the environment has been created:

    [left foot static / dynamic friction, right foot static / dynamic friction,   4
     joint default-position offsets (simulator joint order),                      29
     torso CoM offset,                                                            3
     left / right wrist added mass]                                               2
"""

from __future__ import annotations

import torch

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
COM_BODY = "torso_link"
MASS_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
XI_DIM = 4 + 29 + 3 + 2

# Normalisation (centre, half range) matching the event ranges of tracking_env_cfg.EventCfg.
XI_OFFSET = [0.95, 0.75, 0.95, 0.75] + [0.0] * 29 + [0.0] * 3 + [0.5, 0.5]
XI_SCALE = [0.65, 0.45, 0.65, 0.45] + [0.01] * 29 + [0.025, 0.05, 0.05] + [0.5, 0.5]


def _shapes_per_body(robot) -> list[int]:
    counts = []
    for link_path in robot.root_physx_view.link_paths[0]:
        counts.append(robot._physics_sim_view.create_rigid_body_view(link_path).max_shapes)
    if sum(counts) != robot.root_physx_view.max_shapes:
        raise RuntimeError("failed to resolve the collision shapes of each body")
    return counts


def read_sim_context(robot, device) -> torch.Tensor:
    """Read xi of every environment from the simulator (after the start-up events ran)."""
    counts = _shapes_per_body(robot)
    materials = robot.root_physx_view.get_material_properties().to(device)  # (N, shapes, 3)
    parts = []
    for name in FOOT_BODIES:
        b = robot.body_names.index(name)
        start = sum(counts[:b])
        parts.append(materials[:, start : start + counts[b], :2].mean(1))
    data = robot.data
    nominal = getattr(data, "default_joint_pos_nominal", data.default_joint_pos[0])
    parts.append(data.default_joint_pos - nominal)
    parts.append(getattr(data, "com_offset_randomization", torch.zeros(robot.num_instances, 3, device=device)).to(device))
    masses = robot.root_physx_view.get_masses().to(device)
    ids = [robot.body_names.index(n) for n in MASS_BODIES]
    parts.append(masses[:, ids] - data.default_mass.to(device)[:, ids])
    return torch.cat(parts, dim=-1)


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    offset = torch.tensor(XI_OFFSET, device=xi.device, dtype=xi.dtype)
    scale = torch.tensor(XI_SCALE, device=xi.device, dtype=xi.dtype)
    return (xi - offset) / scale

"""Simulator observation, reward and termination terms of the HOI tracking task (supertracker hoi_tracking_v1).

The formulas are those of object_heads.py, evaluated on the simulated object; the imagination task computes the
same quantities from world-model states (scripts/supermimic/check_parity.py compares them).
"""

from __future__ import annotations

import torch

from .object_heads import ObjectHeadsCfg, contact_match, hand_bps, object_errors, object_reward_terms, object_token

_CFG = ObjectHeadsCfg()


def _anchor(env, command):
    origin = env.scene.env_origins
    return (
        command.robot_anchor_pos_w - origin,
        command.robot_anchor_quat_w,
        command.robot_anchor_lin_vel_w,
        command.robot_anchor_ang_vel_w,
    )


def object_track(env, command_name: str, future_idx: list[int]) -> torch.Tensor:
    """(N, 1, T, 35) object token at the reference frames ``future_idx`` (negative: the random far frame)."""
    command = env.command_manager.get_term(command_name)
    ix = command._global_time_indices_manual(future_idx)
    dt = (ix - command._global_time_index()[:, None]).float() * env.step_dt
    token = object_token(
        _anchor(env, command), env.object_state(), command.object_reference(ix), dt, env.object_dims,
        command.object_valid.float(),
    )
    return token


def _object_terms(env, command_name: str) -> dict[str, torch.Tensor]:
    key = (env.common_step_counter, "hoi_object_terms")
    cache = getattr(env, "_hoi_reward_cache", None)
    if cache is not None and cache[0] == key:
        return cache[1]
    command = env.command_manager.get_term(command_name)
    frames = command._global_time_index()
    terms = object_reward_terms(
        _CFG,
        env.object_prev,
        env.object_state(),
        command.object_reference(command.previous_frame(frames)),
        command.object_reference(frames),
        command.robot_body_pos_w - env.scene.env_origins[:, None],
        command.body_pos_w - env.scene.env_origins[:, None],
        command.object_valid.float(),
    )
    env._hoi_reward_cache = (key, terms)
    return terms


def object_motion(env, command_name: str) -> torch.Tensor:
    return _object_terms(env, command_name)["object_motion"]


def object_relation(env, command_name: str) -> torch.Tensor:
    return _object_terms(env, command_name)["object_relation"]


def object_pos_failure(
    env, command_name: str, threshold: float, disable_flag: bool = False, rot_threshold: float = 0.0
) -> torch.Tensor:
    """Object farther than threshold (m) from its reference, or turned farther than rot_threshold (rad, 0: off)."""
    if disable_flag:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    command = env.command_manager.get_term(command_name)
    errors = object_errors(env.object_state(), command.object_reference(command._global_time_index()))
    failed = errors["position"] > threshold
    if rot_threshold > 0:
        failed |= errors["rotation"] > rot_threshold
    return command.object_valid & failed


def physical_invalid(env, command_name: str) -> torch.Tensor:
    """Non-finite or exploding robot / object state (caught before an automatic reset hides it)."""
    command = env.command_manager.get_term(command_name)
    robot = torch.cat([command.robot_body_pos_w.flatten(1), command.robot_anchor_lin_vel_w, command.robot_anchor_ang_vel_w], -1)
    obj = env.object_state()
    invalid = ~torch.isfinite(robot).all(-1)
    invalid |= command.robot_anchor_lin_vel_w.norm(dim=-1) > 100.0
    invalid |= command.object_valid & ~torch.isfinite(obj).all(-1)
    invalid |= command.object_valid & (obj[:, 7:10].norm(dim=-1) > 100.0)
    return invalid


def contact_links(env, command):
    """Tracked-body indices of env.cfg.contact_links and their columns in the library's contact labels (cached)."""
    if getattr(env, "_contact_links", None) is None:
        bodies, labels = list(command.cfg.body_names), command.train_lib.contact_body_names
        env._contact_links = (
            torch.tensor([bodies.index(n) for n in env.cfg.contact_links], device=env.device),
            torch.tensor([labels.index(n) for n in env.cfg.contact_links], device=env.device),
        )
    return env._contact_links


def object_contact(env, command_name: str) -> torch.Tensor:
    """Reference contact matching of the contact links with the object (object_heads.contact_match, HDMI's
    contact_matching with a distance proxy for the force)."""
    command = env.command_manager.get_term(command_name)
    body_ids, cols = contact_links(env, command)
    pos = command.robot_body_pos_w[:, body_ids] - env.scene.env_origins[:, None]
    labels = command.lib.contact_labels_at(command._global_time_index(), cols) * command.object_valid.float()[:, None]
    return contact_match(pos, env.object_state(), env.object_contact_points, labels)


def contact_label_obs(env, command_name: str) -> torch.Tensor:
    """(N, 1, 1, F*B + 1) reference contact labels at the steps env.cfg.contact_label_steps (HDMI's
    reference_contact_labels), validity last (motions without labels: invalid)."""
    command = env.command_manager.get_term(command_name)
    offsets = torch.tensor(env.cfg.contact_label_steps, device=env.device)
    return command.lib.contact_token(command._global_time_index(), offsets, len(command.train_lib.contact_body_names))


def hand_bps_obs(env, command_name: str) -> torch.Tensor:
    """(N, 1, 1, H*P*3 + 1) hand-object BPS (object_heads.hand_bps) with the object validity last."""
    command = env.command_manager.get_term(command_name)
    data = env.scene["robot"].data
    ids = env.hand_link_ids
    pos = data.body_link_pos_w[:, ids] - env.scene.env_origins[:, None]
    valid = command.object_valid.float()
    bps = hand_bps(pos, data.body_link_quat_w[:, ids], env.hand_bps_basis, env.object_state(), env.object_surface, valid)
    return torch.cat([bps, valid[:, None]], dim=-1)[:, None, None]

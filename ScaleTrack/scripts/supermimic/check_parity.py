"""Check that the analytic heads of the SuperMimic task reproduce the IsaacLab managers of the tracking task.

Steps the SuperMimic environment with random actions (no observation / action noise) and compares, every step:
  * FK of the state against the articulation's link poses and CoM velocities,
  * the analytic policy proprioception / task observation / mode inputs against the observation manager,
  * the 3-frame observation and action histories rebuilt from the replay buffer against the managers' histories,
  * the analytic reward and termination of every transition against the reward / termination managers.

Example:
    python scripts/supermimic/check_parity.py --motion_lib /dev/shm/sm_lib/train_lafan --num_envs 256 --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="G1-BFM-Transformer-Tracking-SuperMimic")
parser.add_argument("--motion_lib", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--action_std", type=float, default=0.3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import scaletrack.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from my_rsl_rl.storage import ReplayBuffer
from scaletrack.tasks.tracking.supermimic.rotations import quat_angle_sq

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.commands.motion.motion_lib_dir = args.motion_lib
cfg.observations.policy.enable_corruption = False
cfg.observations.policy_task.enable_corruption = False
cfg.events.push_robot = None
cfg.action_noise_std = 0.0
cfg.actions.joint_pos.noise_std = 0.0
env = gym.make(args.task, cfg=cfg).unwrapped
obs, _ = env.reset()
task = env.make_imagination_task()
N = env.num_envs
all_envs = torch.arange(N, device=env.device)
ctx = task.context(all_envs)
robot = task.robot
cmd = task.command
A = env.action_manager.total_action_dim
J = len(robot.joint_names)
buf = ReplayBuffer(N, 400, task.state_dim, A, ["frame", "mode", "roff"], env.device)
hoi = hasattr(task, "obj_ref")
if hoi:
    from scaletrack.tasks.tracking.supermimic.object_heads import object_reward_terms

    O = task.obj_slice

worst: dict[str, float] = {}


def record(name: str, value: torch.Tensor | float):
    value = float(value)
    worst[name] = max(worst.get(name, 0.0), value)


state, info = task.read_state(), task.step_info()
ep_start = torch.ones(N, dtype=torch.bool, device=env.device)
prev_action = torch.zeros(N, A, device=env.device)
n_term = 0
n_trunc = 0
for step in range(args.steps):
    # ---- observations of the current step
    bodies = task.bodies(state, ctx)
    data = robot.data
    ids = cmd.body_indexes
    record("fk_pos_m", (bodies["pos"] - (data.body_pos_w[:, ids] - env.scene.env_origins[:, None])).norm(dim=-1).max())
    record("fk_rot_rad", quat_angle_sq(bodies["quat"], data.body_quat_w[:, ids]).sqrt().max())
    prop = task.prop_obs(state, ctx, noise=False)
    for tag, sel in (("", ~ep_start), ("_after_reset", ep_start)):
        if sel.any():
            record(f"fk_ang_vel{tag}", (bodies["ang_vel"] - data.body_ang_vel_w[:, ids])[sel].norm(dim=-1).max())
            record(f"fk_com_lin_vel{tag}", (bodies["com_lin_vel"] - data.body_lin_vel_w[:, ids])[sel].norm(dim=-1).max())
            err = (prop - obs["policy"][:, -1])[sel].abs().amax(0)
            for name, sl in (("gravity", slice(0, 3)), ("ang_vel", slice(3, 6)), ("joint_pos", slice(6, 6 + J)), ("joint_vel", slice(6 + J, 6 + 2 * J))):
                record(f"obs_prop_{name}{tag}", err[sl].max())
    task_obs = task.policy_task_obs(bodies, info["frame"], info["roff"], noise=False, s=state)
    if hoi:
        task_obs, obj_obs, *bps = task_obs
        if task.use_contact_labels:
            contact = bps.pop()
            record("obs_contact_ref", (contact - obs["contact_ref"]).abs().max())
            record("contact_ref_valid_envs", obs["contact_ref"][..., -1].sum())
            record("contact_ref_nonzero_labels", (obs["contact_ref"][..., :-1] != 0).sum())
        if bps:
            bps_err = (bps[0] - obs["hand_bps"]).abs()
            record("obs_hand_bps", bps_err.max())
            # Nearest surface points are discontinuous: a tie broken differently moves one vector by a sample spacing.
            record("obs_hand_bps_mean", bps_err.mean())
            record("obs_hand_bps_frac_over_1mm", (bps_err > 1e-3).float().mean())
            record("hand_bps_nonzero_envs", (obs["hand_bps"][..., :-1].abs().sum(-1) > 0).sum())
        record("obs_object_token", (obj_obs - obs["object_actor"]).abs().max())
        valid = state[:, task.spec.valid_index] > 0.5
        record("object_valid_mismatch", (valid != cmd.object_valid).sum())
        if ep_start.any():
            # After a reset the object of an object motion sits at its reference.
            sel = ep_start & valid
            if sel.any():
                # The reset frame: one before the scored frame, except after the explicit env.reset().
                ref_now = task.obj_ref[info["frame"] - int(step > 0)][sel]
                record("object_reset_pos_m", (state[sel][:, O][:, :3] - ref_now[:, :3]).norm(dim=-1).max())
    record("obs_task", (task_obs - obs["policy_task"]).abs().max())
    mode, mapping = task.mode_inputs(info["mode"])
    record("obs_mode", (mode - obs["mode"]).abs().max())
    record("obs_mode_mapping", (mapping - obs["mode_mapping"]).abs().max())

    # ---- histories rebuilt from the replay buffer (needs the current row in the buffer)
    actions = args.action_std * torch.randn(N, A, device=env.device)
    buf.add(state, actions, actions, info, ep_start)
    # the newest row of every env is the current step
    buf._start_candidates = all_envs * buf.T + (buf.ptr - 1) % buf.T
    sb = buf.sample_starts(N, 3)
    sel = sb.env_ids
    ctx_s = task.context(sel)
    hist_prop = torch.stack([task.prop_obs(sb.hist_states[i], ctx_s, noise=False) for i in range(3)], 1)
    record("hist_prop", (hist_prop - obs["policy"][sel]).abs().max())
    record("hist_action", (sb.hist_actions.transpose(0, 1) - obs["action"][sel]).abs().max())
    for k in ("frame", "mode", "roff"):
        record(f"hist_info_{k}", (sb.info[k] != info[k][sel]).sum())

    # ---- transition
    obs, rew, terminated, truncated, _ = env.step(actions)
    reset = terminated | truncated
    s_next = task.read_state()
    reset_ids = reset.nonzero().squeeze(-1)
    s_next[reset_ids] = task.terminal_state(reset_ids)
    b_next = task.bodies(s_next, ctx)
    ref = task.reference(info["frame"])
    hard = task.heads.termination(b_next, ref)
    if hoi:
        obj_err = (s_next[:, O][:, :3] - task.obj_ref[info["frame"]][:, :3]).norm(dim=-1)
        hard |= (s_next[:, task.spec.valid_index] > 0.5) & (obj_err > task.obj_cfg.term_threshold)
        hard |= env.termination_manager.get_term("physical_invalid")
    record("termination_mismatch", (hard != terminated).float().sum())
    imagined_trunc = task.transition(s_next, b_next, actions, prev_action, info["frame"], s_prev=state)["truncated"]
    sim_trunc = env.termination_manager.get_term("motion_time_out")
    record("truncation_mismatch", (imagined_trunc != sim_trunc).float().sum())
    n_trunc += int(sim_trunc.sum())
    n_term += int(terminated.sum())
    terms = task.heads.reward_terms(s_next, b_next, actions, prev_action, ref, terminated.float())
    if hoi:
        f = info["frame"]
        obj_terms = object_reward_terms(
            task.obj_cfg, state[:, O], s_next[:, O], task.obj_ref[task.previous_frame(f)], task.obj_ref[f],
            b_next["pos"], task.lib.body_pos_w[f], s_next[:, task.spec.valid_index],
        )  # fmt: skip
        terms["object_motion"] = task.obj_cfg.w_motion * obj_terms["object_motion"] * env.step_dt
        terms["object_relation"] = task.obj_cfg.w_relation * obj_terms["object_relation"] * env.step_dt
        if task.use_contact_reward:
            terms["object_contact"] = task.contact_reward(s_next, b_next, f)
            # Imagined batches are replay samples, not environments: the term must not depend on the env order.
            perm = torch.randperm(s_next.shape[0], device=s_next.device)
            b_perm = {k: v[perm] for k, v in b_next.items() if torch.is_tensor(v)}
            record("contact_reward_permutation", (task.contact_reward(s_next[perm], b_perm, f[perm]) - terms["object_contact"][perm]).abs().max())
            half = s_next.shape[0] // 2
            b_half = {k: v[:half] for k, v in b_next.items() if torch.is_tensor(v)}
            record("contact_reward_subbatch", (task.contact_reward(s_next[:half], b_half, f[:half]) - terms["object_contact"][:half]).abs().max())
            record("contact_reward_labelled_envs", (task.lib.contact_labels_at(f, task.contact_label_cols) != 0).any(-1).sum())
            record("contact_reward_mean", terms["object_contact"].mean() / env.step_dt)
        n_moving = int((obj_terms["object_motion"] > 0).sum())
        record("object_reward_active_envs", n_moving)
    reward = sum(terms.values())
    record("reward_abs", (reward - rew).abs().max())
    record("reward_rel", ((reward - rew).abs() / rew.abs().clamp_min(1e-3)).max())
    for name, term in terms.items():
        idx = env.reward_manager.active_terms.index(name)
        record(f"reward/{name}", (term - env.reward_manager._step_reward[:, idx] * env.step_dt).abs().max())
    buf.add_terminal(reset_ids, task.terminal_state(reset_ids))

    prev_action = torch.where(reset[:, None], torch.zeros_like(actions), actions)
    state, info, ep_start = task.read_state(), task.step_info(), reset

print(f"[parity] {args.steps} steps x {N} envs, {n_term} terminations, {n_trunc} motion-end truncations")
for k, v in worst.items():
    print(f"[parity] max {k:<32s} {v:.3e}")
os._exit(0)

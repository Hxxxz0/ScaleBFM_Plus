"""Imagination interface of the BFM tracking task.

The training algorithm (my_rsl_rl.algorithms.supermimic) is task agnostic: it stores simulator transitions, fits
the world model and back-propagates imagined returns. Everything it needs to know about the tracking MDP goes
through :class:`TrackingImaginationTask`, built from the live environment:

* the state layout and world-model parameterisation (``spec``),
* per-environment simulation context xi,
* the bookkeeping of a step (tracked frame, sparse-tracking mode, random future offset),
* analytic policy observations, rewards, termination probabilities and critic features.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .context import XI_DIM, normalize_xi, read_sim_context
from .heads import HeadsCfg, RefFrames, TrackingHeads
from .kinematics import HumanoidKinematics
from .state import STATE_DIM, StateSpec, StateSpecCfg, read_state

ASSET_XML = Path(__file__).resolve().parents[3] / "assets" / "robots" / "g1_29dof" / "g1_29dof.xml"


class TrackingImaginationTask:
    state_dim = STATE_DIM
    xi_dim = XI_DIM
    reward_term_names = (
        "motion_body_height", "motion_body_pos", "motion_body_rot", "motion_body_lin_vel", "motion_body_ang_vel",
        "action_rate_l2", "joint_limit", "survival",
    )  # fmt: skip

    def __init__(self, env, heads_cfg: HeadsCfg | None = None, spec_cfg: StateSpecCfg | None = None, xml_path: str | None = None):
        self.env = env
        self.device = env.device
        self.robot = env.scene["robot"]
        self.command = env.command_manager.get_term("motion")
        self.action_term = env.action_manager.get_term("joint_pos")
        self.lib = self.command.train_lib
        cfg = heads_cfg or HeadsCfg()
        cfg.step_dt = env.step_dt
        self.spec = StateSpec(spec_cfg or StateSpecCfg(dt=env.step_dt), self.device)

        data = self.robot.data
        kin = HumanoidKinematics(str(xml_path or ASSET_XML), self.robot.joint_names, self.command.cfg.body_names, self.device)
        q0 = getattr(data, "default_joint_pos_nominal", data.default_joint_pos[0]).clone()
        self.heads = TrackingHeads(cfg, kin, q0, data.soft_joint_pos_limits[0].clone(), self.command._mode_table.clone())
        self.policy_offsets = torch.tensor(cfg.policy_future_idx, device=self.device)
        self.critic_offsets = torch.tensor(cfg.critic_future_idx, device=self.device)

        # Fixed per-environment quantities (all randomisation happens at start-up).
        self.xi = read_sim_context(self.robot, self.device)
        self.xi_norm = normalize_xi(self.xi)
        self.joint_offset = data.default_joint_pos - q0
        self.com_pos_b = self.robot.root_physx_view.get_coms().to(self.device)[:, self.command.body_indexes, :3].clone()
        self.num_modes = self.command._mode_table.shape[0]

    # ------------------------------------------------------------------ simulator side

    def read_state(self) -> torch.Tensor:
        return read_state(self.robot, self.env.scene.env_origins)

    def terminal_state(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.env.terminal_state[env_ids]

    def step_info(self) -> dict[str, torch.Tensor]:
        """Integer bookkeeping of the current step of every env (stored with each replay row)."""
        return {
            "frame": self.command.global_frame.clone(),
            "mode": self.command.mode_ids.clone(),
            "roff": self.command._rand_timestep[:, 0].clone(),
        }

    def executed_actions(self) -> torch.Tensor:
        return self.action_term.executed_raw

    # ------------------------------------------------------------------ imagination side

    def context(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "xi_norm": self.xi_norm[env_ids],
            "joint_offset": self.joint_offset[env_ids],
            "com_pos_b": self.com_pos_b[env_ids],
        }

    def reference(self, frames: torch.Tensor, with_vel: bool = True) -> RefFrames:
        lib = self.lib
        return RefFrames(
            lib.body_pos_w[frames],
            lib.body_quat_w[frames],
            lib.body_lin_vel_w[frames] if with_vel else None,
            lib.body_ang_vel_w[frames] if with_vel else None,
        )

    def bodies(self, s: torch.Tensor, ctx: dict[str, torch.Tensor], with_vel: bool = True) -> dict[str, torch.Tensor]:
        return self.heads.bodies(s, ctx["com_pos_b"] if with_vel else None, with_vel)

    def prop_obs(self, s: torch.Tensor, ctx: dict[str, torch.Tensor], noise: bool) -> torch.Tensor:
        obs = self.heads.prop_obs(s, ctx["joint_offset"])
        return self.heads.add_prop_noise(obs) if noise else obs

    def policy_task_obs(self, bodies: dict, frame: torch.Tensor, roff: torch.Tensor, noise: bool) -> torch.Tensor:
        offsets = torch.where(self.policy_offsets < 0, roff[:, None], self.policy_offsets[None, :])
        ref = self.reference(self.lib.future(frame, offsets), with_vel=False)
        obs = self.heads.task_obs(bodies, ref, offsets)
        return self.heads.add_task_noise(obs) if noise else obs

    def mode_inputs(self, mode_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.heads.mode(mode_ids), self.heads.mode_mapping(mode_ids)

    def transition(self, s_next, bodies_next, action, prev_action, frame) -> dict[str, torch.Tensor]:
        """Reward, soft termination and truncation of the transition into s_next tracked against ``frame``."""
        ref = self.reference(frame)
        done = self.heads.soft_termination(bodies_next, ref)
        terms = self.heads.reward_terms(s_next, bodies_next, action, prev_action, ref, done)
        reward = sum(terms.values())
        # motion_time_out: the simulator truncates the transition scored against the last frame of the motion.
        truncated = frame >= self.lib.frame_last[frame]
        return {"reward": reward, "done": done, "truncated": truncated, "terms": terms}

    def next_frame(self, frame: torch.Tensor) -> torch.Tensor:
        return torch.minimum(frame + 1, self.lib.frame_last[frame])

    def critic_features(self, s, bodies, frame, mode_ids, ctx) -> torch.Tensor:
        ref = self.reference(self.lib.future(frame, self.critic_offsets.expand(frame.shape[0], -1)))
        task = self.heads.critic_task_obs(bodies, ref, self.critic_offsets.expand(frame.shape[0], -1))
        return torch.cat([self.spec.features(s), ctx["xi_norm"], self.heads.mode(mode_ids), task.flatten(-2)], dim=-1)

    @property
    def critic_dim(self) -> int:
        return self.spec.in_scale.shape[0] + XI_DIM + self.heads.num_bodies + len(self.critic_offsets) * (
            self.heads.num_bodies * (3 + 3 + 6 + 6 + 3 + 3) + 1
        )

    @property
    def max_step_reward(self) -> float:
        return self.heads.cfg.max_step_reward

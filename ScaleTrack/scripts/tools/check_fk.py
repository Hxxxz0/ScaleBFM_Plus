"""Compare the analytic FK (supermimic.kinematics) of a robot MJCF with the simulator on random robot states.

    /isaac-sim/python.sh scripts/tools/check_fk.py --robot g1_inspire --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", default="g1_inspire")
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import importlib

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from scaletrack.assets import ASSET_DIR
from scaletrack.tasks.tracking.supermimic.kinematics import HumanoidKinematics
from scaletrack.tasks.tracking.supermimic.rotations import quat_angle_sq, quat_apply, quat_apply_inverse

robot_cfg = getattr(importlib.import_module(f"scaletrack.robots.{args.robot}"), f"{args.robot.upper()}_CFG")


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device, gravity=(0.0, 0.0, 0.0)))
scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=3.0))
sim.reset()
robot = scene["robot"]
n, dev = args.num_envs, robot.device
bodies = robot.body_names
kin = HumanoidKinematics(f"{ASSET_DIR}/robots/{args.robot}/{args.robot}.xml", robot.joint_names, bodies, dev)
com_b = robot.root_physx_view.get_coms().to(dev)[:, :, :3]
lo, hi = robot.data.joint_pos_limits[..., 0], robot.data.joint_pos_limits[..., 1]
worst = {}
for trial in range(5):
    q = lo + (hi - lo) * torch.rand_like(lo)
    qd = torch.randn_like(q)
    root = torch.zeros(n, 13, device=dev)
    root[:, :3] = scene.env_origins + torch.tensor([0.0, 0.0, 1.5], device=dev)
    root[:, 3:7] = torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1)
    root[:, 7:] = torch.randn(n, 6, device=dev)
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(q, qd)
    sim.forward()
    scene.update(dt=0.0)
    d = robot.data
    # root_link_lin_vel_w is stale right after a state write: derive the link velocity from the CoM velocity.
    w_w = d.root_com_vel_w[:, 3:]
    v_link = d.root_com_vel_w[:, :3] - torch.cross(w_w, quat_apply(d.root_link_quat_w, d.body_com_pos_b[:, 0]), dim=-1)
    w_b = quat_apply_inverse(d.root_link_quat_w, w_w)
    out = kin.forward(d.root_link_pos_w, d.root_link_quat_w, d.joint_pos, v_link, w_b, d.joint_vel, com_b)
    errs = {
        "pos": (out["pos"] - d.body_link_pos_w).norm(dim=-1).max(),
        "rot": quat_angle_sq(out["quat"], d.body_link_quat_w).sqrt().max(),
        "ang_vel": (out["ang_vel"] - d.body_ang_vel_w).norm(dim=-1).max(),
        "com_lin_vel": (out["com_lin_vel"] - d.body_com_lin_vel_w).norm(dim=-1).max(),
    }
    for k, v in errs.items():
        worst[k] = max(worst.get(k, 0.0), float(v))
print("[fk] " + " ".join(f"{k}={v:.2e}" for k, v in worst.items()), flush=True)
import os

os._exit(0)

"""Spawn a robot config once and dump its simulator joint / body order, CoM offsets, masses and limits to JSON.

    /isaac-sim/python.sh scripts/tools/inspect_robot.py --robot g1_inspire --output <asset dir>/sim_meta.json --headless
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", default="g1_inspire")
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import importlib

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

module = importlib.import_module(f"scaletrack.robots.{args.robot}")
robot_cfg = getattr(module, f"{args.robot.upper()}_CFG")


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
sim.reset()
robot = scene["robot"]
view = robot.root_physx_view
meta = dict(
    joint_names=robot.joint_names,
    body_names=robot.body_names,
    com_pos_b=view.get_coms()[0, :, :3].tolist(),
    masses=view.get_masses()[0].tolist(),
    joint_limits=robot.data.joint_pos_limits[0].tolist(),
    default_joint_pos=robot.data.default_joint_pos[0].tolist(),
)
with open(args.output, "w") as f:
    json.dump(meta, f, indent=1)
print(f"[inspect] joints={len(meta['joint_names'])} bodies={len(meta['body_names'])} mass={sum(meta['masses']):.2f}", flush=True)
app.close()

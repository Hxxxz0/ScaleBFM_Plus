"""Convert the G1 + Inspire hand URDF (xiaohu-art/Weave) into the USD used by scaletrack.robots.g1_inspire.

Fixed links (hand bases, finger tips, sensors) are merged into their parents; the 12 mimic finger joints become
ordinary joints whose targets are set by the mimic action (supermimic.actions.NoisyMimicJointPositionAction).

    /isaac-sim/python.sh scripts/tools/convert_g1_inspire.py --urdf <weave>/assets/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf --headless
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", required=True)
parser.add_argument(
    "--output_dir",
    default=str(Path(__file__).resolve().parents[2] / "source/scaletrack/scaletrack/assets/robots/g1_inspire"),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

cfg = UrdfConverterCfg(
    asset_path=args.urdf,
    usd_dir=args.output_dir,
    usd_file_name="g1_inspire.usd",
    fix_base=False,
    merge_fixed_joints=True,
    convert_mimic_joints_to_normal_joints=True,
    replace_cylinders_with_capsules=True,
    force_usd_conversion=True,
    make_instanceable=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
)
converter = UrdfConverter(cfg)
print(f"[convert] {converter.usd_path}", flush=True)
app.close()

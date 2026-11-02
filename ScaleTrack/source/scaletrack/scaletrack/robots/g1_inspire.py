"""Unitree G1 (29 body joints) with Inspire dexterous hands (12 joints per hand).

The body reuses the G1 29-DoF actuators, default pose and action scales unchanged. Each hand has 6 driven joints
(thumb yaw / pitch, four finger proximals) and 6 mimic joints whose position targets follow their driver
(``G1_INSPIRE_MIMIC``), as in the URDF ``<mimic>`` tags; the asset (scripts/tools/convert_g1_inspire.py) turns the
mimic joints into ordinary joints. Hand gains follow xiaohu-art/Weave (robots/g1_inspire.py).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from scaletrack.assets import ASSET_DIR

from .g1_29dof import G1_29DOF_ACTION_SCALE, G1_29DOF_CYLINDER_CFG

SIDES = ("L", "R")
FINGERS = ("index", "middle", "ring", "pinky")
HAND_DRIVER_JOINTS = [
    f"{s}_{j}_joint" for s in SIDES for j in ("thumb_proximal_yaw", "thumb_proximal_pitch", *[f"{f}_proximal" for f in FINGERS])
]
# mimic joint -> (driver joint, multiplier); the offsets are 0.
G1_INSPIRE_MIMIC = {}
for s in SIDES:
    G1_INSPIRE_MIMIC[f"{s}_thumb_intermediate_joint"] = (f"{s}_thumb_proximal_pitch_joint", 1.6)
    G1_INSPIRE_MIMIC[f"{s}_thumb_distal_joint"] = (f"{s}_thumb_proximal_pitch_joint", 2.4)
    for f in FINGERS:
        G1_INSPIRE_MIMIC[f"{s}_{f}_intermediate_joint"] = (f"{s}_{f}_proximal_joint", 1.0)
# Finger links tracked like body parts (the tips are merged into them).
FINGERTIP_BODIES = {s: [f"{s}_thumb_distal", *[f"{s}_{f}_intermediate" for f in FINGERS]] for s in SIDES}
# Link carrying the merged hand base: frame of the hand-object BPS (with HAND_BASE_OFFSET, the URDF base joint).
HAND_LINKS = {"L": "left_wrist_yaw_link", "R": "right_wrist_yaw_link"}
HAND_BASE_OFFSET = (0.0415, 0.0, 0.0)

G1_INSPIRE_CFG = G1_29DOF_CYLINDER_CFG.replace(
    spawn=G1_29DOF_CYLINDER_CFG.spawn.replace(
        usd_path=f"{ASSET_DIR}/robots/g1_inspire/g1_inspire.usd",
        # Finger links touch each other in grasps; Weave trains these motions without self collisions.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    actuators={
        **G1_29DOF_CYLINDER_CFG.actuators,
        "hands": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_.*_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim={
                "[LR]_thumb_proximal_pitch_joint": 1.7,
                "[LR]_thumb_(proximal_yaw|intermediate|distal)_joint": 2.9,
                "[LR]_(index|middle|ring|pinky)_.*_joint": 4.3,
            },
            stiffness=5.0,
            damping=0.5,
            armature=0.01,
        ),
    },
)

G1_INSPIRE_ACTION_SCALE = {**G1_29DOF_ACTION_SCALE, **{j: 1.0 for j in HAND_DRIVER_JOINTS}}

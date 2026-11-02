"""HOI tracking environment: the SuperMimic tracking task plus one dynamic rigid object per environment.

The objects (asset, mass properties, friction, collision hull, surface samples, shape descriptor) are described by the
packed training library: ``objects.json`` (several objects, scripts/pretrain/data_process/build_hdmi_objects.py) or
``object_meta.json`` (one object, pack_hoi_library.py). Every environment holds one object for its lifetime (a
standard IsaacLab RigidObject; several assets are spawned with MultiAssetSpawnerCfg and the environment's object is
read back from the stage). Per-object quantities are kept as (num_objects, ...) tables and gathered per environment
(``object_ids``). Observations, rewards and terminations follow supertracker's hoi_tracking_v1 profile; the
G1 + Inspire variant tracks the fingertips like body parts and observes the hand-object relation (hand BPS).
"""

from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils.prims import clone
from isaaclab.utils import configclass
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdPhysics

from . import hoi_mdp
from .env import SuperMimicTrackingEnv
from .env_cfg import G1BFMTrackingSuperMimicEnvCfg
from .hoi_command import HOIMotionCommandCfg
from .object_heads import hand_bps_basis
from .state import read_object_state, read_state

OBJECT_META = "object_meta.json"
OBJECTS = "objects.json"
OBJECT_ID_ATTR = "scaletrack:object_id"


@clone
def spawn_dynamic_object(prim_path, cfg: HOIObjectSpawnerCfg, translation=None, orientation=None, **kwargs):
    """Spawn a collision-only USD asset as a dynamic rigid body with the measured mass properties and material."""
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(cfg.mass_kg)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*cfg.com_local_m))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*cfg.diagonal_inertia_kg_m2))
    q = cfg.principal_axes_wxyz
    mass.CreatePrincipalAxesAttr(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    body.CreateDisableGravityAttr(False)
    body.CreateMaxDepenetrationVelocityAttr(1.0)
    body.CreateSolverPositionIterationCountAttr(8)
    body.CreateSolverVelocityIterationCountAttr(4)
    prim.CreateAttribute(OBJECT_ID_ATTR, Sdf.ValueTypeNames.Int).Set(int(cfg.object_id))
    # Material without combine-mode overrides, as authored by supertracker's spawn_hoi_scenes.
    material_path = f"/World/Materials/hoi_object_{cfg.object_id}"
    if not prim.GetStage().GetPrimAtPath(material_path).IsValid():
        sim_utils.spawn_rigid_body_material(
            material_path,
            sim_utils.RigidBodyMaterialCfg(
                static_friction=cfg.static_friction, dynamic_friction=cfg.dynamic_friction, restitution=cfg.restitution
            ),
        )
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.CollisionAPI):
            sim_utils.bind_physics_material(str(child.GetPath()), material_path)
    return prim


@configclass
class HOIObjectSpawnerCfg(sim_utils.UsdFileCfg):
    func = spawn_dynamic_object
    object_id: int = 0
    mass_kg: float = 1.0
    com_local_m: tuple = (0.0, 0.0, 0.0)
    diagonal_inertia_kg_m2: tuple = (0.01, 0.01, 0.01)
    principal_axes_wxyz: tuple = (1.0, 0.0, 0.0, 0.0)
    static_friction: float = 0.9
    dynamic_friction: float = 0.9
    restitution: float = 0.0


def load_objects(lib_dir: str) -> list[dict]:
    """Objects of a packed library, in the order of its ``object_names`` (object id = list index)."""
    lib = Path(lib_dir)
    if (lib / OBJECTS).exists():
        table = json.loads((lib / OBJECTS).read_text())["objects"]
        names = json.loads((lib / "index.json").read_text()).get("object_names", list(table))
        return [dict(table[n], name=n) for n in names]
    if (lib / OBJECT_META).exists():
        meta = json.loads((lib / OBJECT_META).read_text())
        return [dict(meta, name="object", surface_points=meta["hull_points"], shape_bps=[])]
    raise FileNotFoundError(f"{lib} has neither {OBJECTS} nor {OBJECT_META}: pack it with pack_hoi_parts.py / pack_hoi_library.py")


def spawner_cfg(obj: dict, object_id: int) -> HOIObjectSpawnerCfg:
    d = obj["dynamics"]
    return HOIObjectSpawnerCfg(
        usd_path=obj["usd_path"],
        object_id=object_id,
        mass_kg=d["mass_kg"],
        com_local_m=tuple(d["com_local_m"]),
        diagonal_inertia_kg_m2=tuple(d["diagonal_inertia_kg_m2"]),
        principal_axes_wxyz=tuple(d["principal_axes_wxyz"]),
        static_friction=d["static_friction"],
        dynamic_friction=d["dynamic_friction"],
        restitution=d["restitution"],
    )


def _pad(rows: list, device) -> torch.Tensor:
    """(K, max_len, 3) with every list padded by repeating its first point (min / nearest-point safe)."""
    n = max(len(r) for r in rows)
    return torch.tensor([list(r) + [r[0]] * (n - len(r)) for r in rows], dtype=torch.float32, device=device)


def object_descriptor(obj: dict) -> list[float]:
    """Static per-object context of the world model and critic (roughly unit scale)."""
    d = obj["dynamics"]
    return [d["mass_kg"] / 5.0, *[c / 0.2 for c in d["com_local_m"]], *[10.0 * i for i in d["diagonal_inertia_kg_m2"]],
            *obj["dimensions_m"], *obj.get("shape_bps", [])]  # fmt: skip


class HOITrackingEnv(SuperMimicTrackingEnv):
    """SuperMimicTrackingEnv whose state, terminal state and reset context include the environment's object."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self.objects = load_objects(cfg.commands.motion.motion_lib_dir)
        allocation = list(cfg.object_allocation) if cfg.object_allocation else list(range(len(self.objects)))
        cfgs = [spawner_cfg(self.objects[k], k) for k in allocation]
        if len(cfgs) == 1:
            cfg.scene.object.spawn = cfgs[0]
        else:
            cfg.scene.object.spawn = sim_utils.MultiAssetSpawnerCfg(assets_cfg=cfgs, random_choice=False)
            cfg.scene.replicate_physics = False
        self.object_prev = None
        self.object_ids = None
        super().__init__(cfg, render_mode, **kwargs)
        self.object_prev = self.object_state()

    def load_managers(self):
        # Object tables and the environment -> object map are needed by the command and observation managers.
        dev = self.device
        objs = self.objects
        stage = self.scene.stage
        ids = []
        for i in range(self.num_envs):
            attr = stage.GetPrimAtPath(f"{self.scene.env_prim_paths[i]}/Object").GetAttribute(OBJECT_ID_ATTR)
            ids.append(int(attr.Get()) if attr and attr.IsValid() and attr.Get() is not None else 0)
        self.object_ids = torch.tensor(ids, dtype=torch.long, device=dev)
        self.object_dims_table = torch.tensor([o["dimensions_m"] for o in objs], dtype=torch.float32, device=dev)
        self.object_hull_table = _pad([o["hull_points"] for o in objs], dev)
        self.object_surface_table = _pad([o["surface_points"] for o in objs], dev)
        self.object_desc_table = torch.tensor([object_descriptor(o) for o in objs], dtype=torch.float32, device=dev)
        self.object_dims = self.object_dims_table[self.object_ids]
        self.object_hull = self.object_hull_table[self.object_ids]
        self.object_surface = self.object_surface_table[self.object_ids]
        # Dense collision-surface samples of the contact reward (add_contact_points.py sidecar), if packed.
        contact_points = Path(self.cfg.commands.motion.motion_lib_dir) / "object_contact_points.npy"
        self.object_contact_table = self.object_contact_points = None
        if contact_points.exists():
            self.object_contact_table = torch.from_numpy(np.load(contact_points)).to(dev)
            self.object_contact_points = self.object_contact_table[self.object_ids]
        robot = self.scene["robot"]
        hand = self.cfg.hand_bps_links
        self.hand_link_ids = [robot.body_names.index(n) for n in hand] if hand else []
        self.hand_bps_basis = hand_bps_basis(self.cfg.hand_bps_center).to(dev)
        super().load_managers()

    def make_imagination_task(self, **kwargs):
        from .hoi_task import HOIImaginationTask

        return HOIImaginationTask(self, **kwargs)

    def object_state(self, env_ids=slice(None)) -> torch.Tensor:
        """(n, 13) env-local object state (origin velocity)."""
        command = self.command_manager.get_term("motion")
        return read_object_state(self.scene["object"], self.scene.env_origins, command.object_valid, env_ids)[:, :13]

    def step(self, action):
        # Object pose before the step: the object reward pays its increment over the step.
        self.object_prev = self.object_state()
        return super().step(action)

    def _read_terminal_state(self, env_ids):
        command = self.command_manager.get_term("motion")
        robot = read_state(self.scene["robot"], self.scene.env_origins, env_ids)
        obj = read_object_state(
            self.scene["object"], self.scene.env_origins, command.object_valid, env_ids, self.object_ids
        )
        return torch.cat([robot, obj], dim=-1)

    def _failed(self, env_ids):
        return super()._failed(env_ids) | self.termination_manager.get_term("object_pos")[env_ids]


@configclass
class ObjectActorCfg(ObsGroup):
    track = ObsTerm(func=hoi_mdp.object_track, params={"command_name": "motion", "future_idx": [0, 1, 2, 3, 4, -1]})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class HandBPSCfg(ObsGroup):
    bps = ObsTerm(func=hoi_mdp.hand_bps_obs, params={"command_name": "motion"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class G1BFMTrackingSuperMimicHOIEnvCfg(G1BFMTrackingSuperMimicEnvCfg):
    # Object id of every spawned asset in order (environment i holds entry i % len); empty = each object once.
    object_allocation: list = []
    # Hand links of the hand-object BPS (empty: no BPS) and the basis centre in their frame.
    hand_bps_links: list = []
    hand_bps_center: tuple = (0.12, 0.0, 0.0)
    # Robot MJCF of the analytic FK (empty: the G1 29-DoF default of supermimic.task).
    robot_xml: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=HOIObjectSpawnerCfg(usd_path=""),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 20.0, 0.3)),
        )
        motion = self.commands.motion
        self.commands.motion = HOIMotionCommandCfg(
            **{k: getattr(motion, k) for k in motion.__dataclass_fields__ if k != "class_type"}
        )
        self.observations.object_actor = ObjectActorCfg()
        self.rewards.object_motion = RewTerm(func=hoi_mdp.object_motion, weight=2.0, params={"command_name": "motion"})
        self.rewards.object_relation = RewTerm(func=hoi_mdp.object_relation, weight=0.5, params={"command_name": "motion"})
        self.terminations.object_pos = DoneTerm(
            func=hoi_mdp.object_pos_failure, params={"command_name": "motion", "threshold": 0.5, "disable_flag": False}
        )
        self.terminations.physical_invalid = DoneTerm(func=hoi_mdp.physical_invalid, params={"command_name": "motion"})
        self.sim.physx.gpu_collision_stack_size = 2**28


@configclass
class ObjectCriticCfg(ObjectActorCfg):
    track = ObsTerm(
        func=hoi_mdp.object_track, params={"command_name": "motion", "future_idx": [0, 1, 2, 4, 8, 16, 32]}
    )


@configclass
class G1BFMTrackingPPOBenchmarkHOIEnvCfg(G1BFMTrackingSuperMimicHOIEnvCfg):
    """Same HOI environment for PPO: no external action noise, the official privileged critic plus object tokens."""

    action_noise_std: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        from scaletrack.tasks.tracking.config.g1_29dof.flat_env_cfg import G1BFMTrackingEnvCfg

        original = G1BFMTrackingEnvCfg()
        self.observations.critic = original.observations.critic
        self.observations.critic_task = original.observations.critic_task
        self.observations.object_critic = ObjectCriticCfg()


# ---------------------------------------------------------------------- G1 + Inspire hands, all HDMI objects


@configclass
class G1InspireSuperMimicHOIEnvCfg(G1BFMTrackingSuperMimicHOIEnvCfg):
    """G1 with Inspire hands: fingertips tracked as body parts, mimic finger joints, hand-object BPS."""

    hand_bps_links: list = ["left_wrist_yaw_link", "right_wrist_yaw_link"]

    def __post_init__(self):
        super().__post_init__()
        from scaletrack.assets import ASSET_DIR
        from scaletrack.robots.g1_inspire import (
            FINGERTIP_BODIES,
            G1_INSPIRE_ACTION_SCALE,
            G1_INSPIRE_CFG,
            G1_INSPIRE_MIMIC,
        )

        from .actions import NoisyMimicJointPositionActionCfg

        self.robot_xml = f"{ASSET_DIR}/robots/g1_inspire/g1_inspire.xml"
        self.scene.robot = G1_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # The pelvis contact sensor (no reward or termination reads it) matches the G1 29-DoF USD layout only.
        self.scene.contact_forces = None
        joint_pos = self.actions.joint_pos
        self.actions.joint_pos = NoisyMimicJointPositionActionCfg(
            asset_name=joint_pos.asset_name,
            joint_names=["^(?!.*(intermediate|distal)).*$"],
            use_default_offset=True,
            scale=G1_INSPIRE_ACTION_SCALE,
            noise_std=self.action_noise_std,
            mimic=G1_INSPIRE_MIMIC,
        )
        fingers = {s: FINGERTIP_BODIES[s] for s in ("L", "R")}
        motion = self.commands.motion
        motion.body_names = list(motion.body_names) + fingers["L"] + fingers["R"]
        # A mode tracking a wrist also tracks that hand's fingertips.
        for name, links in list(motion.mode_candidates.items()):
            extra = []
            if "left_wrist_yaw_link" in links:
                extra += fingers["L"]
            if "right_wrist_yaw_link" in links:
                extra += fingers["R"]
            motion.mode_candidates[name] = list(links) + extra
        self.observations.hand_bps = HandBPSCfg()
        # The fingertips barely move the mean body errors: track the finger joints directly (open hands where the
        # reference has none). The imagination task reads this term (weight, std, joints) from here.
        from isaaclab.managers import SceneEntityCfg

        from scaletrack.tasks.tracking.mdp.rewards import joint_pos_exp

        self.rewards.motion_finger_joint = RewTerm(
            func=joint_pos_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.5, "asset_cfg": SceneEntityCfg("robot", joint_names=["[LR]_.*_joint"])},
        )
        self.commands.motion.hoi_mass = 0.6


@configclass
class ContactLabelCfg(ObsGroup):
    labels = ObsTerm(func=hoi_mdp.contact_label_obs, params={"command_name": "motion"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class G1InspireSuperMimicHOIContactEnvCfg(G1InspireSuperMimicHOIEnvCfg):
    """G1 + Inspire HOI with the reference contact labels of the next frames as actor and critic input (HDMI's
    reference_contact_labels; the library needs the add_contact_labels.py sidecar)."""

    contact_label_steps: list = [0, 5, 10, 15, 20]

    def __post_init__(self):
        super().__post_init__()
        self.observations.contact_ref = ContactLabelCfg()


@configclass
class G1InspireSuperMimicHOIContactRewardEnvCfg(G1InspireSuperMimicHOIContactEnvCfg):
    """Contact labels as input plus HDMI's contact matching reward (distance proxy, object_heads.contact_match) on the
    fingertip links, with more Weave data: each Weave object held by twice as many environments and hoi_mass 0.7
    (about 55 % Weave, 15 % OMOMO, 30 % flat-ground starts). The library needs the add_contact_points.py sidecar."""

    # Tracked fingertip links (HDMI's contact bodies minus the untracked thumb intermediate links).
    contact_links: list = [
        "L_thumb_distal", "L_index_intermediate", "L_middle_intermediate", "L_ring_intermediate", "L_pinky_intermediate",
        "R_thumb_distal", "R_index_intermediate", "R_middle_intermediate", "R_ring_intermediate", "R_pinky_intermediate",
    ]  # fmt: skip

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_contact = RewTerm(func=hoi_mdp.object_contact, weight=2.0, params={"command_name": "motion"})
        # Object ids 0-4 OMOMO, 5-13 Weave (hdmi_inspire_v1 object_names order).
        self.object_allocation = list(range(5)) + 2 * list(range(5, 14))
        self.commands.motion.hoi_mass = 0.7


@configclass
class G1InspireSuperMimicHOIContactRetryEnvCfg(G1InspireSuperMimicHOIContactRewardEnvCfg):
    """Contact labels and contact reward plus supertracker's failure-window mining and adaptive data shares:
    failures (object motions included) restart 80 % of the time 15 +- 8 frames before the failure frame, the object
    also fails when turned more than 1 rad from its reference (a toppled pole), each object's share of object starts
    follows its difficulty (hoi_mass 0.7 scaled, clipped to [0.5, 0.95]), and the pole and chair objects are held by
    three times as many environments as an OMOMO object."""

    def __post_init__(self):
        super().__post_init__()
        from .reset_sampling import FailureResetCfg

        motion = self.commands.motion
        motion.failure_reset = FailureResetCfg(
            enabled=True, local_probability=0.8, lookback_frames=15, jitter_frames=8, max_consecutive_local_resets=5
        )
        motion.object_retries = True
        motion.hoi_mass_range = (0.5, 0.95)
        self.terminations.object_pos.params["rot_threshold"] = 1.0
        # Object ids 0-4 OMOMO, 5-13 Weave; clothesstand 5, floorlamp 6, tripod 11, whitechair 12, woodchair 13.
        hard = [5, 6, 11, 12, 13]
        self.object_allocation = list(range(5)) + 2 * [7, 8, 9, 10] + 3 * hard


@configclass
class G1InspirePPOHOIContactRetryEnvCfg(G1InspireSuperMimicHOIContactRetryEnvCfg):
    """The v5 environment (G1InspireSuperMimicHOIContactRetryEnvCfg) for PPO: no external action noise, the official
    privileged critic plus object tokens (as G1BFMTrackingPPOBenchmarkHOIEnvCfg), and failure retries active from
    the start (there is no world-model warm-up)."""

    action_noise_std: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        from scaletrack.tasks.tracking.config.g1_29dof.flat_env_cfg import G1BFMTrackingEnvCfg

        original = G1BFMTrackingEnvCfg()
        self.observations.critic = original.observations.critic
        self.observations.critic_task = original.observations.critic_task
        self.observations.object_critic = ObjectCriticCfg()
        self.commands.motion.failure_reset.activate_after_world_model_warmup = False

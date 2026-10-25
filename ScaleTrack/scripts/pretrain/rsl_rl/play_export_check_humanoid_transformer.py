"""
Tips:
1. Turn off default_joint_pos randomization
2. Modify future idx in observation terms

If you wanna test the control mode selection:
3. Determine a specific control mode in env
4. Turn inference in get_actor_obs to False in act_inference
5. Recover the annotation in the loop
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--export_format",
    choices=["tensorrt", "onnx"],
    default="tensorrt",
    help="TensorRT module for this machine, or a portable ONNX model (checked with onnxruntime).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import copy
import time
import math
import gymnasium as gym
import os
import re
import json
import pathlib
import torch
import numpy as np
import torch.nn as nn
import xml.etree.ElementTree as ET

from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import scaletrack.tasks  # noqa: F401

@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)

@torch.jit.script
def quat_mul_inverse_left(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)    
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w1_inv = w1
    x1_inv = -x1
    y1_inv = -y1
    z1_inv = -z1
    ww = (z1_inv + x1_inv) * (x2 + y2)
    yy = (w1_inv - y1_inv) * (w2 + z2)
    zz = (w1_inv + y1_inv) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1_inv - x1_inv) * (x2 - y2))
    w = qq - ww + (z1_inv - y1_inv) * (y2 - z2)
    x = qq - xx + (x1_inv + w1_inv) * (x2 + w2)
    y = qq - yy + (w1_inv - x1_inv) * (y2 + z2)
    z = qq - zz + (z1_inv + y1_inv) * (w2 - x2)
    
    return torch.stack([w, x, y, z], dim=-1).view(shape)

@torch.jit.script
def quat_mul_inverse_right(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w2_inv = w2
    x2_inv = -x2
    y2_inv = -y2
    z2_inv = -z2
    ww = (z1 + x1) * (x2_inv + y2_inv)
    yy = (w1 - y1) * (w2_inv + z2_inv)
    zz = (w1 + y1) * (w2_inv - z2_inv)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2_inv - y2_inv))
    w = qq - ww + (z1 - y1) * (y2_inv - z2_inv)
    x = qq - xx + (x1 + w1) * (x2_inv + w2_inv)
    y = qq - yy + (w1 - x1) * (y2_inv + z2_inv)
    z = qq - zz + (z1 + y1) * (w2_inv - x2_inv)
    
    return torch.stack([w, x, y, z], dim=-1).view(shape)

class HumanoidTransformerPolicyWrapperWithMode(nn.Module):

    def __init__(
        self,
        policy,
        mode_mappings: torch.Tensor,
        mode_vectors: torch.Tensor,
        default_dof_pos: torch.Tensor,
        action_scale: torch.Tensor,
        context_len: int,
        future_len: int, 
        local_translation: torch.Tensor,
        local_rotation: torch.Tensor,
        parent_indices: torch.Tensor,
        joint_axis: torch.Tensor,
        selected_link_indices: torch.Tensor,
        lab_to_xml_joint_indices: torch.Tensor,
    ) -> None:
        super().__init__()

        # Modules and Components
        self.task_embedder = policy.actor_task_embedder
        self.prop_embedder = policy.actor.prop_projection
        self.action_embedder = policy.actor.action_projection
        self.transformer_blocks = policy.actor.transformer_blocks
        self.final_norm = policy.actor.final_norm
        self.projection_head = policy.actor.projection_head
        self.register_buffer("empty_embedding", policy.actor.empty_embedding)
        attn_mask = torch.zeros(2*context_len, 2*context_len, dtype=torch.bool, device=mode_vectors.device)
        row_idx = torch.arange(2*context_len - 1, device=mode_vectors.device)
        col_idx = torch.full((2*context_len - 1,), 2*context_len - 1, device=mode_vectors.device)
        attn_mask[row_idx, col_idx] = True
        self.register_buffer("self_attn_mask", attn_mask)

        # Observation and Action
        self.register_buffer("mode_mappings", mode_mappings)
        self.register_buffer("mode_vectors", mode_vectors)
        self.register_buffer("gravity_vec", torch.tensor([[0,0,-1]], dtype=torch.float, device=mode_vectors.device).unsqueeze(1).expand(-1, context_len, -1))
        self.register_buffer("default_dof_pos", default_dof_pos)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("selected_link_indices", selected_link_indices)
        tan_vec = torch.zeros(1, future_len, selected_link_indices.shape[-1], 3, dtype=torch.float,device=mode_vectors.device)
        norm_vec = torch.zeros(1, future_len, selected_link_indices.shape[-1], 3, dtype=torch.float,device=mode_vectors.device)
        tan_vec[..., 0] = 1
        norm_vec[..., -1] = 1
        self.register_buffer("tan_vec", tan_vec)
        self.register_buffer("norm_vec", norm_vec)

        # FK Modules
        self.register_buffer("local_translation", local_translation) # (num_joints, 3)
        self.register_buffer("local_rotation", local_rotation) # (num_joints, 4)
        self.register_buffer("parent_indices", parent_indices) # (num_joints + 1)
        self.register_buffer("joint_axis", joint_axis) # (num_joints, 3)
        self.register_buffer("lab_to_xml_joint_indices", lab_to_xml_joint_indices)

    def forward(
        self,
        root_quat_buffer: torch.Tensor, # wxyz
        base_ang_vel_buffer: torch.Tensor,
        dof_pos_buffer: torch.Tensor, # in Isaaclab order
        dof_vel_buffer: torch.Tensor, # in Isaaclab order
        last_action_buffer: torch.Tensor,
        target_body_pos_future_to_robot_base: torch.Tensor, # (bs, num_future, num_link, 3)
        target_body_rot_future_to_robot_base: torch.Tensor,
        mode_index: torch.Tensor,
        time_offsets: torch.Tensor,
    ) -> torch.Tensor:
        
        # Build Context
        projected_gravity_buffer = quat_apply_inverse(root_quat_buffer, self.gravity_vec) # (bs, num_context, 3)
        dof_pos_rel_buffer = dof_pos_buffer - self.default_dof_pos
        prop_obs = torch.cat([
            projected_gravity_buffer,
            base_ang_vel_buffer,
            dof_pos_rel_buffer,
            dof_vel_buffer * 0.05
        ], dim=-1)
        prop_token = self.prop_embedder(prop_obs)
        action_token = self.action_embedder(last_action_buffer)

        x = torch.empty(prop_token.shape[0], 2*prop_token.shape[1], prop_token.shape[2], dtype=prop_token.dtype, device=prop_token.device)
        x[:, ::2] = prop_token
        x[:, 1:-1:2] = action_token[:, 1:]
        x[:, 2*prop_token.shape[1]-1] = self.empty_embedding

        # Extract current state from buffer
        dof_pos = dof_pos_buffer[:, -1]

        # FK Module
        half_angles = dof_pos[:, self.lab_to_xml_joint_indices].unsqueeze(-1) / 2
        sin_half = torch.sin(half_angles)
        cos_half = torch.cos(half_angles)
        joint_rot = torch.cat([cos_half, self.joint_axis.unsqueeze(0) * sin_half], dim=-1)
        
        body_pos = torch.zeros(1, len(self.parent_indices), 3, dtype=torch.float, device=joint_rot.device)
        body_quat = torch.zeros(1, len(self.parent_indices), 4, dtype=torch.float, device=joint_rot.device)
        body_quat[..., 0] = 1
        
        for j in range(1, len(self.parent_indices)):
            j_rot = joint_rot[:, j-1]
            local_trans = self.local_translation[j-1:j]
            local_rot = self.local_rotation[j-1:j]
            parent_idx = self.parent_indices[j:j+1]
            
            parent_pos = body_pos[:, parent_idx].squeeze(1) # (1, 1, 3)
            parent_rot = body_quat[:, parent_idx].squeeze(1) # (1, 1, 4)
    
            world_trans = quat_apply(parent_rot, local_trans)
            curr_pos = parent_pos + world_trans
            curr_rot = quat_mul(local_rot, j_rot)
            curr_rot = quat_mul(parent_rot, curr_rot)
            
            body_pos[:, j] = curr_pos
            body_quat[:, j] = curr_rot

        body_pos_to_robot_base = body_pos[:, self.selected_link_indices]
        body_quat_to_robot_base = body_quat[:, self.selected_link_indices]

        # Build Task Observation
        target_body_pos_future_rel_to_robot_base = target_body_pos_future_to_robot_base - body_pos_to_robot_base[:, None, :, :]
        
        target_body_rot_future_to_robot_base_tan_norm = torch.cat([
            quat_apply(target_body_rot_future_to_robot_base, self.tan_vec),
            quat_apply(target_body_rot_future_to_robot_base, self.norm_vec),
        ], dim=-1)
        target_body_rot_future_rel_to_robot_base = quat_mul_inverse_right(
            target_body_rot_future_to_robot_base,
            body_quat_to_robot_base[:, None].expand(-1, target_body_rot_future_to_robot_base.shape[1], -1, -1)
        )
        target_body_rot_future_rel_to_robot_base_tan_norm = torch.cat([
            quat_apply(target_body_rot_future_rel_to_robot_base, self.tan_vec),
            quat_apply(target_body_rot_future_rel_to_robot_base, self.norm_vec),
        ], dim=-1)
        
        task_obs = torch.cat([
            target_body_pos_future_to_robot_base.flatten(2,3),
            target_body_pos_future_rel_to_robot_base.flatten(2,3),
            target_body_rot_future_to_robot_base_tan_norm.flatten(2,3),
            target_body_rot_future_rel_to_robot_base_tan_norm.flatten(2,3),
            time_offsets
        ], dim=-1) # (bs, nf, ndim)

        # Apply control mode
        mapping = self.mode_mappings[mode_index]
        task_obs_masked = task_obs * mapping.unsqueeze(1)
        mode_vec = self.mode_vectors[mode_index]
        task_input = torch.cat([task_obs_masked, mode_vec.unsqueeze(1).expand(-1, task_obs_masked.shape[1], -1)], dim=-1)
        task_tokens = self.task_embedder(task_input)

        # Forward
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, task_tokens, self_attn_mask=self.self_attn_mask)
        x = self.final_norm(x)

        action = self.projection_head(x[:, -1, :])
        
        # Return Direct PD target and action for buffer
        return action * self.action_scale + self.default_dof_pos, action


ONNX_INPUT_NAMES = [
    "root_quat_buffer",
    "base_ang_vel_buffer",
    "dof_pos_buffer",
    "dof_vel_buffer",
    "last_action_buffer",
    "target_body_pos_future",
    "target_body_rot_future",
    "mode_index",
    "time_offsets",
]


class OnnxPolicy:
    """Run the exported ONNX model with onnxruntime behind the same call signature as the TensorRT module."""

    def __init__(self, path: str):
        import onnxruntime

        self.session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]

    def __call__(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        feeds = {name: x.detach().cpu().numpy() for name, x in zip(ONNX_INPUT_NAMES, inputs) if name in self.input_names}
        return [torch.from_numpy(y).to(inputs[0].device) for y in self.session.run(None, feeds)]


def build_mode_mappings(
    mode_table: torch.Tensor,
    feature_dims_per_link,
    with_time: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build (num_modes, task_obs_dim) mapping from mode table and feature dims.

    Replicates the logic of mdp.mode_mapping for each row of mode_table, so the
    exported ONNX can apply the correct mask per mode index without env.
    task_obs_dim = sum(feature_dims_per_link) + (1 if with_time else 0).
    """
    device = device or mode_table.device
    mode_table = mode_table.to(device)
    assert mode_table.dim() == 2, "mode_table must be (num_modes, num_links)"
    num_modes, num_links = mode_table.shape[0], mode_table.shape[1]
    mappings = []
    for dim in feature_dims_per_link:
        # (num_modes, num_links, dim) * (num_modes, num_links, 1) -> view (num_modes, -1)
        part = (
            torch.ones(num_modes, mode_table.shape[1], dim, device=device, dtype=torch.float32)
            * mode_table.unsqueeze(-1)
        ).view(num_modes, -1)
        mappings.append(part)
    if with_time:
        mappings.append(torch.ones(num_modes, 1, device=device, dtype=torch.float32))
    return torch.cat(mappings, dim=-1)

def parse_xml(xml_file, device = "cpu"):
    body_names = []
    parent_indices = []
    local_translation = []
    local_rotation = []
    joint_axis = []
    joint_names = []
   
    tree = ET.parse(xml_file)
    xml_doc_root = tree.getroot()
    xml_world_body = xml_doc_root.find("worldbody")
    assert xml_world_body is not None, "worldbody not found"
    
    xml_body_root = xml_world_body.find("body")
    assert xml_body_root is not None, "body not found"
    
    compiler_data = xml_doc_root.find("compiler")
    rot_unit = compiler_data.attrib.get("angle", "degree")
    assert rot_unit in ["degree", "radian"], f"Invalid rotation unit: {rot_unit}"
    
    def _add_xml_body(xml_node, parent_index, body_index):
        body_name = xml_node.attrib.get("name")
        pos_data = xml_node.attrib.get("pos", "0 0 0")
        pos = np.fromstring(pos_data, dtype=float, sep=" ")
        
        rot_data = xml_node.attrib.get("quat", "1 0 0 0")
        rot = np.fromstring(rot_data, dtype=float, sep=" ")
        
        if body_index == 0:
            pass
        else:
            curr_joints = xml_node.findall("joint")
            num_joints = len(curr_joints)
            assert num_joints == 1
            _axis = np.fromstring(curr_joints[0].attrib.get("axis"), dtype=float, sep=" ")
            axis = torch.from_numpy(_axis)
            local_rotation.append(rot)
            local_translation.append(pos)
            joint_axis.append(axis)
            joint_names.append(curr_joints[0].attrib.get("name"))
        
        body_names.append(body_name)
        parent_indices.append(parent_index)
        
        curr_index = body_index
        body_index += 1
        for child in xml_node.findall("body"):
            body_index = _add_xml_body(child, curr_index, body_index)
            
        return body_index
    
    _add_xml_body(xml_body_root, -1, 0)
    
    parent_indices = torch.tensor(parent_indices, dtype=torch.long, device=device)
    local_translation = torch.tensor(np.array(local_translation), dtype=torch.float, device=device)
    local_rotation = torch.tensor(np.array(local_rotation), dtype=torch.float, device=device)
    joint_axis = torch.stack(joint_axis, dim=0).float().to(device) # weishuai: The original variable is float64

    return body_names, joint_names, parent_indices, joint_axis, local_translation, local_rotation

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    future_idx = [0, 1, 2, 3, 4, 5]
    for term_name in (
        "target_body_pos",
        "target_body_pos_rel",
        "target_body_rot",
        "target_body_rot_rel",
        "timestamp",
    ):
        getattr(env_cfg.observations.policy_task, term_name).params["future_idx"] = future_idx.copy()
    mode_mapping_params = env_cfg.observations.mode_mapping.mode_mapping.params
    mode_feature_dims = list(mode_mapping_params["feature_dims_per_link"])
    mode_mapping_with_time = mode_mapping_params.get("with_time", False)
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "add_joint_default_pos"):
        env_cfg.events.add_joint_default_pos = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
    env_cfg.commands.motion.motion_file = args_cli.motion_file
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)
    checkpoint_stem = pathlib.Path(resume_path).stem
    mode_table_path = os.path.join(log_dir, "mode_table.pt")
    model_suffix = "_tensorrt.pt" if args_cli.export_format == "tensorrt" else ".onnx"
    model_path = os.path.join(log_dir, f"{checkpoint_stem}{model_suffix}")
    metadata_path = os.path.join(log_dir, f"{checkpoint_stem}_{args_cli.export_format}_metadata.json")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    
    ppo_runner._set_env_is_evaluating()

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.reset()

    # prepare wrapper initialization
    mode_vector = env.unwrapped.command_manager.get_term('motion')._mode_table # weishuai: Also we use this as device anchor
    torch.save(mode_vector, mode_table_path)
    mode_mappings = build_mode_mappings(mode_vector, mode_feature_dims, with_time=mode_mapping_with_time)
    default_dof_pos =  env.unwrapped.command_manager.get_term('motion').robot.data.default_joint_pos.clone()
    action_scale = env.unwrapped.action_manager.get_term('joint_pos')._scale
    context_len = obs["policy"].shape[-2]
    future_len = len(future_idx)
    time_offsets = torch.as_tensor(future_idx, dtype=torch.long, device=mode_vector.device)[None, :, None]

    xml_path = env.unwrapped.scene.cfg.robot.spawn.usd_path.replace('.usda','.xml')
    body_names, joint_names, parent_indices, joint_axis, local_translation, local_rotation = parse_xml(xml_path, mode_vector.device)
    local_rotation = local_rotation / local_rotation.norm(dim=-1, keepdim=True)

    selected_body_name = env.unwrapped.command_manager.get_term('motion').cfg.body_names
    selected_body_indices = torch.tensor([body_names.index(name) for name in selected_body_name], dtype=torch.long, device=mode_vector.device)
    
    robot_joint_names = env.unwrapped.command_manager.get_term('motion').robot.joint_names
    lab_to_xml_joint_idx = [robot_joint_names.index(name) for name in joint_names]
    lab_to_xml_joint_idx = torch.tensor(lab_to_xml_joint_idx, dtype=torch.long, device=mode_vector.device)

    with torch.inference_mode():
        # Initialize wrapper
        wrapper = HumanoidTransformerPolicyWrapperWithMode(
            ppo_runner.alg.policy,
            mode_mappings,
            mode_vector,
            default_dof_pos,
            action_scale,
            context_len,
            future_len,
            local_translation,
            local_rotation,
            parent_indices,
            joint_axis,
            selected_body_indices,
            lab_to_xml_joint_idx
        )
        wrapper.eval().cuda()

        # Prepare fake input
        root_pos = env.unwrapped.command_manager.get_term('motion').robot_anchor_pos_w
        root_quat = env.unwrapped.command_manager.get_term('motion').robot_anchor_quat_w
        base_ang_vel = env.unwrapped.command_manager.get_term('motion').robot.data.root_ang_vel_b
        dof_pos = env.unwrapped.command_manager.get_term('motion').robot.data.joint_pos
        dof_vel = env.unwrapped.command_manager.get_term('motion').robot.data.joint_vel
        
        body_pos_w_future = env.unwrapped.command_manager.get_term('motion').body_pos_w_future_manual(future_idx)
        body_quat_w_future = env.unwrapped.command_manager.get_term('motion').body_quat_w_future_manual(future_idx)
        target_body_pos_future_to_robot_base = quat_apply_inverse(root_quat, body_pos_w_future - root_pos[:, None, None, :])
        root_quat_expand = root_quat[:, None, None, :].expand(-1, body_quat_w_future.shape[1], body_quat_w_future.shape[2], -1)
        target_body_rot_future_to_robot_base = quat_mul_inverse_left(root_quat_expand, body_quat_w_future)
        
        trt_inputs = [
            root_quat.unsqueeze(1).expand(-1, obs["policy"].shape[-2], -1),
            base_ang_vel.unsqueeze(1).expand(-1, obs["policy"].shape[-2], -1),
            dof_pos.unsqueeze(1).expand(-1, obs["policy"].shape[-2], -1),
            dof_vel.unsqueeze(1).expand(-1, obs["policy"].shape[-2], -1),
            torch.zeros(1, obs["policy"].shape[-2], obs["action"].shape[-1], dtype=torch.float, device=mode_vector.device),
            target_body_pos_future_to_robot_base,
            target_body_rot_future_to_robot_base,
            torch.tensor([0], dtype=torch.long, device=mode_vector.device),
            time_offsets
        ]

        if args_cli.export_format == "tensorrt":
            import torch_tensorrt

            trt_module = torch_tensorrt.compile(
                wrapper,
                ir="dynamo",
                inputs=trt_inputs,
            )
            torch_tensorrt.save(trt_module, model_path, output_format="torchscript", inputs=trt_inputs)
        else:
            # Traced on a CPU copy: ONNX constant folding cannot mix devices, and the policy stays on the GPU.
            torch.onnx.export(
                copy.deepcopy(wrapper).cpu(),
                tuple(x.cpu() for x in trt_inputs),
                model_path,
                input_names=ONNX_INPUT_NAMES,
                output_names=["joint_pos_target", "action"],
                opset_version=17,
                dynamo=False,
            )
        
        stiffness_dict = {}
        damping_dict = {}
        torque_limit_dict = {}
        for a in env.unwrapped.scene.cfg.robot.actuators.values():
            s = a.stiffness
            d = a.damping
            t = a.effort_limit_sim
            names = a.joint_names_expr
            if not isinstance(s, dict):
                s = {n: s for n in names}
            if not isinstance(d, dict):
                d = {n: d for n in names}
            if not isinstance(t, dict):
                t = {n: t for n in names}
            stiffness_dict.update(s)
            damping_dict.update(d)
            torque_limit_dict.update(t)

        stiffness_list = []
        damping_list = []
        torque_limit_list = []
        for joint in env.unwrapped.command_manager.get_term('motion').robot.joint_names:
            for pattern, val in stiffness_dict.items():
                if re.match(pattern, joint):
                    stiffness_list.append(val)
            for pattern, val in damping_dict.items():
                if re.match(pattern, joint):
                    damping_list.append(val)
            for pattern, val in torque_limit_dict.items():
                if re.match(pattern, joint):
                    torque_limit_list.append(val)

        actor = ppo_runner.alg.policy.actor
        task_embedder = ppo_runner.alg.policy.actor_task_embedder
        task_projection_layers = [
            module for module in task_embedder.task_projection.modules() if isinstance(module, nn.Linear)
        ]
        if not task_projection_layers:
            raise ValueError("Actor task embedder must contain at least one linear projection layer.")

        policy_architecture = {
            "prop_obs_dim": actor.prop_projection.in_features,
            "task_obs_dim": task_projection_layers[0].in_features,
            "action_dim": actor.action_projection.in_features,
            "output_dim": actor.projection_head.out_features,
            "embedding_dim": actor.embed_dim,
            "num_heads": actor.transformer_blocks[0].self_attention.num_heads,
            "ff_dim": actor.transformer_blocks[0].feed_forward.w.out_features,
            "num_layers": len(actor.transformer_blocks),
            "reduced_task_dim": task_embedder.W.shape[-1] if hasattr(task_embedder, "W") else None,
            "task_embedder_hidden_dims": [layer.out_features for layer in task_projection_layers[:-1]],
        }
                
        metadata = {
            "schema_version": 1,
            "policy_architecture": policy_architecture,
            "selected_body_names": selected_body_name,
            "body_names": env.unwrapped.command_manager.get_term('motion').robot.body_names,
            "joint_names": env.unwrapped.command_manager.get_term('motion').robot.joint_names,
            "action_names": env.unwrapped.action_manager.get_term('joint_pos')._joint_names,
            "history_buffer_size": context_len,
            "future_idx": future_idx,
            "mode_feature_dims": mode_feature_dims,
            "mode_mapping_with_time": mode_mapping_with_time,
            "default_dof_pos": default_dof_pos[0].reshape(-1).detach().cpu().tolist(),
            "action_scale": (action_scale[0] if action_scale.dim() > 1 else action_scale)
            .reshape(-1)
            .detach()
            .cpu()
            .tolist(),
            "stiffness": stiffness_list,
            "damping": damping_list,
            "torque_limit": torque_limit_list
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    print(f"[INFO]: Saved mode table to: {mode_table_path}")
    print(f"[INFO]: Saved {args_cli.export_format} model to: {model_path}")
    print(f"[INFO]: Saved {args_cli.export_format} metadata to: {metadata_path}")

    if args_cli.export_format == "tensorrt":
        model = torch.jit.load(model_path).to(mode_vector.device)
    else:
        model = OnnxPolicy(model_path)
    mode_index = torch.tensor([7], dtype=torch.long, device=mode_vector.device)

    root_pos = env.unwrapped.command_manager.get_term('motion').robot_anchor_pos_w
    root_quat = env.unwrapped.command_manager.get_term('motion').robot_anchor_quat_w
    base_ang_vel = env.unwrapped.command_manager.get_term('motion').robot.data.root_ang_vel_b
    dof_pos = env.unwrapped.command_manager.get_term('motion').robot.data.joint_pos
    dof_vel = env.unwrapped.command_manager.get_term('motion').robot.data.joint_vel
    
    body_pos_w_future = env.unwrapped.command_manager.get_term('motion').body_pos_w_future_manual(future_idx)
    body_quat_w_future = env.unwrapped.command_manager.get_term('motion').body_quat_w_future_manual(future_idx)

    actions = torch.zeros(1, obs["action"].shape[-1], dtype=torch.float, device=mode_vector.device)

    # The buffer is initialized with first frame value
    root_quat_buffer = root_quat.unsqueeze(1).expand(-1, obs["policy"].shape[-2], -1)
    base_ang_vel_buffer = base_ang_vel.unsqueeze(1).expand(-1, obs["policy"].shape[-2], -1)
    dof_pos_buffer = dof_pos.unsqueeze(1).expand(-1, obs["policy"].shape[-2], -1)
    dof_vel_buffer = dof_vel.unsqueeze(1).expand(-1, obs["policy"].shape[-2], -1)
    actions_buffer = actions.unsqueeze(1).expand(-1, obs["policy"].shape[-2], -1)

    timestep = 0
    while simulation_app.is_running():

        with torch.inference_mode():
            root_pos = env.unwrapped.command_manager.get_term('motion').robot_anchor_pos_w
            root_quat = env.unwrapped.command_manager.get_term('motion').robot_anchor_quat_w
            base_ang_vel = env.unwrapped.command_manager.get_term('motion').robot.data.root_ang_vel_b
            dof_pos = env.unwrapped.command_manager.get_term('motion').robot.data.joint_pos
            dof_vel = env.unwrapped.command_manager.get_term('motion').robot.data.joint_vel
            
            body_pos_w_future = env.unwrapped.command_manager.get_term('motion').body_pos_w_future_manual(future_idx)
            body_quat_w_future = env.unwrapped.command_manager.get_term('motion').body_quat_w_future_manual(future_idx)

            root_quat_buffer = torch.roll(root_quat_buffer, -1, 1)
            base_ang_vel_buffer = torch.roll(base_ang_vel_buffer, -1, 1)
            dof_pos_buffer = torch.roll(dof_pos_buffer, -1, 1)
            dof_vel_buffer = torch.roll(dof_vel_buffer, -1, 1)
            actions_buffer = torch.roll(actions_buffer, -1, 1)

            root_quat_buffer[:, -1] = root_quat
            base_ang_vel_buffer[:, -1] = base_ang_vel
            dof_pos_buffer[:, -1] = dof_pos
            dof_vel_buffer[:, -1] = dof_vel
            actions_buffer[:, -1] = actions

            projected_gravity = quat_apply_inverse(root_quat_buffer, torch.tensor([[0,0,-1]], dtype=torch.float, device=mode_vector.device).unsqueeze(1).expand(-1, root_quat_buffer.shape[1], -1))
            dof_pos_rel = dof_pos_buffer - default_dof_pos
            prop_obs = torch.cat([
                projected_gravity,
                base_ang_vel_buffer,
                dof_pos_rel,
                dof_vel_buffer * 0.05
            ], dim=-1)

            print(f"Prop Obs: {torch.allclose(prop_obs, obs['policy'], atol=1e-04)}")

            half_angles = dof_pos[..., lab_to_xml_joint_idx].unsqueeze(-1) / 2
            sin_half = torch.sin(half_angles) # (bs, num_joints, 1)
            cos_half = torch.cos(half_angles)
            joint_rot = torch.cat([
                cos_half, joint_axis.unsqueeze(0) * sin_half
            ], dim=-1)
            
            body_pos = torch.zeros(1, len(parent_indices), 3, dtype=torch.float, device=joint_rot.device)
            body_quat = torch.zeros(1, len(parent_indices), 4, dtype=torch.float, device=joint_rot.device)
            body_quat[..., 0] = 1
            
            for j in range(1, len(parent_indices)):
                j_rot = joint_rot[:, j-1]
                local_trans = local_translation[j-1:j]
                local_rot = local_rotation[j-1:j]
                parent_idx = parent_indices[j:j+1]
                
                parent_pos = body_pos[:, parent_idx].squeeze(1) # (1, 1, 3)
                parent_rot = body_quat[:, parent_idx].squeeze(1) # (1, 1, 4)
        
                world_trans = quat_apply(parent_rot, local_trans)
                curr_pos = parent_pos + world_trans
                curr_rot = quat_mul(local_rot, j_rot)
                curr_rot = quat_mul(parent_rot, curr_rot)
                
                body_pos[:, j] = curr_pos
                body_quat[:, j] = curr_rot

            body_pos_to_robot_base = body_pos[:, selected_body_indices]
            body_quat_to_robot_base = body_quat[:, selected_body_indices]
                        
            future_time_offsets = torch.as_tensor(future_idx, dtype=torch.long, device=mode_vector.device).unsqueeze(0)
            tan_vec = torch.zeros(1, future_time_offsets.shape[-1], selected_body_indices.shape[-1], 3, dtype=torch.float, device=mode_vector.device)
            norm_vec = torch.zeros(1, future_time_offsets.shape[-1], selected_body_indices.shape[-1], 3, dtype=torch.float, device=mode_vector.device)
            tan_vec[..., 0] = 1
            norm_vec[..., -1] = 1
        
            target_body_pos_future_to_robot_base = quat_apply_inverse(root_quat, body_pos_w_future - root_pos[:, None, None, :])
            target_body_pos_future_rel_to_robot_base = target_body_pos_future_to_robot_base - body_pos_to_robot_base[:, None, :, :]
            
            root_quat_expand = root_quat[:, None, None, :].expand(-1, body_quat_w_future.shape[1], body_quat_w_future.shape[2], -1)
            target_body_rot_future_to_robot_base = quat_mul_inverse_left(root_quat_expand, body_quat_w_future)
            target_body_rot_future_to_robot_base_tan_norm = torch.cat([
                quat_apply(target_body_rot_future_to_robot_base, tan_vec),
                quat_apply(target_body_rot_future_to_robot_base, norm_vec),
            ], dim=-1)
            target_body_rot_future_rel_to_robot_base = quat_mul_inverse_right(
                target_body_rot_future_to_robot_base,
                body_quat_to_robot_base[:, None].expand(-1, body_quat_w_future.shape[1], -1, -1)
            )
            target_body_rot_future_rel_to_robot_base_tan_norm = torch.cat([
                quat_apply(target_body_rot_future_rel_to_robot_base, tan_vec),
                quat_apply(target_body_rot_future_rel_to_robot_base, norm_vec),
            ], dim=-1)
            
            task_obs = torch.cat([
                target_body_pos_future_to_robot_base.flatten(2,3),
                target_body_pos_future_rel_to_robot_base.flatten(2,3),
                target_body_rot_future_to_robot_base_tan_norm.flatten(2,3),
                target_body_rot_future_rel_to_robot_base_tan_norm.flatten(2,3),
                future_time_offsets.unsqueeze(-1)
            ], dim=-1) # (bs, nf, ndim)

            print(f"Task obs: {torch.allclose(task_obs, obs['policy_task'], atol=1e-04)}")
            
            print(f"Action obs: {torch.allclose(actions_buffer, obs['action'], atol=1e-04)}")

            timer = time.time()
            _, actions = model(*[
                root_quat_buffer,
                base_ang_vel_buffer,
                dof_pos_buffer,
                dof_vel_buffer,
                actions_buffer,
                target_body_pos_future_to_robot_base,
                target_body_rot_future_to_robot_base,
                mode_index,
                future_time_offsets.unsqueeze(-1)
            ])
            span = time.time() - timer
            print(f"Inference takes: {span}s")
            print(f"Inference runs at a frequency of: {1/span}HZ")
            
            normal_actions = policy(obs)
            print(f"Output Actions: {torch.allclose(actions, normal_actions, atol=1e-02)}")

            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

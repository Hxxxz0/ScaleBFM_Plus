"""Record fixed test motions with a SuperMimic checkpoint, without constructing a trainer.

Use the checkpoint run's frozen scaletrack and my_rsl_rl sources in PYTHONPATH.
The recording uses the runner's evaluation setup and deterministic full-task inference.
Reference markers are visual only; they do not change observations or physics.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--motion_lib_dir", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--motion_ids", type=int, nargs="+", default=[233, 2116, 2550, 2524, 2500])
parser.add_argument("--max_seconds", type=float, default=20.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args, multi_gpu=False).app

import hashlib
import json
import os
import subprocess
import time
import traceback

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import isaaclab.sim as sim_utils
import scaletrack.tasks  # noqa: F401
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from my_rsl_rl.modules import ActorCriticHumanoidTransformer
from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner
from my_rsl_rl.utils import resolve_obs_groups


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    task = "G1-BFM-Transformer-Tracking-SuperMimic"
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.commands.motion.motion_lib_dir = args.motion_lib_dir
    cfg.commands.motion.test_motion_lib_dir = ""
    cfg.commands.motion.shard_across_ranks = False
    cfg.viewer.origin_type = "world"
    cfg.viewer.resolution = (args.width, args.height)
    cfg.viewer.eye = (2.8, -2.8, 1.9)
    cfg.viewer.lookat = (0.0, 0.0, 0.85)
    raw_env = gym.make(task, cfg=cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(raw_env)
    base = env.unwrapped
    command = base.command_manager.get_term("motion")

    # Reuse the exact evaluation conditions from the training runner.
    evaluator = OnPolicyRunner.__new__(OnPolicyRunner)
    evaluator.env, evaluator.command_name = env, "motion"
    evaluator._set_env_is_evaluating(test_set=False)
    base.action_manager.get_term("joint_pos").noise_std = 0.0
    train_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point").to_dict()
    obs = env.get_observations()
    for name, shape in train_cfg["supermimic"]["ppo_critic_obs_shapes"].items():
        obs[name] = torch.zeros(obs.batch_size[0], *shape, device=obs.device)
    train_cfg["obs_groups"] = resolve_obs_groups(obs, train_cfg["obs_groups"], ["critic"])
    policy_cfg = dict(train_cfg["policy"])
    policy_cfg.pop("class_name")
    policy = ActorCriticHumanoidTransformer(obs, train_cfg["obs_groups"], env.num_actions, **policy_cfg).to(args.device)
    for module in policy.modules():
        if hasattr(module, "need_weights"):
            module.need_weights = False
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()
    update = int(Path(args.checkpoint).stem.split("_")[-1])

    reference = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/record_reference",
        markers={"body": sim_utils.SphereCfg(radius=0.025, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.9, 0.85), emissive_color=(0.0, 0.15, 0.14), roughness=0.7))},
    ))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = round(1.0 / base.step_dt)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    metric_keys = ["error_body_pos_g", "error_body_pos", "error_body_rot", "error_joint_pos"]
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "update": update, "task": task, "seed": args.seed, "fps": fps,
        "motion_library": args.motion_lib_dir, "motion_index_sha256": sha256(Path(args.motion_lib_dir) / "index.json"),
        "source_scaletrack": scaletrack.__file__,
        "inference": "act_inference: complete task, all-ones mode, no action or observation noise",
        "reset_disturbance": cfg.commands.motion.enable_reset_disturbance,
        "success_criterion": "max over time of mean body global position error <= 0.5 m",
        "clips": [],
    }
    manifest_path = output_dir / "recording.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (previous["checkpoint_sha256"] == manifest["checkpoint_sha256"]
                and previous["motion_index_sha256"] == manifest["motion_index_sha256"]):
            manifest["clips"] = previous["clips"]
    for motion_id in args.motion_ids:
        name = command.motion_names[motion_id]
        print(f"[record] preparing motion={motion_id} name={name}", flush=True)
        command.motion_ids[:] = motion_id
        obs, _ = env.reset()
        policy.reset()
        command._update_metrics()
        # Avoid the end-of-motion automatic reset appearing in the video.
        num_steps = min(int(command.time_totals[motion_id]) - 2, int(args.max_seconds * fps) - 1)
        clip_path = output_dir / f"sm_{update}_{motion_id}_{name.split('/')[0]}.mp4"
        encoder = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{args.width}x{args.height}", "-r", str(fps), "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(clip_path),
        ], stdin=subprocess.PIPE)
        values = []
        camera_xy = command.robot_anchor_pos_w[0, :2].clone()
        started = time.monotonic()
        try:
            for step in range(num_steps + 1):
                # Match evaluate_test_set's no_grad context. Inference tensors cannot
                # be zeroed by the command manager when resetting the next clip.
                with torch.no_grad():
                    if step:
                        obs, _, dones, _ = env.step(policy.act_inference(obs))
                        if dones.any():
                            raise RuntimeError(f"Unexpected automatic reset in motion {motion_id} at step {step}")
                        values.append([float(command.metrics[key][0]) for key in metric_keys])
                    reference.visualize(command.body_pos_w[0])
                    center = 0.5 * (command.robot_anchor_pos_w[0, :2] + command.anchor_pos_w[0, :2])
                    camera_xy = 0.92 * camera_xy + 0.08 * center
                    x, y = camera_xy.tolist()
                    base.sim.set_camera_view((x + 2.3, y - 2.3, 1.8), (x, y, 0.85), cfg.viewer.cam_prim_path)
                    # The initial RGB annotator needs several render-only warmup frames.
                    if not step:
                        raw_env.render()
                        for _ in range(8):
                            base.sim.render()
                    frame = raw_env.render()
                if not frame.size or frame.max() == 0:
                    raise RuntimeError("Renderer returned an empty/black frame")
                canvas = Image.fromarray(frame)
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, args.width, 99), fill=(20, 24, 30))
                draw.text((16, 10), f"SuperMimic | update {update} | {name.split('/')[0]} | motion {motion_id}", font=font, fill="white")
                short_name = name.split("/")[-1]
                draw.text((16, 40), short_name[:92], font=small, fill=(210, 215, 222))
                error = float(command.metrics["error_body_pos_g"][0])
                draw.text((16, 68), f"t={step / fps:.2f}s   mean body global error={error:.3f}m   cyan dots: reference", font=small, fill=(70, 235, 220))
                encoder.stdin.write(np.asarray(canvas).tobytes())
                if step % 100 == 0:
                    print(f"[record] motion={motion_id} step={step}/{num_steps} error={error:.3f}m", flush=True)
        finally:
            encoder.stdin.close()
            returncode = encoder.wait(timeout=60)
        if returncode:
            raise RuntimeError(f"ffmpeg failed: {returncode}")
        metrics = np.asarray(values)
        clip = {
            "path": str(clip_path.resolve()), "motion_id": motion_id, "motion_name": name,
            "frames": num_steps + 1, "duration_s": (num_steps + 1) / fps,
            "full_motion_frames": int(command.time_totals[motion_id]),
            "mean_metrics": dict(zip(metric_keys, metrics.mean(0).tolist())),
            "max_metrics": dict(zip(metric_keys, metrics.max(0).tolist())),
            "segment_success": bool(metrics[:, 0].max() <= 0.5),
            "record_wall_seconds": time.monotonic() - started,
        }
        manifest["clips"] = [entry for entry in manifest["clips"] if entry["motion_id"] != motion_id]
        manifest["clips"].append(clip)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print("[record] complete " + json.dumps(clip), flush=True)
    env.close()
    print("[record] all videos saved", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        # Isaac's shutdown callback can loop indefinitely after a playback error.
        os._exit(1)
    # Encoders and the environment are closed above; skip the headless Kit shutdown loop.
    os._exit(0)

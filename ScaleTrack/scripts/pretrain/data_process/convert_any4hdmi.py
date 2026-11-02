"""Convert HDMI any4hdmi HOI datasets (qpos-only motions) into packed reference parts for a robot.

Per dataset (one object), every motion is resampled to 50 Hz (linear joints / positions, slerp quaternions), mapped by
joint name onto the robot's simulator joints (joints missing in the source, e.g. the hands of OMOMO's G1, are 0 = open
hand), differentiated (central differences) and pushed through the robot's analytic FK (supermimic.kinematics, checked
against IsaacLab by scripts/tools/check_fk.py) to get the tracked bodies' link poses, CoM linear velocities and
angular velocities, the same quantities as the IsaacLab-converted motions. The object's reference state comes from its
free joint; its supported start frames follow supertracker.hoi_reset.supported_frames (resting on the floor, slow,
motion ahead). Output, per dataset and split, is a packed part (pack_hoi_library.py --parts merges parts):

    <output>/<split>/<object>/index.json        motion names, lengths, body / joint names, object name, eval_start
    <output>/<split>/<object>/<field>.npy       FIELDS + object fields + start_ok, all motions concatenated

Splits hold out ~10% of the source motions (Weave variants _vNN stay with their source clip).

    /isaac-sim/python.sh scripts/pretrain/data_process/convert_any4hdmi.py \
        --datasets <data_root>/any4hdmi --objects <data_root>/any4hdmi/objects/objects.json \
        --output <data_root>/any4hdmi/converted_g1_inspire
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "source/scaletrack/test"))
from sm_offline import load  # noqa: E402  (pure-torch supermimic modules without IsaacLab)

kin_mod = load("kinematics")
rot = load("rotations")

ASSET = ROOT / "source/scaletrack/scaletrack/assets/robots/g1_inspire"
G1_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link", "right_knee_link",
    "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]  # fmt: skip
FINGERTIPS = [f"{s}_{b}" for s in "LR" for b in ("thumb_distal", "index_intermediate", "middle_intermediate",
                                                  "ring_intermediate", "pinky_intermediate")]  # fmt: skip
TRACKED_BODIES = G1_BODIES + FINGERTIPS
FPS = 50.0


def slerp(q0, q1, t):
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot).clip(max=1.0)
    theta = np.arccos(dot)
    s = np.sin(theta)
    small = s < 1e-6
    w0 = np.where(small, 1 - t, np.sin((1 - t) * theta) / np.where(small, 1, s))
    w1 = np.where(small, t, np.sin(t * theta) / np.where(small, 1, s))
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def resample(x, quat_slices, fps_in):
    T = x.shape[0]
    if abs(fps_in - FPS) < 1e-6:
        return x.copy()
    t_new = np.arange(0.0, (T - 1) / fps_in + 1e-9, 1.0 / FPS)
    s = t_new * fps_in
    i0 = np.clip(np.floor(s).astype(int), 0, T - 2)
    a = (s - i0)[:, None]
    out = (1 - a) * x[i0] + a * x[i0 + 1]
    for sl in quat_slices:
        out[:, sl] = slerp(x[i0, sl], x[i0 + 1, sl], a)
    return out


def ang_vel_world(q):
    """World angular velocity of a wxyz quaternion trajectory (central differences, one-sided at the ends)."""
    qt = torch.from_numpy(q).double()
    nxt = torch.cat([qt[1:], qt[-1:]])
    prv = torch.cat([qt[:1], qt[:-1]])
    span = torch.full((len(q), 1), 2.0 / FPS, dtype=torch.float64)
    span[0] = span[-1] = 1.0 / FPS
    rv = rot.rotvec_from_quat(rot.quat_mul(nxt, rot.quat_conjugate(prv)))
    return (rv / span).float().numpy()


def supported_frames(bottom, speed, angular, original_start=0, max_rest_height=0.15):
    """supertracker.hoi_reset.supported_frames with the rest height taken relative to the clip (+ the last two frames
    are never starts). The source data rest objects at inconsistent heights (Weave smallbox 1 cm into the floor,
    OMOMO objects 4-11 cm above it): a frame rests when the object is within 5 mm of its lowest height in the clip."""
    rest = bottom.min()
    moving = (speed > 0.08) | (angular > 0.15)
    future = np.cumsum(moving[::-1])[::-1] - moving
    valid = (bottom - rest <= 0.005) & (abs(rest) <= max_rest_height) & (speed < 0.15) & (angular < 0.3) & (future >= 10)
    valid[:original_start] = False
    valid[original_start] = True
    valid[-2:] = False
    return valid


class Reference:
    """Tracked-body reference of robot trajectories through the analytic FK."""

    def __init__(self, device="cpu"):
        meta = json.loads((ASSET / "sim_meta.json").read_text())
        self.joint_names = meta["joint_names"]
        com = torch.tensor(meta["com_pos_b"])[[meta["body_names"].index(b) for b in TRACKED_BODIES]]
        self.com = com.to(device)
        self.pelvis_com = torch.tensor(meta["com_pos_b"][meta["body_names"].index("pelvis")], device=device)
        self.kin = kin_mod.HumanoidKinematics(str(ASSET / "g1_inspire.xml"), self.joint_names, TRACKED_BODIES, device)
        self.device = device

    @torch.no_grad()
    def bodies(self, root_pos, root_quat, joint_pos, root_lin_vel_w, root_ang_vel_w, joint_vel, chunk=65536):
        """All inputs (F, ·) float32 numpy/torch; root_lin_vel_w is the link-origin velocity. Returns numpy arrays."""
        outs = {k: [] for k in ("pos", "quat", "com_lin_vel", "ang_vel")}
        for lo in range(0, len(root_pos), chunk):
            sl = slice(lo, lo + chunk)
            t = [torch.as_tensor(x[sl], dtype=torch.float32, device=self.device)
                 for x in (root_pos, root_quat, joint_pos, root_lin_vel_w, root_ang_vel_w, joint_vel)]  # fmt: skip
            w_b = rot.quat_apply_inverse(t[1], t[4])
            o = self.kin.forward(t[0], t[1], t[2], t[3], w_b, t[5], self.com.expand(len(t[0]), -1, -1))
            for k in outs:
                outs[k].append(o[k].cpu().numpy())
        return {k: np.concatenate(v) for k, v in outs.items()}


def split_of(name: str) -> str:
    group = re.sub(r"_v\d+$", "", name)
    return "eval" if int(hashlib.sha256(group.encode()).hexdigest(), 16) % 10 == 0 else "train"


def convert_dataset(ref: Reference, d: Path, obj_name: str, hull: np.ndarray, out: Path):
    manifest = json.loads((d / manifest_file(d)).read_text())
    names = manifest["qpos_names"]
    fps = 1.0 / manifest["timestep"]
    col = {n: i for i, n in enumerate(names)}
    root = [col[f"pelvis_{k}"] for k in ("tx", "ty", "tz", "qw", "qx", "qy", "qz")]
    obj = [col[f"object_root_{k}"] for k in ("tx", "ty", "tz", "qw", "qx", "qy", "qz")]
    joint_src = np.array([col.get(j, -1) for j in ref.joint_names])
    files = sorted((d / manifest.get("motions_subdir", "motions")).glob("*.npz"))
    parts = {"train": [], "eval": []}
    for f in files:
        q = np.load(f)["qpos"].astype(np.float64)
        q = resample(q, [slice(root[3], root[3] + 4), slice(obj[3], obj[3] + 4)], fps)
        T = len(q)
        if T < 12:
            continue
        rp, rq = q[:, root[:3]], q[:, root[3:]]
        rq /= np.linalg.norm(rq, axis=-1, keepdims=True)
        jp = np.where(joint_src >= 0, q[:, np.maximum(joint_src, 0)], 0.0)
        op, oq = q[:, obj[:3]], q[:, obj[3:]]
        oq /= np.linalg.norm(oq, axis=-1, keepdims=True)
        rv, jv, ov = (np.gradient(x, 1.0 / FPS, axis=0) for x in (rp, jp, op))
        rw, ow = ang_vel_world(rq), ang_vel_world(oq)
        b = ref.bodies(rp, rq, jp, rv, rw, jv)
        # Lowest point of the object's collision hull at every frame.
        R = rot.quat_apply(torch.from_numpy(oq).float()[:, None].expand(-1, len(hull), 4), torch.from_numpy(hull).float()[None].expand(T, -1, -1))
        bottom = R[..., 2].amin(-1).numpy() + op[:, 2]
        ok = supported_frames(bottom, np.linalg.norm(ov, axis=-1), np.linalg.norm(ow, axis=-1))
        rec = dict(
            name=f"{obj_name}/{f.stem}", length=T,
            joint_pos=jp, joint_vel=jv, body_pos_w=b["pos"], body_quat_w=b["quat"], body_lin_vel_w=b["com_lin_vel"],
            body_ang_vel_w=b["ang_vel"], object_pos_w=op, object_quat_w=oq, object_lin_vel_w=ov, object_ang_vel_w=ow,
            start_ok=ok, num_starts=int(ok.sum()), source=str(f),
        )  # fmt: skip
        parts[split_of(f.stem)].append(rec)
    for split, recs in parts.items():
        if not recs:
            continue
        o = out / split / obj_name
        o.mkdir(parents=True, exist_ok=True)
        for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
                  "object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w"):  # fmt: skip
            np.save(o / f"{k}.npy", np.concatenate([r[k] for r in recs]).astype(np.float32))
        np.save(o / "start_ok.npy", np.concatenate([r["start_ok"] for r in recs]))
        index = dict(
            motion_names=[r["name"] for r in recs], lengths=[r["length"] for r in recs], body_names=TRACKED_BODIES,
            joint_names=ref.joint_names, fps=int(FPS), object_name=obj_name, eval_start=[0] * len(recs),
            num_starts=[r["num_starts"] for r in recs], source_files=[r["source"] for r in recs],
            source_manifest=str(d / manifest_file(d)), source_fps=fps,
        )  # fmt: skip
        (o / "index.json").write_text(json.dumps(index))
        print(f"{obj_name}/{split}: {len(recs)} motions, {sum(r['length'] for r in recs)} frames, "
              f"starts median {int(np.median([r['num_starts'] for r in recs]))}", flush=True)


def manifest_file(d: Path) -> str:
    return "manifest.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", type=Path, required=True)
    ap.add_argument("--objects", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="object names to convert (default: all)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    objects = json.loads(a.objects.read_text())["objects"]
    ref = Reference(a.device)
    t0 = time.time()
    for name, o in objects.items():
        if a.only and name not in a.only:
            continue
        convert_dataset(ref, Path(o["dataset_dir"]), name, np.asarray(o["hull_points"], dtype=np.float32), a.output)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

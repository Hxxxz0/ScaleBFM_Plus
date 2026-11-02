"""Pack a multi-object HOI library for the G1 + Inspire robot from converted parts and a flat-ground base library.

Parts come from convert_any4hdmi.py (one directory per object and split, already in the robot's joint order and
tracked bodies). Base motions (a packed G1 29-DoF library, motions without an object) are carried over to the robot:
joints are reordered by name with the hand joints at 0 (open hands), the root linear velocity is re-derived by central
differences of the pelvis position, and all tracked bodies (including the fingertips) are recomputed with the robot's
FK. The first 14 bodies must agree with the base arrays (same body kinematics), which is checked.

Output (motion_lib.py format): FIELDS + OBJECT_FIELDS arrays, index.json with object_ids / object_names /
object_valid / eval_start, and objects.json (copied, the object assets of object_names).

    /isaac-sim/python.sh scripts/pretrain/data_process/pack_hoi_parts.py \
        --parts <converted>/train --objects <objects>/objects.json \
        --base_library <sm_lib>/train_lafan_holdout_v2 --base_fraction 0.2 --output_dir <sm_lib>/hdmi_inspire_v1/train
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from convert_any4hdmi import FPS, TRACKED_BODIES, Reference
from pack_motion_library import FIELDS

OBJECT_FIELDS = ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w")


def base_motions(ref: Reference, lib: Path, count: int, fraction: float, seed: int):
    index = json.loads((lib / "index.json").read_text())
    n = len(index["motion_names"])
    count = count or int(round(fraction * n))
    chosen = np.sort(np.random.default_rng(seed).choice(n, size=min(count, n), replace=False))
    starts = np.concatenate([[0], np.cumsum(index["lengths"])])
    rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in chosen])
    lengths = [index["lengths"][i] for i in chosen]
    src = {k: np.load(lib / f"{k}.npy", mmap_mode="r")[rows] for k in FIELDS}
    F = len(rows)
    jmap = np.array([index["joint_names"].index(j) if j in index["joint_names"] else -1 for j in ref.joint_names])
    jp = np.where(jmap >= 0, src["joint_pos"][:, np.maximum(jmap, 0)], 0.0).astype(np.float32)
    jv = np.where(jmap >= 0, src["joint_vel"][:, np.maximum(jmap, 0)], 0.0).astype(np.float32)
    rp, rq, rw = src["body_pos_w"][:, 0], src["body_quat_w"][:, 0], src["body_ang_vel_w"][:, 0]
    rv = np.empty_like(rp)
    s = 0
    for L in lengths:
        rv[s : s + L] = np.gradient(rp[s : s + L], 1.0 / FPS, axis=0)
        s += L
    b = ref.bodies(rp, rq, jp, rv, rw, jv)
    nb = len(index["body_names"])
    if index["body_names"] != TRACKED_BODIES[:nb]:
        raise SystemExit("base library bodies are not the first tracked bodies")
    err = np.abs(b["pos"][:, :nb] - src["body_pos_w"]).max()
    if err > 1e-3:
        raise SystemExit(f"base bodies disagree with the robot FK by {err:.2e} m")
    print(f"base: {len(chosen)}/{n} motions, {F} frames, FK body agreement {err:.1e} m", flush=True)
    ok = np.ones(F, dtype=bool)
    ends = np.cumsum(lengths) - 1
    ok[ends] = ok[ends - 1] = False
    zeros3 = np.zeros((F, 3), np.float32)
    quat = np.zeros((F, 4), np.float32)
    quat[:, 0] = 1
    arrays = dict(
        joint_pos=jp, joint_vel=jv, body_pos_w=b["pos"], body_quat_w=b["quat"], body_lin_vel_w=b["com_lin_vel"],
        body_ang_vel_w=b["ang_vel"], object_pos_w=zeros3, object_quat_w=quat, object_lin_vel_w=zeros3,
        object_ang_vel_w=zeros3, start_ok=ok,
    )  # fmt: skip
    return [index["motion_names"][i] for i in chosen], lengths, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", type=Path, required=True, help="directory of converted per-object parts (one split)")
    ap.add_argument("--objects", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--base_library", type=Path)
    ap.add_argument("--base_fraction", type=float, default=1.0)
    ap.add_argument("--base_count", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.output_dir.exists():
        raise SystemExit(f"{a.output_dir} exists")
    t0 = time.time()
    objects = json.loads(a.objects.read_text())
    object_names = list(objects["objects"])
    ref = Reference(a.device)
    names, lengths, object_ids, eval_start = [], [], [], []
    chunks = {k: [] for k in FIELDS + OBJECT_FIELDS + ("start_ok",)}
    if a.base_library is not None:
        n, L, arrays = base_motions(ref, a.base_library, a.base_count, a.base_fraction, a.seed)
        names += n
        lengths += L
        object_ids += [-1] * len(n)
        eval_start += [0] * len(n)
        for k in chunks:
            chunks[k].append(arrays[k])
    for part in sorted(p for p in a.parts.iterdir() if (p / "index.json").exists()):
        index = json.loads((part / "index.json").read_text())
        if index["joint_names"] != ref.joint_names or index["body_names"] != TRACKED_BODIES:
            raise SystemExit(f"{part}: joint / body order differs from the robot")
        oid = object_names.index(index["object_name"])
        names += index["motion_names"]
        lengths += index["lengths"]
        object_ids += [oid] * len(index["motion_names"])
        eval_start += index["eval_start"]
        for k in chunks:
            chunks[k].append(np.load(part / f"{k}.npy"))
        print(f"part {index['object_name']}: {len(index['motion_names'])} motions, {sum(index['lengths'])} frames", flush=True)
    if len(set(names)) != len(names):
        raise SystemExit("duplicate motion names")
    a.output_dir.mkdir(parents=True)
    for k, v in chunks.items():
        arr = np.concatenate(v)
        if len(arr) != sum(lengths):
            raise SystemExit(f"{k}: {len(arr)} frames, index says {sum(lengths)}")
        np.save(a.output_dir / f"{k}.npy", arr if k == "start_ok" else arr.astype(np.float32))
    index = dict(
        motion_names=names, lengths=lengths, body_names=TRACKED_BODIES, joint_names=ref.joint_names, fps=int(FPS),
        object_names=object_names, object_ids=object_ids, object_valid=[i >= 0 for i in object_ids],
        eval_start=eval_start, base_library=str(a.base_library) if a.base_library else None, parts=str(a.parts),
    )  # fmt: skip
    (a.output_dir / "index.json").write_text(json.dumps(index))
    shutil.copy2(a.objects, a.output_dir / "objects.json")
    hoi = sum(L for L, i in zip(lengths, object_ids) if i >= 0)
    print(f"packed {len(names)} motions / {sum(lengths)} frames ({hoi} with an object) into {a.output_dir} "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

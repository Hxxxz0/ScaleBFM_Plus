"""Pack a motion library for HOI tracking: flat-ground motions plus motions with one dynamic object.

The object motions come from Supertracker's ``hoi_tracking_v1`` manifests (read only): the flat-ground physical
interactions (``source_category == "robot-object"``, an Omni box carried, placed, pushed or rotated), all with the
same object asset. Optionally a random subset of an existing packed library is prepended (motions without an object).

Besides the arrays of pack_motion_library.py the library holds

    object_pos_w.npy (F, 3), object_quat_w.npy (F, 4) wxyz, object_lin_vel_w.npy (F, 3), object_ang_vel_w.npy (F, 3)
        reference object state, env-local (zeros / identity for motions without an object)
    start_ok.npy (F,) bool      frames an episode may start from: the audited supported starts of object motions
                                (object at rest on the floor, motion ahead), every frame but the last two otherwise
    index.json                  + object_valid, eval_start (audited original start, local frame) per motion
    object_meta.json            asset, mass properties, friction, dimensions and collision-hull points of the object

Run with the Isaac Sim python (pxr reads the collision mesh):

    /isaac-sim/python.sh scripts/pretrain/data_process/pack_hoi_library.py --split train \
        --base_library /path/to/sm_lib/train --base_fraction 0.2 --output_dir /path/to/hoi_lib/train
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import time
from pathlib import Path

from pack_motion_library import FIELDS, G1_TRACKED_BODIES, file_hash

OBJECT_FIELDS = ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w")
GROUP = "omni_box"


def hull_points(usd_path: Path) -> np.ndarray:
    """Vertices of the convex hull of the collision meshes (object frame)."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull

    stage = Usd.Stage.Open(str(usd_path))
    points = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for p in UsdGeom.Mesh(prim).GetPointsAttr().Get():
                points.append(list(xf.Transform(p)))
    points = np.asarray(points, dtype=np.float64)
    return points[ConvexHull(points).vertices].astype(np.float32)


def object_motions(root: Path, split: str):
    manifest = json.loads((root / "datasets/hoi_tracking_v1" / f"{split}.json").read_text())
    rows = [r for r in manifest["records"] if r.get("source_category") == "robot-object"]
    if not rows:
        raise SystemExit(f"no robot-object motions in the {split} manifest")
    return rows


def load_object_motion(root: Path, row: dict):
    with np.load(root / "datasets" / row["path"]) as z:
        if int(z["fps"]) != 50:
            raise ValueError(f"{row['id']}: motion must be 50 FPS")
        names = [str(x) for x in z["robot_body_names"]]
        idx = [names.index(b) for b in G1_TRACKED_BODIES]
        out = {k: np.ascontiguousarray(z[k][:, idx] if k.startswith("body") else z[k], dtype=np.float32) for k in FIELDS}
        for k in OBJECT_FIELDS:
            out[k] = np.ascontiguousarray(z[k][:, 0], dtype=np.float32)
        joints = [str(x) for x in z["robot_joint_names"]]
    T = out["joint_pos"].shape[0]
    if T != row["frames"]:
        raise ValueError(f"{row['id']}: {T} frames, manifest says {row['frames']}")
    ok = np.zeros(T, dtype=bool)
    starts = [f for f in row["dynamic_training_start_frames"] if f <= T - 3]
    if row["dynamic_start_frame"] > T - 3 or not starts:
        raise ValueError(f"{row['id']}: no supported start before the last two frames")
    ok[starts] = True
    ok[row["dynamic_start_frame"]] = True
    out["start_ok"] = ok
    return out, joints


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("train", "validation", "test"), required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--supertracker_root", type=Path, required=True)
    ap.add_argument("--base_library", type=Path, help="packed library whose motions (no object) are prepended")
    ap.add_argument("--base_fraction", type=float, default=1.0, help="random fraction of the base motions")
    ap.add_argument("--base_count", type=int, default=0, help="number of random base motions (overrides fraction)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--category_groups", action="store_true", help="group object motions by category (evaluation)")
    a = ap.parse_args()
    if a.output_dir.exists():
        raise SystemExit(f"{a.output_dir} exists")
    t0 = time.time()
    root = a.supertracker_root

    rows = object_motions(root, a.split)
    scenes = {r["scene_path"] for r in rows}
    parts = []
    for path in sorted(scenes):
        parts += [p for p in json.loads((root / "datasets" / path).read_text())["parts"] if p.get("physics_role") == "dynamic"]
    assets = {p["sha256"] for p in parts}
    if len(assets) != 1:
        raise SystemExit(f"expected one object asset, found {len(assets)}")
    part = parts[0]
    if any(p["dynamics"] != part["dynamics"] for p in parts):
        raise SystemExit("object dynamics differ between scenes")
    usd = root / "datasets" / part["usd_path"]
    if file_hash(usd) != part["sha256"]:
        raise SystemExit(f"{usd}: hash differs from the scene description")

    names, lengths, valid, eval_start = [], [], [], []
    chunks = {k: [] for k in FIELDS + OBJECT_FIELDS + ("start_ok",)}
    joint_names = None
    if a.base_library is not None:
        index = json.loads((a.base_library / "index.json").read_text())
        if index["body_names"] != G1_TRACKED_BODIES:
            raise SystemExit("base library bodies differ")
        joint_names = index["joint_names"]
        n = len(index["motion_names"])
        count = a.base_count or int(round(a.base_fraction * n))
        chosen = np.sort(np.random.default_rng(a.seed).choice(n, size=min(count, n), replace=False))
        starts = np.concatenate([[0], np.cumsum(index["lengths"])])
        rows_base = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in chosen])
        for k in FIELDS:
            chunks[k].append(np.load(a.base_library / f"{k}.npy", mmap_mode="r")[rows_base])
        F = len(rows_base)
        for k, shape in (("object_pos_w", 3), ("object_lin_vel_w", 3), ("object_ang_vel_w", 3)):
            chunks[k].append(np.zeros((F, shape), np.float32))
        quat = np.zeros((F, 4), np.float32)
        quat[:, 0] = 1.0
        chunks["object_quat_w"].append(quat)
        ok = np.ones(F, dtype=bool)
        for i in chosen:
            L = index["lengths"][i]
            names.append(index["motion_names"][i])
            lengths.append(L)
            valid.append(False)
            eval_start.append(0)
        ends = np.cumsum(lengths) - 1
        ok[ends] = False
        ok[ends - 1] = False
        chunks["start_ok"].append(ok)
        print(f"base: {len(chosen)}/{n} motions, {F} frames ({time.time() - t0:.0f}s)", flush=True)

    for row in rows:
        arrays, joints = load_object_motion(root, row)
        if joint_names is None:
            joint_names = joints
        elif joints != joint_names:
            raise SystemExit(f"{row['id']}: joint order differs")
        for k in chunks:
            chunks[k].append(arrays[k])
        group = f"{GROUP}_{row['category']}" if a.category_groups else GROUP
        names.append(f"{group}/{row['id']}")
        lengths.append(int(arrays["joint_pos"].shape[0]))
        valid.append(True)
        eval_start.append(int(row["dynamic_start_frame"]))
    if len(set(names)) != len(names):
        raise SystemExit("duplicate motion names")

    a.output_dir.mkdir(parents=True)
    for k, v in chunks.items():
        np.save(a.output_dir / f"{k}.npy", np.concatenate(v))
    index = dict(
        motion_names=names,
        lengths=lengths,
        body_names=G1_TRACKED_BODIES,
        joint_names=joint_names,
        fps=50,
        object_valid=valid,
        eval_start=eval_start,
        source_motion_file=str(root / "datasets/hoi_tracking_v1" / f"{a.split}.json"),
        base_library=str(a.base_library) if a.base_library else None,
    )
    (a.output_dir / "index.json").write_text(json.dumps(index))
    meta = dict(
        usd_path=str(usd),
        asset_sha256=part["sha256"],
        dimensions_m=part["dynamics"]["dimensions_m"],
        dynamics=part["dynamics"],
        hull_points=hull_points(usd).tolist(),
        scenes=sorted(scenes),
    )
    (a.output_dir / "object_meta.json").write_text(json.dumps(meta, indent=1))
    print(
        f"packed {len(names)} motions ({sum(valid)} with the object) / {sum(lengths)} frames into {a.output_dir} "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

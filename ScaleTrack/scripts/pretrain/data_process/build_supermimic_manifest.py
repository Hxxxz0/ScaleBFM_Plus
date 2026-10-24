"""Build the flat-ground train / eval motion YAML files for SuperMimic-style training.

Datasets (all 50 Hz G1 npz in IsaacLab body/joint order):
  * BONES-SEED: filtered with the SuperMimic paper rules (Appendix E): a name filter that removes clips needing
    terrain or objects, and a feasibility filter (elevated support, ground penetration, floating feet).
  * any4hdmi_g1_v1: train + validation records (test records go to the eval set), feasibility filter only.
  * LAFAN1: all 40 full clips for training; the 1000-frame test clips go to the eval set.

Every file is also validated (fps, joint/body name order, finite values, length >= 2), so the motion loader
never meets a broken file.

Example:
    python scripts/pretrain/data_process/build_supermimic_manifest.py \
        --datasets_root /path/to/datasets \
        --output_dir source/scaletrack/data/yaml_files/supermimic_flat
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

FPS = 50

# SONIC's keyword list minus pure-acrobatics terms, plus the paper's examples (same list as SuperMimic-Reproduced).
SONIC_KEYWORDS = [
    "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on", "seat", "table", "_sit_", "sit_",
    "ladder", "crutch", "_bed_", "_ride_", "scooter", "stepdown", "box_hspu", "50cm_box_", "on_box",
    "fall_from", "on_1m", "form_box", "off_1m", "230m", "jump_over_obstacle_", "lift_crate_come_up_",
    "stair", "box_jump", "box_dips", "walking_on_edge", "push_obstacle",
]
PAPER_EXTRA = ["_box_", "box_", "_stair", "jump_off_50cm", "off_50cm", "fence", "look_over"]
KEYWORDS = sorted(set(SONIC_KEYWORDS + PAPER_EXTRA))

ELEVATED_Z = 0.20
ELEVATED_S = 0.6
PENETRATE_Z_HARD = -0.12
PENETRATE_Z_SOFT = -0.08
PENETRATE_FRAC = 0.80
FLOAT_Z = 0.02
FLOAT_S = 0.5
UPRIGHT_GZ = 0.9
STAND_Z = (0.60, 0.85)
STILL_V = 0.1
STILL_QD = 0.5

# Foot collision capsules of g1_29dof.xml in the ankle_roll_link frame: (fromto start, fromto end), radius 0.01.
FOOT_CAPSULES = {
    "left_ankle_roll_link": [
        ((0.1, -0.026, -0.025), (0.05, -0.027, -0.025)),
        ((-0.044, -0.018, -0.025), (0.123, -0.018, -0.025)),
        ((-0.052, -0.01, -0.025), (0.13, -0.01, -0.025)),
        ((-0.054, 0.0, -0.025), (0.132, 0.0, -0.025)),
        ((-0.052, 0.01, -0.025), (0.13, 0.01, -0.025)),
        ((-0.044, 0.018, -0.025), (0.123, 0.018, -0.025)),
        ((0.1, 0.026, -0.025), (0.05, 0.026, -0.025)),
    ],
    "right_ankle_roll_link": [
        ((0.1, -0.026, -0.025), (0.05, -0.026, -0.025)),
        ((-0.044, -0.018, -0.025), (0.123, -0.018, -0.025)),
        ((-0.052, -0.01, -0.025), (0.13, -0.01, -0.025)),
        ((-0.054, 0.0, -0.025), (0.132, 0.0, -0.025)),
        ((-0.052, 0.01, -0.025), (0.13, 0.01, -0.025)),
        ((-0.044, 0.018, -0.025), (0.123, 0.018, -0.025)),
        ((0.1, 0.026, -0.025), (0.05, 0.026, -0.025)),
    ],
}
FOOT_RADIUS = 0.01


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return int((edges[1::2] - edges[0::2]).max())


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (..., 3) by wxyz quaternions q (..., 4)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _foot_sole_height(pos: np.ndarray, quat: np.ndarray, names: list[str]) -> np.ndarray:
    """Lowest point over both feet's collision capsules per frame (the lowest point of a capsule is at an end)."""
    lowest = np.full(pos.shape[0], np.inf)
    for body, capsules in FOOT_CAPSULES.items():
        b = names.index(body)
        pts = np.array([p for cap in capsules for p in cap])  # (14, 3)
        world = pos[:, b, None, :] + _quat_rotate(quat[:, b, None, :], pts[None])
        lowest = np.minimum(lowest, world[..., 2].min(-1) - FOOT_RADIUS)
    return lowest


def name_filter(name: str) -> str | None:
    n = name.lower()
    for k in KEYWORDS:
        if k in n:
            return f"name:{k}"
    return None


def feasibility_filter(d: dict, names: list[str]) -> str | None:
    pos, quat, jv, lin = d["body_pos_w"], d["body_quat_w"], d["joint_vel"], d["body_lin_vel_w"]
    zmin = pos[:, :, 2].min(1)
    if _longest_run(zmin > ELEVATED_Z) >= ELEVATED_S * FPS:
        return "elevated_support"
    if (zmin < PENETRATE_Z_HARD).any() or (zmin < PENETRATE_Z_SOFT).mean() > PENETRATE_FRAC:
        return "ground_penetration"
    pel = names.index("pelvis")
    q = quat[:, pel]
    gz = -(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
    upright = gz < -UPRIGHT_GZ
    standing = (pos[:, pel, 2] > STAND_Z[0]) & (pos[:, pel, 2] < STAND_Z[1])
    still = (np.linalg.norm(lin[:, pel], axis=1) < STILL_V) & (np.sqrt((jv**2).mean(1)) < STILL_QD)
    gate = upright & standing & still
    if _longest_run(gate) >= FLOAT_S * FPS:
        floating = gate & (_foot_sole_height(pos, quat, names) > FLOAT_Z)
        if _longest_run(floating) >= FLOAT_S * FPS:
            return "floating_feet"
    return None


def _check_one(item: tuple) -> dict:
    key, path, apply_name, apply_feas, ref_joints, ref_bodies = item
    out = {"key": key, "path": path, "frames": 0, "reason": None}
    try:
        with np.load(path) as z:
            d = {k: z[k] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
            fps = float(z["fps"])
            joints = [str(x) for x in z["robot_joint_names"]]
            bodies = [str(x) for x in z["robot_body_names"]]
    except Exception as e:  # noqa: BLE001
        out["reason"] = f"invalid:load:{type(e).__name__}"
        return out
    out["frames"] = int(d["joint_pos"].shape[0])
    if fps != FPS:
        out["reason"] = f"invalid:fps:{fps}"
    elif joints != ref_joints or bodies != ref_bodies:
        out["reason"] = "invalid:name_order"
    elif out["frames"] < 2:
        out["reason"] = "invalid:too_short"
    elif not all(np.isfinite(v).all() for v in d.values()):
        out["reason"] = "invalid:non_finite"
    elif apply_name and name_filter(Path(path).stem):
        out["reason"] = name_filter(Path(path).stem)
    elif apply_feas:
        out["reason"] = feasibility_filter(d, bodies)
    return out


def _records(json_path: Path) -> list[dict]:
    return json.load(open(json_path))["records"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bones_eval_clips", type=int, default=2000)
    parser.add_argument("--any4hdmi_eval_clips", type=int, default=500)
    parser.add_argument("--workers", type=int, default=min(mp.cpu_count(), 96))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.datasets_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(root / "lafan_g1_v1/motions/dance1_subject1.npz") as z:
        ref_joints = [str(x) for x in z["robot_joint_names"]]
        ref_bodies = [str(x) for x in z["robot_body_names"]]

    # (key, path, apply_name_filter, apply_feasibility_filter, set, source)
    items: list[tuple] = []
    bones = json.load(open(root / "bones_seed_g1_v1/metadata/converted.json"))
    for rid, rec in bones.items():
        target = "eval" if rec["split"] == "test" else "train"
        items.append((f"bones/{rid}", str(root / "bones_seed_g1_v1" / rec["path"]), True, True, target, "bones"))
    for split, target in (("train", "train"), ("validation", "train"), ("test", "eval")):
        for rec in _records(root / f"any4hdmi_g1_v1/{split}.json"):
            key = "a4h/" + Path(rec["path"]).stem
            items.append((key, str(root / rec["path"]), False, True, target, "any4hdmi"))
    for rec in _records(root / "lafan_g1_v1/full.json"):
        items.append(("lafan/" + Path(rec["path"]).stem, str(root / rec["path"]), False, False, "train", "lafan"))
    for rec in _records(root / "lafan_g1_v1/test.json"):
        key = "lafan_test/" + rec["id"].split(":", 1)[1].replace(":", "_")
        items.append((key, str(root / rec["path"]), False, False, "eval", "lafan"))

    keys = [it[0] for it in items]
    dup = [k for k, c in Counter(keys).items() if c > 1]
    assert not dup, f"duplicate motion keys: {dup[:10]}"
    print(f"checking {len(items)} motions with {args.workers} workers...")

    with mp.Pool(args.workers) as pool:
        results = pool.map(
            _check_one, [(k, p, n, f, ref_joints, ref_bodies) for k, p, n, f, _, _ in items], chunksize=64
        )

    rng = random.Random(args.seed)
    report = {"keywords": KEYWORDS, "sources": {}}
    selected = {"train": {}, "eval": {}}
    eval_pool: dict[str, list] = {"bones": [], "any4hdmi": [], "lafan": []}
    for (key, path, _, _, target, source), res in zip(items, results):
        stats = report["sources"].setdefault(f"{source}/{target}", {"total": 0, "kept": 0, "frames_kept": 0, "reasons": Counter()})
        stats["total"] += 1
        if res["reason"] is not None:
            stats["reasons"][res["reason"].split(":")[0] if res["reason"].startswith("name:") else res["reason"]] += 1
            continue
        stats["kept"] += 1
        stats["frames_kept"] += res["frames"]
        if target == "train":
            selected["train"][key] = path
        else:
            eval_pool[source].append((key, path))

    for source, limit in (("bones", args.bones_eval_clips), ("any4hdmi", args.any4hdmi_eval_clips), ("lafan", None)):
        pool_items = sorted(eval_pool[source])
        if limit is not None and len(pool_items) > limit:
            pool_items = rng.sample(pool_items, limit)
        selected["eval"].update(dict(sorted(pool_items)))

    for name in ("train", "eval"):
        with open(out_dir / f"{name}.yaml", "w") as f:
            yaml.safe_dump(selected[name], f, sort_keys=False)

    for stats in report["sources"].values():
        stats["hours_kept"] = round(stats["frames_kept"] / FPS / 3600, 3)
        stats["reasons"] = dict(stats["reasons"].most_common())
    report["num_train"] = len(selected["train"])
    report["num_eval"] = len(selected["eval"])
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["sources"], indent=2))
    print(f"train: {report['num_train']} motions, eval: {report['num_eval']} motions -> {out_dir}")


if __name__ == "__main__":
    main()

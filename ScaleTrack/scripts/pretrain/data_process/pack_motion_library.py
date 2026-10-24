"""Pack the motions listed in a YAML file into one contiguous motion library.

The packed library stores, for the tracked bodies only, the same arrays the motion command reads from the
individual npz files, concatenated along time:

    <output_dir>/index.json                  motion names, lengths, body and joint names
    <output_dir>/joint_pos.npy               (F, J) float32
    <output_dir>/joint_vel.npy               (F, J)
    <output_dir>/body_pos_w.npy              (F, B, 3)
    <output_dir>/body_quat_w.npy             (F, B, 4) wxyz
    <output_dir>/body_lin_vel_w.npy          (F, B, 3)
    <output_dir>/body_ang_vel_w.npy          (F, B, 3)

Loading ~140k compressed npz files takes a long time at every start; the packed arrays can be memory-mapped
and sliced per rank in seconds. Put the output on a fast local file system (e.g. /dev/shm).

Example:
    python scripts/pretrain/data_process/pack_motion_library.py \
        --motion_file /path/to/train.yaml --output_dir /dev/shm/sm_lib/train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import numpy as np
import shutil
import tempfile
import time
import yaml
from pathlib import Path

# Tracked bodies of G1-BFM-Transformer-Tracking (config/g1_29dof/flat_env_cfg.py), in command order.
G1_TRACKED_BODIES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subset_library(source: Path, manifest: Path, output: Path) -> None:
    """Copy selected packed motions in bounded memory, validate bit identity, publish atomically."""
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    index = json.loads((source / "index.json").read_text())
    names = list(yaml.safe_load(manifest.read_text()))
    if not names or len(set(index["motion_names"])) != len(index["motion_names"]):
        raise ValueError("empty selection or duplicate source names")
    lookup = {n: i for i, n in enumerate(index["motion_names"])}
    selected = [lookup[n] for n in names]
    lengths = [index["lengths"][i] for i in selected]
    starts = np.concatenate(([0], np.cumsum(index["lengths"], dtype=np.int64)))
    spans = []
    for i in selected:
        begin, end = int(starts[i]), int(starts[i + 1])
        if spans and spans[-1][1] == begin:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((begin, end))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        hashes = {}
        for field in FIELDS:
            original = np.load(source / f"{field}.npy", mmap_mode="r")
            if original.shape[0] != starts[-1]:
                raise ValueError(f"{field}: source frame count differs from index")
            path = temporary / f"{field}.npy"
            result = np.lib.format.open_memmap(
                path, mode="w+", dtype=original.dtype, shape=(sum(lengths), *original.shape[1:])
            )
            # At most 64 MiB per copy/verification; coalescing avoids 138k tiny writes.
            chunk_frames = max(1, (64 * 2**20) // original[0].nbytes)
            destination = 0
            for begin, end in spans:
                for offset in range(begin, end, chunk_frames):
                    block = original[offset : min(end, offset + chunk_frames)]
                    result[destination : destination + len(block)] = block
                    destination += len(block)
            result.flush()
            # Read from the completed file and compare every byte with the selected source.
            verified = np.load(path, mmap_mode="r")
            destination = 0
            for begin, end in spans:
                for offset in range(begin, end, chunk_frames):
                    block = original[offset : min(end, offset + chunk_frames)]
                    if not np.array_equal(
                        verified[destination : destination + len(block)].view(np.uint8), block.view(np.uint8)
                    ):
                        raise ValueError(f"{field}: subset differs from source")
                    destination += len(block)
            del result, verified, original
            hashes[path.name] = file_hash(path)
            print(f"validated {field}: {sum(lengths)} frames", flush=True)
        new_index = index | dict(motion_names=names, lengths=lengths, source_motion_file=str(manifest.resolve()))
        (temporary / "index.json").write_text(json.dumps(new_index))
        hashes["index.json"] = file_hash(temporary / "index.json")
        lineage = dict(
            source_library=str(source.resolve()),
            source_index_sha256=file_hash(source / "index.json"),
            manifest_sha256=file_hash(manifest),
            output_sha256=hashes,
            validation="all selected array bytes equal source; dtype and order preserved",
        )
        (temporary / "lineage.json").write_text(json.dumps(lineage, indent=2) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def _load(item: tuple[str, str, list[str]]):
    name, path, bodies = item
    with np.load(path) as z:
        if float(z["fps"]) != 50:
            raise ValueError(f"{name}: motion must be 50 FPS")
        names = [str(x) for x in z["robot_body_names"]]
        idx = [names.index(b) for b in bodies]
        out = {
            k: np.ascontiguousarray(z[k][:, idx] if k.startswith("body") else z[k], dtype=np.float32) for k in FIELDS
        }
        joints = [str(x) for x in z["robot_joint_names"]]
    return name, out, joints


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=min(mp.cpu_count(), 96))
    parser.add_argument("--source_library", type=Path, help="subset this packed library instead of unpacking NPZs")
    args = parser.parse_args()

    if args.source_library is not None:
        subset_library(args.source_library, Path(args.motion_file), Path(args.output_dir))
        return

    with open(args.motion_file) as f:
        motions = yaml.safe_load(f)
    names = list(motions.keys())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    chunks = {k: [] for k in FIELDS}
    lengths = []
    joint_names = None
    with mp.Pool(args.workers) as pool:
        items = [(n, motions[n], G1_TRACKED_BODIES) for n in names]
        for i, (name, arrays, joints) in enumerate(pool.imap(_load, items, chunksize=32)):
            assert name == names[i]
            if joint_names is None:
                joint_names = joints
            elif joints != joint_names:
                raise ValueError(f"{name}: joint order differs from the first motion")
            for k in FIELDS:
                chunks[k].append(arrays[k])
            lengths.append(int(arrays["joint_pos"].shape[0]))
            if (i + 1) % 10000 == 0:
                print(f"loaded {i + 1}/{len(names)} motions ({time.time() - t0:.0f}s)", flush=True)

    for k in FIELDS:
        np.save(out_dir / f"{k}.npy", np.concatenate(chunks[k]))
        chunks[k] = None
    index = {
        "motion_names": names,
        "lengths": lengths,
        "body_names": G1_TRACKED_BODIES,
        "joint_names": joint_names,
        "fps": 50,
        "source_motion_file": str(Path(args.motion_file).resolve()),
    }
    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f)
    print(f"packed {len(names)} motions / {sum(lengths)} frames into {out_dir} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

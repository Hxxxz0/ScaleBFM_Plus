"""GPU motion library loaded from a packed library (scripts/pretrain/data_process/pack_motion_library.py).

All motions are concatenated along one global frame axis. A library can hold only a shard of the motions so that
every data-parallel rank keeps a different part of a large dataset on its GPU.
"""

from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path

FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def shard_motions(lengths: np.ndarray, rank: int, world_size: int, seed: int = 0) -> np.ndarray:
    """Greedy frame-balanced split of motions over ranks; returns the sorted motion indices of ``rank``."""
    order = np.random.default_rng(seed).permutation(len(lengths))
    order = order[np.argsort(-lengths[order], kind="stable")]
    load = np.zeros(world_size, dtype=np.int64)
    owner = np.empty(len(lengths), dtype=np.int64)
    for m in order:
        r = int(np.argmin(load))
        owner[m] = r
        load[r] += lengths[m]
    return np.sort(np.nonzero(owner == rank)[0])


class MotionLibrary:
    def __init__(
        self,
        lib_dir: str,
        body_names: list[str],
        joint_names: list[str],
        device,
        rank: int = 0,
        world_size: int = 1,
        bin_frames: int = 50,
        seed: int = 0,
    ):
        lib_dir = Path(lib_dir)
        with open(lib_dir / "index.json") as f:
            index = json.load(f)
        if index["body_names"] != list(body_names):
            raise ValueError(f"packed library bodies {index['body_names']} differ from the command bodies {body_names}")
        if index["joint_names"] != list(joint_names):
            raise ValueError("packed library joint order differs from the simulator joint order")
        all_lengths = np.asarray(index["lengths"], dtype=np.int64)
        all_starts = np.concatenate([[0], np.cumsum(all_lengths)[:-1]])
        motions = shard_motions(all_lengths, rank, world_size, seed) if world_size > 1 else np.arange(len(all_lengths))

        self.device = torch.device(device)
        self.motion_names = [index["motion_names"][i] for i in motions]
        # Use the same global group order on every data shard, including empty groups.
        self.group_names = sorted({name.split("/", 1)[0] for name in index["motion_names"]})
        group_ids = {name: i for i, name in enumerate(self.group_names)}
        self.motion_group = torch.tensor(
            [group_ids[name.split("/", 1)[0]] for name in self.motion_names], device=self.device, dtype=torch.long
        )
        lengths = all_lengths[motions]
        rows = np.concatenate([np.arange(all_starts[m], all_starts[m] + all_lengths[m]) for m in motions])
        for k in FIELDS:
            arr = np.load(lib_dir / f"{k}.npy", mmap_mode="r")
            setattr(self, k, torch.from_numpy(np.ascontiguousarray(arr[rows])).to(self.device))

        self.num_motions = len(motions)
        self.num_frames = int(lengths.sum())
        self.lengths = torch.from_numpy(lengths).to(self.device)
        self.starts = torch.zeros_like(self.lengths)
        self.starts[1:] = torch.cumsum(self.lengths, 0)[:-1]
        self.frame_motion = torch.repeat_interleave(torch.arange(self.num_motions, device=self.device), self.lengths)
        self.frame_last = (self.starts + self.lengths - 1)[self.frame_motion]

        # 1 s bins of every motion (the last bin of a motion may be shorter).
        bins_per_motion = (self.lengths + bin_frames - 1) // bin_frames
        self.num_bins = int(bins_per_motion.sum())
        self.bin_motion = torch.repeat_interleave(torch.arange(self.num_motions, device=self.device), bins_per_motion)
        first_bin = torch.zeros_like(bins_per_motion)
        first_bin[1:] = torch.cumsum(bins_per_motion, 0)[:-1]
        local_bin = torch.arange(self.num_bins, device=self.device) - first_bin[self.bin_motion]
        self.bin_start = self.starts[self.bin_motion] + local_bin * bin_frames
        self.bin_end = torch.minimum(
            self.bin_start + bin_frames, self.starts[self.bin_motion] + self.lengths[self.bin_motion]
        )
        frame_local = torch.arange(self.num_frames, device=self.device) - self.starts[self.frame_motion]
        self.frame_bin = first_bin[self.frame_motion] + frame_local // bin_frames

    def future(self, frame: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Global frames frame + offsets clamped to the end of each frame's motion; frame (...,), offsets (..., F)."""
        return torch.minimum(frame[..., None] + offsets, self.frame_last[frame][..., None])

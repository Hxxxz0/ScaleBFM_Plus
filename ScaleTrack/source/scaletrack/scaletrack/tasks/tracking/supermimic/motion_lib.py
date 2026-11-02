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
# Optional per-frame arrays of libraries with one dynamic object per motion (HOI): the object reference (zeros and
# the identity for motions without an object) and the frames an episode may start from.
OBJECT_FIELDS = ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w", "start_ok")


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
        self.has_objects = (lib_dir / "object_pos_w.npy").exists()
        if self.has_objects:
            for k in OBJECT_FIELDS:
                arr = np.load(lib_dir / f"{k}.npy", mmap_mode="r")
                setattr(self, k, torch.from_numpy(np.ascontiguousarray(arr[rows])).to(self.device))
            # Per motion: which object (index into ``object_names``, -1: none) and the local frame its evaluation
            # starts from (the audited start). Single-object libraries only list ``object_valid``.
            if "object_ids" in index:
                ids = np.asarray(index["object_ids"], dtype=np.int64)
                self.object_names = list(index["object_names"])
            else:
                ids = np.where(np.asarray(index["object_valid"], dtype=bool), 0, -1)
                self.object_names = [index.get("object_name", "object")]
            self.object_id = torch.tensor(ids[motions], device=self.device)
            self.object_valid = self.object_id >= 0
            self.eval_start = torch.tensor(np.asarray(index["eval_start"], dtype=np.int64)[motions], device=self.device)
        # Reference contact labels (add_contact_labels.py sidecar): (frames, B) int8, +1 / -1 / 0 per body.
        self.has_contact_labels = (lib_dir / "contact_labels.npy").exists()
        if self.has_contact_labels:
            meta = json.loads((lib_dir / "contact_labels.json").read_text())
            self.contact_body_names = list(meta["body_names"])
            arr = np.load(lib_dir / "contact_labels.npy", mmap_mode="r")
            self.contact_labels = torch.from_numpy(np.ascontiguousarray(arr[rows])).to(self.device)
            self.contact_valid = torch.tensor(np.asarray(meta["valid"], dtype=bool)[motions], device=self.device)

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
        if self.has_objects:
            self.frame_object_valid = self.object_valid[self.frame_motion].float()
            self.bin_object = self.object_id[self.bin_motion]
            # Allowed start frames grouped by bin, for sampling a start uniformly among the allowed frames of a bin.
            ok = torch.nonzero(self.start_ok).squeeze(-1)
            self.ok_frames = ok
            self.bin_ok_count = torch.zeros(self.num_bins, dtype=torch.long, device=self.device).index_add_(
                0, self.frame_bin[ok], torch.ones_like(ok)
            )
            self.bin_ok_start = torch.cumsum(self.bin_ok_count, 0) - self.bin_ok_count

    def contact_token(self, frame: torch.Tensor, offsets: torch.Tensor, num_bodies: int) -> torch.Tensor:
        """(N, 1, 1, F*B + 1) reference contact labels at frame + offsets (F) with the motion's label validity last;
        zero and invalid for motions (or libraries) without labels."""
        valid = torch.zeros(frame.shape[0], 1, device=frame.device)
        if not self.has_contact_labels:
            return torch.cat([torch.zeros(frame.shape[0], offsets.shape[-1] * num_bodies, device=frame.device), valid], -1)[:, None, None]
        valid = self.contact_valid[self.frame_motion[frame]].float()[:, None]
        labels = self.contact_labels[self.future(frame, offsets.expand(frame.shape[0], -1))].float().flatten(1)
        return torch.cat([labels * valid, valid], -1)[:, None, None]

    def contact_labels_at(self, frame: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        """(N, L) labels of the label columns ``cols`` at ``frame``; 0 for motions (or libraries) without labels."""
        if not self.has_contact_labels:
            return torch.zeros(frame.shape[0], cols.shape[0], device=frame.device)
        return self.contact_labels[frame][:, cols].float() * self.contact_valid[self.frame_motion[frame]].float()[:, None]

    def future(self, frame: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Global frames frame + offsets clamped to the end of each frame's motion; frame (...,), offsets (..., F)."""
        return torch.minimum(frame[..., None] + offsets, self.frame_last[frame][..., None])

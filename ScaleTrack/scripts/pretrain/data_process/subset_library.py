"""Copy a random subset of the motions of a packed library (motion_lib.py format) into a new library.

Used to evaluate on training motions with the test protocol: by default the subset has the per-object composition of
a reference library (e.g. the held-out eval library), so train and test metrics are comparable.

    /isaac-sim/python.sh scripts/pretrain/data_process/subset_library.py \
        --library <sm_lib>/hdmi_inspire_v1/train --like <sm_lib>/hdmi_inspire_v1/eval \
        --output_dir <sm_lib>/hdmi_inspire_v1/train_probe
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--like", type=Path, help="library whose number of motions per object id is reproduced")
    ap.add_argument("--per_object", type=int, default=0, help="motions per object id (without --like)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    index = json.loads((args.library / "index.json").read_text())
    ids = np.asarray(index.get("object_ids", [-1] * len(index["lengths"])), dtype=np.int64)
    if args.like:
        wanted = collections.Counter(json.loads((args.like / "index.json").read_text())["object_ids"])
    else:
        wanted = {int(i): args.per_object for i in np.unique(ids)}
    rng = np.random.default_rng(args.seed)
    motions = np.sort(np.concatenate([
        rng.choice(np.nonzero(ids == i)[0], size=min(n, int((ids == i).sum())), replace=False)
        for i, n in sorted(wanted.items())
    ]))  # fmt: skip

    lengths = np.asarray(index["lengths"], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    rows = np.concatenate([np.arange(starts[m], starts[m] + lengths[m]) for m in motions])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.library.glob("*.npy")):
        np.save(args.output_dir / path.name, np.load(path, mmap_mode="r")[rows])
    for path in args.library.iterdir():
        if path.suffix not in (".npy",) and path.name != "index.json" and path.is_file():
            shutil.copy2(path, args.output_dir / path.name)

    per_motion = {k for k, v in index.items() if isinstance(v, list) and len(v) == len(lengths)}
    out = {k: ([v[m] for m in motions] if k in per_motion else v) for k, v in index.items()}
    out["subset_of"] = str(args.library)
    (args.output_dir / "index.json").write_text(json.dumps(out))
    counts = collections.Counter(ids[motions].tolist())
    print(f"{len(motions)} motions, {len(rows)} frames -> {args.output_dir}; per object id {sorted(counts.items())}")


if __name__ == "__main__":
    main()

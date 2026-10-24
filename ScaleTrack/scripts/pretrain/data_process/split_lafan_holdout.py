"""Remove held-out LAFAN parents and duplicate any4hdmi copies from an existing library.

The evaluation index defines the holdout. Other datasets keep their existing split.
The result can be passed to pack_motion_library.py --source_library without
decompressing or modifying any original motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import yaml
from pathlib import Path


def lafan_parent(name: str) -> str | None:
    if name.startswith("lafan/"):
        return name.split("/", 1)[1]
    if name.startswith("a4h/any4hdmi__lafan__"):
        return name.split("__lafan__", 1)[1]
    if name.startswith("lafan_test/"):
        match = re.fullmatch(r"lafan_test/(.+)_\d{6}_\d{6}", name)
        if match is None:
            raise ValueError(f"unrecognized LAFAN evaluation segment: {name}")
        return match[1]
    return None


def select_training(train: dict, evaluation: dict) -> tuple[list[str], dict]:
    held_out = {p for n in evaluation["motion_names"] if (p := lafan_parent(n)) is not None}
    standalone = {lafan_parent(n) for n in train["motion_names"] if n.startswith("lafan/")}
    kept, removed = [], []
    for name, length in zip(train["motion_names"], train["lengths"], strict=True):
        parent = lafan_parent(name)
        reason = None
        if parent in held_out:
            reason = "held_out_parent"
        elif name.startswith("a4h/any4hdmi__lafan__") and parent in standalone:
            reason = "duplicate_lafan_parent"
        if reason:
            removed.append(dict(name=name, frames=length, parent=parent, reason=reason))
        else:
            kept.append(name)
    remaining = {lafan_parent(n) for n in kept if lafan_parent(n) is not None}
    if remaining & held_out:
        raise ValueError("LAFAN parent overlap remains after filtering")
    return kept, dict(
        held_out_parents=sorted(held_out),
        removed=removed,
        train_motions=len(kept),
        train_frames=sum(train["lengths"]) - sum(r["frames"] for r in removed),
        lafan_train_parents=sorted(remaining),
        lafan_parent_overlap=[],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_library", type=Path, required=True)
    parser.add_argument("--eval_library", type=Path, required=True)
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output already exists: {args.output_dir}")
    train_path, eval_path = args.train_library / "index.json", args.eval_library / "index.json"
    kept, report = select_training(json.loads(train_path.read_text()), json.loads(eval_path.read_text()))
    manifest = yaml.safe_load(args.train_manifest.read_text())
    filtered = {name: manifest[name] for name in kept}
    report["inputs"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_path, eval_path, args.train_manifest)
    }
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "train.yaml"
    output.write_text(yaml.safe_dump(filtered, sort_keys=False))
    report["manifest_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (args.output_dir / "split_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "removed"}, indent=2))


if __name__ == "__main__":
    main()

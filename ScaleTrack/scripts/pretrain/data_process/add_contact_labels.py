"""Add the per-frame reference contact labels of the source datasets to a packed library (sidecar, nothing rewritten).

Weave ships one int8 label per frame and robot body (+1 expected contact with the object, -1 contact forbidden, 0
neutral; contact_manifest.json). Library motions are routed to their source file through the converted parts'
index.json (``source_files``, convert_any4hdmi.py); motions without labels (OMOMO, flat ground) get 0 everywhere and
are flagged invalid.

    <library>/contact_labels.npy    (frames, B) int8, rows aligned with the library's other fields
    <library>/contact_labels.json   body_names (B), per-motion valid flags

    python scripts/pretrain/data_process/add_contact_labels.py --library <sm_lib>/hdmi_inspire_v1/train \
        --parts <converted>/train <converted>/eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def source_routes(parts: list[Path]) -> dict[str, str]:
    """Library motion name -> source motion file, over every converted object directory below ``parts``."""
    routes = {}
    for root in parts:
        for index in sorted(root.glob("*/index.json")):
            meta = json.loads(index.read_text())
            routes.update(zip(meta["motion_names"], meta["source_files"]))
    return routes


class ContactSidecars:
    """Labels of one source motion file, read from the contact sidecar of its dataset directory."""

    def __init__(self):
        self.datasets = {}

    def labels(self, source: str) -> tuple[list[str], np.ndarray] | None:
        root = Path(source).parent.parent
        if root not in self.datasets:
            manifest = root / "contact_manifest.json"
            if not manifest.exists():
                self.datasets[root] = None
            else:
                meta = json.loads(manifest.read_text())
                path = root / meta["labels_file"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != meta["labels_sha256"]:
                    raise ValueError(f"contact label checksum mismatch: {path}")
                offsets = np.concatenate([[0], np.cumsum(meta["motion_lengths"])])
                rows = {f: (offsets[i], offsets[i + 1]) for i, f in enumerate(meta["motion_files"])}
                self.datasets[root] = (meta["body_names"], np.load(path, mmap_mode="r"), rows)
        entry = self.datasets[root]
        if entry is None:
            return None
        names, data, rows = entry
        lo, hi = rows[Path(source).relative_to(root).as_posix()]
        return names, np.asarray(data[lo:hi])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--parts", type=Path, nargs="+", required=True, help="converted split directories")
    args = ap.parse_args()

    index = json.loads((args.library / "index.json").read_text())
    routes = source_routes(args.parts)
    sidecars = ContactSidecars()
    body_names, chunks, valid = None, [], []
    for name, length in zip(index["motion_names"], index["lengths"]):
        found = sidecars.labels(routes[name]) if name in routes else None
        if found is not None:
            names, labels = found
            if body_names is None:
                body_names = names
            elif names != body_names:
                raise ValueError(f"{name}: contact body names differ between datasets")
            if len(labels) != length:
                raise ValueError(f"{name}: {len(labels)} contact frames for a {length}-frame motion")
        chunks.append(labels if found is not None else None)
        valid.append(found is not None)
    if body_names is None:
        raise ValueError("no motion of the library has contact labels")
    zeros = np.zeros((0, len(body_names)), dtype=np.int8)
    labels = np.concatenate(
        [c if c is not None else np.zeros((n, len(body_names)), dtype=np.int8) for c, n in zip(chunks, index["lengths"])]
        or [zeros]
    )
    np.save(args.library / "contact_labels.npy", labels.astype(np.int8))
    (args.library / "contact_labels.json").write_text(json.dumps({"body_names": body_names, "valid": valid}))
    print(f"{sum(valid)}/{len(valid)} motions with contact labels, {labels.shape} -> {args.library}")


if __name__ == "__main__":
    main()

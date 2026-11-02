"""Persist torch.compile artifacts across jobs (torch.compiler Mega-Cache).

Compiling the imagination graphs takes ~10 minutes on every fresh node. The artifacts are saved once per
(torch version, GPU architecture and index, compiled source code, tensor shapes) key and hot-loaded by later jobs, so
that only Dynamo tracing (seconds) remains.

A source edit changes the key, but the artifacts are content addressed (graph hashes): a miss is seeded with the
latest artifacts of the same configuration (``<lineage>.latest``), or of the code trees listed in
``SCALETRACK_COMPILE_CACHE_FROM_CODE`` (colon separated roots of earlier code copies), so that only the graphs whose
code changed are compiled again.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


def source_digest(*directories: str | Path) -> str:
    """SHA-256 over the Python sources below ``directories`` (any change invalidates the cache)."""
    digest = hashlib.sha256()
    for directory in directories:
        for path in sorted(Path(directory).rglob("*.py")):
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _name(key: dict) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:20]


class CompileCache:
    def __init__(self, directory: str | Path | None, key: dict, sources: tuple[Path, ...] = (), root: Path | None = None):
        """``sources``: directories whose Python code is compiled; ``root``: the code tree that contains them."""
        self.path = None
        self.loaded: Path | None = None
        if directory:
            directory = Path(directory)
            # Compiled kernels are specialised to the device index: every local rank keeps its own entry.
            key = dict(
                key, torch=torch.__version__, capability=torch.cuda.get_device_capability(), device=torch.cuda.current_device()
            )
            self.path = directory / f"{_name(dict(key, source=source_digest(*sources)))}.bin"
            self.pointer = directory / f"{_name(key)}.latest"
            self.fallbacks = []
            for old_root in filter(None, os.environ.get("SCALETRACK_COMPILE_CACHE_FROM_CODE", "").split(":")):
                old_sources = [Path(old_root) / Path(d).resolve().relative_to(root) for d in sources]
                self.fallbacks.append(directory / f"{_name(dict(key, source=source_digest(*old_sources)))}.bin")

    def load(self) -> bool:
        if self.path is None:
            return False
        candidates = [self.path]
        if self.pointer.is_file():
            candidates.append(self.path.with_name(self.pointer.read_text().strip()))
        for path in candidates + self.fallbacks:
            if path.is_file():
                torch.compiler.load_cache_artifacts(path.read_bytes())
                self.loaded = path
                return True
        return False

    def save(self) -> bool:
        """Write the artifacts compiled so far, unless an entry for this key exists already."""
        if self.path is None or self.path.is_file():
            return False
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_bytes(artifacts[0])
        temporary.replace(self.path)
        temporary = self.pointer.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(self.path.name)
        temporary.replace(self.pointer)
        return True

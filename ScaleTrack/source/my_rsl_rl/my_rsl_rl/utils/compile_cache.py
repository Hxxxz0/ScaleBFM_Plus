"""Persist torch.compile artifacts across jobs (torch.compiler Mega-Cache).

Compiling the imagination graphs takes ~10 minutes on every fresh node. The artifacts are saved once per
(torch version, GPU architecture and index, compiled source code, tensor shapes) key and hot-loaded by later jobs, so
that only Dynamo tracing (seconds) remains.
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


class CompileCache:
    def __init__(self, directory: str | Path | None, key: dict):
        self.path = None
        if directory:
            # Compiled kernels are specialised to the device index: every local rank keeps its own entry.
            key = dict(
                key, torch=torch.__version__, capability=torch.cuda.get_device_capability(), device=torch.cuda.current_device()
            )
            name = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:20]
            self.path = Path(directory) / f"{name}.bin"

    def load(self) -> bool:
        if self.path is None or not self.path.is_file():
            return False
        torch.compiler.load_cache_artifacts(self.path.read_bytes())
        return True

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
        return True

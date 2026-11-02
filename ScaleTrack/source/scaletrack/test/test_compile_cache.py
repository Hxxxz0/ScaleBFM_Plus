import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from my_rsl_rl.utils.compile_cache import CompileCache, source_digest


def _tree(root: Path, code: str) -> Path:
    package = root / "source" / "pkg"
    package.mkdir(parents=True)
    (package / "algo.py").write_text(code)
    return package


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class CompileCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cache = self.tmp / "cache"
        self.key = {"policy": {"dims": [1, 2]}, "shapes": (2048, 0, 32)}
        self.loaded = []
        patches = [
            mock.patch.object(torch.compiler, "load_cache_artifacts", side_effect=self.loaded.append),
            mock.patch.object(torch.compiler, "save_cache_artifacts", return_value=(b"artifacts", None)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_name_matches_the_previous_key_format(self):
        package = _tree(self.tmp / "code", "x = 1\n")
        cache = CompileCache(self.cache, self.key, sources=(package,), root=self.tmp / "code")
        old_key = dict(
            self.key, source=source_digest(package), torch=torch.__version__,
            capability=torch.cuda.get_device_capability(), device=torch.cuda.current_device(),
        )  # fmt: skip
        name = hashlib.sha256(json.dumps(old_key, sort_keys=True, default=str).encode()).hexdigest()[:20]
        self.assertEqual(cache.path.name, f"{name}.bin")

    def test_source_edit_loads_latest_artifacts_of_the_configuration(self):
        old = _tree(self.tmp / "old", "x = 1\n")
        self.assertTrue(CompileCache(self.cache, self.key, sources=(old,), root=self.tmp / "old").save())
        new = _tree(self.tmp / "new", "x = 2\n")
        cache = CompileCache(self.cache, self.key, sources=(new,), root=self.tmp / "new")
        self.assertTrue(cache.load())
        self.assertNotEqual(cache.loaded, cache.path)
        self.assertEqual(self.loaded, [b"artifacts"])
        self.assertTrue(cache.save())  # the new key gets its own entry and becomes the latest
        self.assertEqual((self.cache / cache.pointer.name).read_text(), cache.path.name)

    def test_artifacts_saved_without_pointer_are_found_from_an_old_code_tree(self):
        old = _tree(self.tmp / "old", "x = 1\n")
        saved = CompileCache(self.cache, self.key, sources=(old,), root=self.tmp / "old")
        self.assertTrue(saved.save())
        saved.pointer.unlink()  # entries written before pointers existed
        new = _tree(self.tmp / "new", "x = 2\n")
        with mock.patch.dict(os.environ, {"SCALETRACK_COMPILE_CACHE_FROM_CODE": str(self.tmp / "old")}):
            cache = CompileCache(self.cache, self.key, sources=(new,), root=self.tmp / "new")
        self.assertTrue(cache.load())
        self.assertEqual(cache.loaded, saved.path)

    def test_other_configuration_does_not_load(self):
        old = _tree(self.tmp / "old", "x = 1\n")
        self.assertTrue(CompileCache(self.cache, self.key, sources=(old,), root=self.tmp / "old").save())
        other = dict(self.key, shapes=(1024, 0, 32))
        self.assertFalse(CompileCache(self.cache, other, sources=(old,), root=self.tmp / "old").load())


if __name__ == "__main__":
    unittest.main()

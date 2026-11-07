import os
from pathlib import Path
import subprocess
import sys
import unittest

from mlx_agent._vision_cache import FeatureCache, content_key


class VisionCacheTests(unittest.TestCase):
    def test_content_identity_includes_every_buffer_and_its_layout(self):
        pixels = ("float32", (2, 2), b"abcdefgh")
        grid = ("int32", (1, 3), b"ijkl")
        key = content_key((pixels, grid))
        self.assertEqual(content_key((pixels, grid)), key)
        for parts in ((("float16", pixels[1], pixels[2]), grid),
                      ((pixels[0], (1, 4), pixels[2]), grid),
                      ((pixels[0], pixels[1], b"abcdefgz"), grid),
                      (pixels, (grid[0], grid[1], b"ijzz")), (grid, pixels)):
            self.assertNotEqual(content_key(parts), key)

    def test_byte_and_entry_bounds_evict_the_least_recently_used_features(self):
        cache = FeatureCache(max_bytes=6, max_entries=2)
        cache.put("a", b"one", 3)
        cache.put("b", b"two", 3)
        self.assertEqual(cache.get("a"), b"one")
        cache.put("c", b"tri", 3)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), b"one")
        self.assertEqual((cache.bytes, len(cache.entries)), (6, 2))
        self.assertFalse(cache.put("oversized", b"toolarge", 7))
        self.assertEqual(cache.bytes, 6)
        cache.clear()
        self.assertEqual((cache.bytes, len(cache.entries)), (0, 0))

    def test_invalid_bounds_are_rejected(self):
        for size, count in ((0, 1), (1, 0), (-1, 2), (True, 2)):
            with self.subTest(size=size, count=count), self.assertRaises(ValueError):
                FeatureCache(max_bytes=size, max_entries=count)

    @unittest.skipUnless(sys.platform == "darwin", "Requires the installed MLX-VLM backend")
    def test_real_vlm_embedding_checks_run_in_an_owned_process(self):
        from mlx_agent.backends import backend_python, is_installed, load_manifests
        manifest = load_manifests()["mlx-vlm"]
        if not is_installed(manifest):
            self.skipTest("Requires the installed MLX-VLM backend")
        root = Path(__file__).resolve().parents[2]
        run = subprocess.run([str(backend_python(manifest)), str(root/"tests/gpu_vision_cache.py"), "-v"],
            cwd=root, env=dict(os.environ, PYTHONPATH=str(root/"src")), capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)

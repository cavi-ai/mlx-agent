import json
import pathlib
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_agent.providers import ProviderRegistry, detect_providers


ROOT = Path(__file__).resolve().parents[2]


class ProviderPerformanceTests(unittest.TestCase):
    def registry_with_tree(self, root):
        manifest = json.loads((ROOT / "plugin.json").read_text())
        manifest["providers"] = {"agentskills": manifest["providers"]["agentskills"]}
        manifest["providers"]["agentskills"]["artifacts"] = [{"source": "tree", "destination": "skills/tree"}]
        (root / "plugin.json").write_text(json.dumps(manifest))
        return ProviderRegistry(root / "plugin.json", home=root / "home", config_root=root / "config")

    def test_directory_artifacts_are_listed_in_path_order_without_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            tree = root / "tree"
            for name in ("Z.txt", "a/b.txt", "a.txt", "nested/deep/x.md", "a/__pycache__/m.cpython-312.pyc", "c.pyc"):
                (tree / name).parent.mkdir(parents=True, exist_ok=True)
                (tree / name).write_text("fixture\n")
            artifacts = self.registry_with_tree(root).definitions()["agentskills"].artifacts
            self.assertEqual(["Z.txt", "a/b.txt", "a.txt", "nested/deep/x.md"],
                             [str(item.source.relative_to(tree)) for item in artifacts])
            self.assertEqual([Path("skills/tree") / item.source.relative_to(tree) for item in artifacts],
                             [item.destination for item in artifacts])
            (tree / "nested/link.md").symlink_to(tree / "Z.txt")
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.registry_with_tree(root).definitions()

    def test_listing_a_directory_artifact_does_not_resolve_each_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for index in range(60):
                (root / "tree" / "d{0}".format(index % 6)).mkdir(parents=True, exist_ok=True)
                (root / "tree" / "d{0}".format(index % 6) / "f{0}.md".format(index)).write_text("fixture\n")
            registry = self.registry_with_tree(root)
            original, calls = pathlib.Path.resolve, []

            def resolve(path, *args, **kwargs):
                calls.append(path)
                return original(path, *args, **kwargs)

            with patch.object(pathlib.Path, "resolve", resolve):
                artifacts = registry.definitions()["agentskills"].artifacts
        self.assertEqual(60, len(artifacts))
        self.assertLess(len(calls), 10)

    def test_version_probes_run_concurrently_and_keep_definition_order(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ProviderRegistry(ROOT / "plugin.json", home=Path(directory) / "home",
                                        config_root=Path(directory) / "config")
            definitions = list(registry.definitions().values())
        native = {"claude", "codex", "agy", "opencode"}

        def probe(argv, **kwargs):
            time.sleep(0.5)
            return subprocess.CompletedProcess(argv, 0, b"9.9.9\n")

        started = time.monotonic()
        detections = detect_providers(definitions, env={"PATH": "/fake/bin"},
                                      executable_lookup=lambda command, path=None: "/fake/bin/" + command
                                      if command in native else None, probe_runner=probe)
        elapsed = time.monotonic() - started
        self.assertEqual([item.id for item in definitions], [item.id for item in detections])
        self.assertEqual({"9.9.9"}, {item.version for item in detections if item.id in native})
        self.assertLess(elapsed, 1.5)


if __name__ == "__main__":
    unittest.main()

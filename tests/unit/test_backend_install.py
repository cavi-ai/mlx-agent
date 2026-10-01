import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from jsonschema import Draft202012Validator

from mlx_agent.backend_install import (
    BACKEND_RUNNER,
    list_backends,
    plan_install,
    plan_remove,
    remove_backend,
    start_install,
)
from mlx_agent.backends import INSTALL_MARKER, BackendError, backend_target, load_manifests

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "schemas" / "backends.schema.json").read_text(encoding="utf-8"))
PY312 = (3, 12, 0)


def golden_backend_list():
    with TemporaryDirectory() as directory:
        report = list_backends(root=Path("/backends"), receipts_dir=directory, find_spec=lambda name: None)
    report["backends"] = [dict(entry, log_path=None) for entry in report["backends"]]
    return report


class BackendInstallTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.root = self.base / "backends"
        self.receipts = self.base / "receipts"
        self.trash = self.base / "Trash"
        self.spawned = []

    def spawn(self, argv, log_path, env):
        self.spawned.append((argv, log_path, env))
        return 4242

    def plan(self, backend_id="mlx-audio"):
        return plan_install(backend_id, root=self.root, python="/venv/bin/python3.12", version_info=PY312, trash_dir=self.trash)

    def test_plan_is_deterministic_and_names_the_lock(self):
        first, second = self.plan(), self.plan()
        self.assertEqual(first["preview_hash"], second["preview_hash"])
        self.assertEqual(first["argv"][:2], ["/venv/bin/python3.12", str(BACKEND_RUNNER)])
        self.assertEqual(first["argv"][-2:], ["--preview-hash", first["preview_hash"]])
        self.assertTrue(first["lock"].endswith("mlx-audio.lock"))
        self.assertGreater(first["requirements"], 5)

    def test_plan_refusals(self):
        for backend_id, code in (("mlx-lm", "builtin_backend"), ("nope", "unknown_backend")):
            with self.subTest(backend_id=backend_id):
                with self.assertRaises(BackendError) as caught:
                    plan_install(backend_id, root=self.root, version_info=PY312)
                self.assertEqual(caught.exception.code, code)
        with self.assertRaises(BackendError) as caught:
            plan_install("mlx-audio", root=self.root, version_info=(3, 13, 0))
        self.assertEqual(caught.exception.code, "python_mismatch")

    def test_start_requires_matching_hash_and_spawns_with_allowlisted_env(self):
        plan = self.plan()
        self.assertEqual(start_install(plan, receipts_dir=str(self.base))["status"], "preview")
        with self.assertRaises(BackendError) as caught:
            start_install(plan, receipts_dir=str(self.base), confirm=True, preview_hash="stale")
        self.assertEqual(caught.exception.code, "preview_stale")
        outcome = start_install(plan, receipts_dir=str(self.base), confirm=True, preview_hash=plan["preview_hash"],
                                spawn=self.spawn, env={"PATH": "/bin", "OPENAI_API_KEY": "x"})
        self.assertEqual(outcome["status"], "started")
        argv, log_path, env = self.spawned[0]
        self.assertEqual(argv, plan["argv"])
        self.assertEqual(env, {"PATH": "/bin"})
        self.assertEqual(outcome["receipt"]["pid"], 4242)

    def test_existing_target_is_refused_with_remove_remediation(self):
        plan = self.plan()
        Path(plan["target"]).mkdir(parents=True)
        with self.assertRaises(BackendError) as caught:
            start_install(plan, receipts_dir=str(self.base), confirm=True, preview_hash=plan["preview_hash"], spawn=self.spawn)
        self.assertEqual(caught.exception.code, "target_exists")
        self.assertIn("backend remove mlx-audio", caught.exception.remediation)
        self.assertEqual(self.spawned, [])

    def test_states_installing_installed_and_failed(self):
        plan = self.plan()
        start_install(plan, receipts_dir=str(self.base), confirm=True, preview_hash=plan["preview_hash"], spawn=self.spawn)
        live = {entry["id"]: entry for entry in list_backends(root=self.root, receipts_dir=str(self.base), pid_alive=lambda pid: True, find_spec=lambda name: None)["backends"]}
        self.assertEqual(live["mlx-audio"]["state"], "installing")
        self.assertEqual(live["mlx-lm"]["state"], "absent")
        target = Path(plan["target"])
        (target / "bin").mkdir(parents=True)
        (target / "bin" / "python").write_text("", encoding="utf-8")
        (target / INSTALL_MARKER).write_text(json.dumps({"id": "mlx-audio", "version": "0.5.7"}), encoding="utf-8")
        done = {entry["id"]: entry for entry in list_backends(root=self.root, receipts_dir=str(self.base), pid_alive=lambda pid: False, find_spec=lambda name: None)["backends"]}
        self.assertEqual(done["mlx-audio"]["state"], "installed")
        Draft202012Validator(SCHEMA).validate(list_backends(root=self.root, receipts_dir=str(self.base), find_spec=lambda name: None))

    def test_dead_runner_without_marker_is_failed(self):
        plan = self.plan()
        start_install(plan, receipts_dir=str(self.base), confirm=True, preview_hash=plan["preview_hash"], spawn=self.spawn)
        entries = {entry["id"]: entry for entry in list_backends(root=self.root, receipts_dir=str(self.base), pid_alive=lambda pid: False, find_spec=lambda name: None)["backends"]}
        self.assertEqual(entries["mlx-audio"]["state"], "failed")
        self.assertIsNotNone(entries["mlx-audio"]["completed_at"])

    def test_remove_moves_to_trash(self):
        manifest = load_manifests()["mlx-audio"]
        target = backend_target(manifest, self.root)
        target.mkdir(parents=True)
        plan = plan_remove("mlx-audio", root=self.root, trash_dir=self.trash)
        self.assertEqual(remove_backend(plan)["status"], "preview")
        outcome = remove_backend(plan, receipts_dir=str(self.base), confirm=True, preview_hash=plan["preview_hash"])
        self.assertFalse(target.exists())
        self.assertTrue(Path(outcome["moved_to"]).exists())
        self.assertTrue(Path(outcome["moved_to"]).is_relative_to(self.trash))
        with self.assertRaises(BackendError) as caught:
            remove_backend(plan, receipts_dir=str(self.base), confirm=True, preview_hash=plan["preview_hash"])
        self.assertEqual(caught.exception.code, "not_installed")

    def test_symlinked_target_is_refused(self):
        plan = self.plan()
        real = self.base / "elsewhere"
        real.mkdir()
        self.root.mkdir(parents=True)
        Path(plan["target"]).symlink_to(real)
        with self.assertRaises(BackendError) as caught:
            start_install(plan, receipts_dir=str(self.base), confirm=True, preview_hash=plan["preview_hash"], spawn=self.spawn)
        self.assertEqual(caught.exception.code, "symlink_refused")

    def test_golden_backend_list(self):
        expected = json.loads((ROOT / "tests" / "fixtures" / "backend-list.json").read_text(encoding="utf-8"))
        self.assertEqual(golden_backend_list(), expected)
        Draft202012Validator(SCHEMA).validate(expected)

    def test_every_optional_manifest_ships_its_lock(self):
        for manifest in load_manifests().values():
            if not manifest["builtin"]:
                self.assertTrue((ROOT / "src" / "mlx_agent" / "resources" / "backends" / manifest["lock"]).is_file(), manifest["id"])


class BackendRunnerTests(unittest.TestCase):
    def test_runner_refuses_existing_target(self):
        from mlx_agent import backend_runner

        with TemporaryDirectory() as directory:
            target = Path(directory, "mlx-audio")
            target.mkdir()
            code = backend_runner.main([
                "--id", "mlx-audio", "--version", "0.5.7", "--package", "mlx_audio",
                "--lock", "/nonexistent.lock", "--target", str(target), "--python", sys.executable,
                "--trash", str(Path(directory, "Trash")), "--preview-hash", "h",
            ])
        self.assertEqual(code, 3)


if __name__ == "__main__":
    unittest.main()

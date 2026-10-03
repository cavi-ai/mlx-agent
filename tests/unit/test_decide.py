import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from mlx_agent.backends import INSTALL_MARKER, load_manifests, with_ports
from mlx_agent.decide import DECIDE_RUNNER, DecideError, plan_decide, run_decide

from .backend_fixtures import synthetic_manifests, synthetic_registries

REQUEST = {
    "state": {"document": "I was charged twice. Please refund the duplicate."},
    "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "technical"]}},
}


class Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class DecideTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifests = dict(synthetic_manifests(), **{"mlx-embeddings": load_manifests()["mlx-embeddings"]})
        self.registries = self.with_ports(synthetic_registries(self.manifests, installed=("mlx-embeddings",)))
        venv = self.root / "backends" / "mlx-embeddings"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        (venv / INSTALL_MARKER).write_text(json.dumps({"id": "mlx-embeddings", "version": self.manifests["mlx-embeddings"]["version"]}), encoding="utf-8")
        self.request = self.write_request(REQUEST)

    def with_ports(self, registries):
        for backend_id, entry in registries.items():
            entry["registry"] = with_ports(entry["registry"], self.manifests[backend_id])
        return registries

    def write_request(self, value, name="request.json"):
        path = self.root / name
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
        return path

    def model(self, model_type):
        directory = self.root / "model-{0}".format(len(list(self.root.glob("model-*"))))
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
        return directory

    def plan(self, path, request=None):
        return plan_decide(path, request or self.request, manifests=self.manifests,
                           registries=self.registries, root=self.root / "backends")

    def test_plan_runs_the_runner_with_the_port_module_in_the_backend_python(self):
        plan = self.plan(self.model("laya"))
        self.assertEqual(plan["backend"], "mlx-embeddings")
        self.assertEqual(plan["argv"][:2], [str(self.root / "backends" / "mlx-embeddings" / "bin" / "python"), str(DECIDE_RUNNER)])
        self.assertEqual(plan["argv"][2:4], ["--module", "mlx_embeddings.classifiers.laya"])
        self.assertEqual(plan["argv"][-2:], ["--request", str(self.request)])

    def test_plan_refusals(self):
        cases = [
            (lambda: self.plan(self.model("whisper")), "not_classification_model"),
            (lambda: self.plan(self.root / "missing"), "model_unreadable"),
            (lambda: self.plan(self.model("laya"), self.root / "none.json"), "request_not_found"),
        ]
        for call, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(DecideError) as caught:
                    call()
                self.assertEqual(caught.exception.code, code)
        model = self.model("Laya")
        invalid = [
            "not json", [], {"questions": REQUEST["questions"]}, {"state": 3, "questions": REQUEST["questions"]},
            {"state": "x", "questions": {}}, {"state": "x", "questions": {"q": {"type": "pick", "instructions": "?"}}},
            {"state": "x", "questions": {"q": {"type": "noul"}}},
        ]
        for index, value in enumerate(invalid):
            with self.subTest(request=value):
                with self.assertRaises(DecideError) as caught:
                    self.plan(model, self.write_request(value, "bad-{0}.json".format(index)))
                self.assertEqual(caught.exception.code, "invalid_request")
        registries = self.with_ports(synthetic_registries(self.manifests))
        with self.assertRaises(DecideError) as caught:
            plan_decide(model, self.request, manifests=self.manifests, registries=registries, root=self.root / "elsewhere")
        self.assertEqual(caught.exception.code, "backend_not_installed")

    def test_run_reads_the_last_json_line(self):
        plan = self.plan(self.model("laya"))
        seen = []
        answer = {"team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "technical": 0.1}}}

        def runner(argv, **kwargs):
            seen.append((argv, kwargs))
            return Completed(stdout="loading\n" + json.dumps({"answers": answer, "usage": {"input_tokens": 30}, "seconds": 0.05}) + "\n")

        with mock.patch("mlx_agent.decide.sync_ports", return_value=[]) as sync:
            result = run_decide(plan, manifests=self.manifests, root=self.root / "backends", runner=runner)
        self.assertEqual(sync.call_args.args[0]["id"], "mlx-embeddings")
        self.assertEqual((result["schema"], result["answers"], result["seconds"]), ("decide/1", answer, 0.05))
        self.assertEqual(seen[0][0], plan["argv"])
        self.assertEqual(seen[0][1]["stdin"], subprocess.DEVNULL)
        self.assertNotIn("HF_TOKEN", seen[0][1]["env"])

    def test_run_failures_are_classified(self):
        patcher = mock.patch("mlx_agent.decide.sync_ports", return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)
        plan = self.plan(self.model("laya"))
        failing = lambda argv, **kwargs: Completed(returncode=1, stderr="Traceback\nValueError: options do not fit\n")
        with self.assertRaises(DecideError) as caught:
            run_decide(plan, manifests=self.manifests, root=self.root / "backends", runner=failing)
        self.assertEqual(caught.exception.code, "decide_failed")
        self.assertIn("options do not fit", str(caught.exception))

        def slow(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

        with self.assertRaises(DecideError) as caught:
            run_decide(plan, manifests=self.manifests, root=self.root / "backends", runner=slow, timeout=5)
        self.assertEqual(caught.exception.code, "decide_timeout")


if __name__ == "__main__":
    unittest.main()

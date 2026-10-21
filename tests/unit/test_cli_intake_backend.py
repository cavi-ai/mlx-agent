import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from mlx_agent import cli
from mlx_agent.backends import BackendError
from mlx_agent.intake_fetch import FetchError
from mlx_agent.intake_source import IntakeSourceError
from mlx_agent.port_analysis import PortAnalysisError


RESOLVED = {
    "source": {"repo": "org/model"}, "verdict": "convertible", "reasons": ["text model"],
    "backend": "mlx-lm", "backend_installed": False,
    "task": {"type": "text_llm", "use_cases": ["chat"], "source": "config"},
    "components": [{"role": "text", "model_type": "llama", "matches": [{"backend": "mlx-lm", "module": "llama", "match": "exact"}]},
                   {"role": "vision", "model_type": "odd", "matches": []}],
    "mlx_conversions": [{"repo": "mlx-community/model-4bit", "downloads": 12}],
    "warnings": ["gated"],
}
PLAN = {"repo": "org/model", "file": None, "preview_hash": "f" * 64}


def run(argv):
    output = io.StringIO()
    with redirect_stdout(output):
        code = cli.main(argv)
    return code, output.getvalue()


class IntakeCliTests(unittest.TestCase):
    def test_resolve_prints_a_summary_or_json(self):
        with patch("mlx_agent.intake.resolve", return_value=RESOLVED) as resolve:
            code, text = run(["intake", "resolve", "org/model", "--revision", "main"])
            self.assertEqual(0, code)
            resolve.assert_called_with("org/model", revision="main")
            self.assertEqual("\n".join((
                "org/model: convertible (text model)", "  backend: mlx-lm (not installed)",
                "  task: text_llm [chat] via config", "  text: llama -> mlx-lm llama (exact)",
                "  vision: odd -> no MLX implementation", "  MLX conversions on the Hub:",
                "    mlx-community/model-4bit (12 downloads)", "  warning: gated")), text.strip())
            code, text = run(["intake", "resolve", "org/model", "--json"])
        self.assertEqual("convertible", json.loads(text)["data"]["verdict"])

    def test_fetch_requires_confirmation_then_starts(self):
        with patch("mlx_agent.intake_fetch.plan_fetch", return_value=PLAN), \
                patch("mlx_agent.intake_fetch.start_fetch", side_effect=[{"status": "preview"}, {"status": "preview"},
                                                                          {"status": "started", "pid": 7}]):
            code, text = run(["intake", "fetch", "org/model"])
            self.assertEqual(2, code)
            self.assertIn("Fetch plan: org/model (snapshot); preview_hash: " + "f" * 64, text)
            code, text = run(["intake", "fetch", "org/model", "--json"])
            self.assertEqual((2, True), (code, json.loads(text)["data"]["requires_confirmation"]))
            code, text = run(["intake", "fetch", "org/model", "--confirm", "--preview-hash", "f" * 64, "--json"])
        self.assertEqual((0, "started"), (code, json.loads(text)["data"]["status"]))

    def test_status_analysis_and_port_plan_report_their_results(self):
        with patch("mlx_agent.intake_fetch.status_fetch", return_value=[]):
            self.assertEqual({"jobs": []}, json.loads(run(["intake", "status", "--json"])[1])["data"])
        with patch("mlx_agent.port_analysis.analyze", return_value={"model_type": "odd"}):
            self.assertEqual({"model_type": "odd"}, json.loads(run(["intake", "port-analysis", "org/model"])[1]))
        with patch("mlx_agent.port_plan.draft_port_plan", return_value={"path": "/tmp/plan.md"}) as draft:
            code, text = run(["intake", "port-plan", "org/model", "--endpoint", "http://127.0.0.1:1234", "--model", "m"])
        self.assertEqual((0, "port plan draft written: /tmp/plan.md"), (code, text.strip()))
        self.assertEqual({"revision": None, "context_tokens": 32768, "max_tokens": 4096, "out_dir": None}, draft.call_args.kwargs)
        code, text = run(["intake", "port-plan", "org/model", "--endpoint", "http://127.0.0.1:1234", "--model", "m",
                          "--context-tokens", "10", "--json"])
        self.assertEqual((2, "intake_failed"), (code, json.loads(text)["error"]["code"]))

    def test_failures_carry_their_code_and_remediation(self):
        cases = (
            ("intake.resolve", IntakeSourceError("not a model"), "invalid_source"),
            ("port_analysis.analyze", PortAnalysisError("port_unavailable", "no config", "Pick a model."), "port_unavailable"),
            ("intake_fetch.plan_fetch", FetchError("fetch_refused", "refused", "Review."), "fetch_refused"),
            ("intake_fetch.status_fetch", OSError("disk"), "intake_failed"),
        )
        commands = {"intake.resolve": ["resolve", "x"], "port_analysis.analyze": ["port-analysis", "x"],
                    "intake_fetch.plan_fetch": ["fetch", "x"], "intake_fetch.status_fetch": ["status"]}
        for name, error, code in cases:
            with self.subTest(name=name), patch("mlx_agent." + name, side_effect=error):
                status, text = run(["intake", *commands[name], "--json"])
                self.assertEqual((2, code), (status, json.loads(text)["error"]["code"]))
                status, text = run(["intake", *commands[name]])
                self.assertEqual(2, status)
                self.assertTrue(text.startswith("intake failed [{0}]".format(code)))


class BackendCliTests(unittest.TestCase):
    def test_list_prints_each_backend(self):
        report = {"root": "/backends", "backends": [{"id": "mlx-lm", "version": "1.0", "state": "installed",
                                                       "model_types": 12, "registry_source": "snapshot"}]}
        with patch("mlx_agent.backend_install.list_backends", return_value=report):
            code, text = run(["backend", "list"])
            self.assertEqual((0, "Backends under /backends:\n  mlx-lm 1.0: installed (12 model types, snapshot)"), (code, text.strip()))
            self.assertEqual(report, json.loads(run(["backend", "list", "--json"])[1])["data"])

    def test_install_and_remove_preview_before_acting(self):
        plan = {"id": "mlx-audio", "preview_hash": "e" * 64}
        with patch("mlx_agent.backend_install.plan_install", return_value=plan), \
                patch("mlx_agent.backend_install.start_install", side_effect=[{"status": "preview"}, {"status": "installing"}]):
            code, text = run(["backend", "install", "mlx-audio"])
            self.assertEqual(2, code)
            self.assertIn("Backend install plan for mlx-audio; preview_hash: " + "e" * 64, text)
            code, text = run(["backend", "install", "mlx-audio", "--confirm", "--preview-hash", "e" * 64, "--json"])
            self.assertEqual((0, "installing"), (code, json.loads(text)["data"]["status"]))
        with patch("mlx_agent.backend_install.plan_remove", return_value=plan), \
                patch("mlx_agent.backend_install.remove_backend", return_value={"status": "preview"}):
            code, text = run(["backend", "remove", "mlx-audio", "--json"])
        self.assertEqual((2, True), (code, json.loads(text)["data"]["requires_confirmation"]))

    def test_failures_carry_their_code(self):
        for error, code in ((BackendError("backend_unknown", "no such backend", "List backends."), "backend_unknown"),
                            (ValueError("bad"), "backend_failed")):
            with self.subTest(code=code), patch("mlx_agent.backend_install.plan_install", side_effect=error):
                status, text = run(["backend", "install", "nope", "--json"])
                self.assertEqual((2, code), (status, json.loads(text)["error"]["code"]))
                status, text = run(["backend", "install", "nope"])
                self.assertTrue(text.startswith("backend failed [{0}]".format(code)))


if __name__ == "__main__":
    unittest.main()

import json
import os
import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock

from mlx_agent import decide, describe, generate, music, speak, transcribe, video
from mlx_agent.backends import BackendError
from mlx_agent.inference_runner import InferenceError, invoke

CONTRACTS = {module.CONTRACT.verb: module.CONTRACT for module in (decide, describe, generate, music, speak, transcribe, video)}
PLAN = {"backend": "fixture", "argv": ["/backend/python", "runner.py"], "input": {"kind": "image"},
        "model_type": "fixture", "out": "/missing/output.wav"}
MANIFESTS = {"fixture": {"id": "fixture"}}


def completed(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class InferenceRunnerTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("mlx_agent.inference_runner.sync_ports", return_value=[])
        self.sync = patcher.start()
        self.addCleanup(patcher.stop)

    def test_every_verb_error_is_an_inference_error(self):
        self.assertEqual(sorted(CONTRACTS), ["decide", "describe", "generate", "music", "speak", "transcribe", "video"])
        for verb, contract in CONTRACTS.items():
            with self.subTest(verb=verb):
                self.assertTrue(issubclass(contract.error, InferenceError))

    def test_every_verb_redacts_and_trims_the_failure_tail(self):
        stderr = "\n".join(["noise"] * 10 + ["ValueError: weights at https://user:pw@hub.example/x",
                                             "auth token=sk-live-0123456789"])
        for verb, contract in CONTRACTS.items():
            with self.subTest(verb=verb), self.assertRaises(contract.error) as caught:
                invoke(PLAN, contract, MANIFESTS, runner=lambda argv, **kw: completed(1, stderr=stderr))
            message = str(caught.exception)
            self.assertEqual(caught.exception.code, verb + "_failed")
            self.assertTrue(message.startswith("The fixture backend could not "), message)
            self.assertNotIn("sk-live-0123456789", message)
            self.assertNotIn("user:pw", message)
            self.assertIn("token=<redacted>", message)
            self.assertEqual(message.count("noise"), 4)

    def test_every_verb_reports_a_timeout_with_its_own_code(self):
        def slow(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

        for verb, contract in CONTRACTS.items():
            with self.subTest(verb=verb), self.assertRaises(contract.error) as caught:
                invoke(PLAN, contract, MANIFESTS, timeout=7, runner=slow)
            self.assertEqual(caught.exception.code, verb + "_timeout")
            self.assertIn("exceeded 7 s", str(caught.exception))

    def test_every_verb_syncs_ports_first_and_maps_a_sync_failure(self):
        self.sync.side_effect = BackendError("port_conflict", "fixture already ships a port", "Remove it.")
        runner = mock.Mock()
        for verb, contract in CONTRACTS.items():
            with self.subTest(verb=verb), self.assertRaises(contract.error) as caught:
                invoke(PLAN, contract, MANIFESTS, runner=runner)
            self.assertEqual((caught.exception.code, caught.exception.remediation), ("port_conflict", "Remove it."))
        self.assertEqual(self.sync.call_count, len(CONTRACTS))
        runner.assert_not_called()

    def test_an_accepted_result_counts_only_on_a_clean_exit(self):
        contract = transcribe.CONTRACT
        stdout = "\n".join(["loading", json.dumps({"text": "first"}), json.dumps([1]), json.dumps({"text": "last"}), "done"])
        result = invoke(PLAN, contract, MANIFESTS, runner=lambda argv, **kw: completed(0, stdout=stdout))
        self.assertEqual(result, {"text": "last"})
        with self.assertRaises(transcribe.TranscribeError) as caught:
            invoke(PLAN, contract, MANIFESTS, runner=lambda argv, **kw: completed(1, stdout=stdout, stderr=""))
        self.assertEqual(caught.exception.code, "transcribe_failed")
        self.assertIn("no output", str(caught.exception))

    def test_a_refusal_wins_over_the_exit_status_and_its_detail_is_redacted(self):
        refusal = json.dumps({"error": "video_unsupported", "detail": "decoder failed with token=sk-live-1"})
        with self.assertRaises(describe.DescribeError) as caught:
            invoke(dict(PLAN, input={"kind": "video"}), describe.CONTRACT, MANIFESTS,
                   runner=lambda argv, **kw: completed(3, stdout=refusal + "\n"))
        self.assertEqual(caught.exception.code, "video_unsupported")
        self.assertIn("token=<redacted>", str(caught.exception))
        self.assertNotIn("sk-live-1", str(caught.exception))

    def test_runner_environment_is_the_backend_allowlist_plus_the_contract(self):
        seen = []
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_secret", "PATH": "/usr/bin"}):
            for contract in (music.CONTRACT, speak.CONTRACT):
                with self.assertRaises(InferenceError):
                    invoke(PLAN, contract, MANIFESTS, runner=lambda argv, **kw: seen.append(kw) or completed(1))
        self.assertEqual([kw["stdin"] for kw in seen], [subprocess.DEVNULL] * 2)
        self.assertNotIn("HF_TOKEN", seen[0]["env"])
        self.assertEqual((seen[0]["env"]["HF_HUB_OFFLINE"], seen[0]["env"]["TRANSFORMERS_OFFLINE"]), ("1", "1"))
        self.assertNotIn("HF_HUB_OFFLINE", seen[1]["env"])


if __name__ == "__main__":
    unittest.main()

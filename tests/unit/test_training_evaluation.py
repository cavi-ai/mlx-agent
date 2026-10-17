import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from mlx_agent import training_evaluation as evaluation


def suite():
    return {"id": "held-out", "name": "Evaluation questions", "prompts": [
        {"id": "first", "text": "Explain previews.", "max_tokens": 32},
        {"id": "second", "text": "Explain rollback.", "max_tokens": 64}]}


class EvaluationSuiteTests(unittest.TestCase):
    def test_suite_preserves_order_text_and_limits_and_rejects_ambiguous_inputs(self):
        original = suite()
        normalized = evaluation.validate_suite(original)
        self.assertEqual(normalized, original)
        normalized["prompts"][0]["text"] = "Edited"
        self.assertEqual(original["prompts"][0]["text"], "Explain previews.")
        for mutate in [lambda x: x["prompts"].clear(),
                       lambda x: x["prompts"].append(x["prompts"][0]),
                       lambda x: x["prompts"][0].update(max_tokens=513),
                       lambda x: x["prompts"][0].update(tool={}),
                       lambda x: x.update(extra=True)]:
            broken = copy.deepcopy(original); mutate(broken)
            with self.assertRaises(ValueError): evaluation.validate_suite(broken)
        with self.assertRaises(ValueError):
            evaluation.preview("/unused", "Ambiguous", prompt_suite=original)

    def test_runtime_loads_one_variant_once_and_resets_seed_for_each_prompt(self):
        from mlx_agent.training_eval_runtime import generate_outputs
        calls, seeds, loads = [], [], []
        plan = {"base_path": "/base", "adapter_path": "/adapter", "prompt_suite": suite(), "seed": 42, "temperature": 0.0}
        def load(path, **kwargs):
            loads.append((path, kwargs)); return "model", "tokenizer"
        def generate(model, tokenizer, **kwargs):
            calls.append(kwargs); return kwargs["prompt"] + " answer"
        result = generate_outputs(plan, "adapter", load, generate, lambda **kw: kw, seeds.append)
        self.assertEqual(len(loads), 1)
        self.assertEqual(loads[0][1], {"adapter_path": "/adapter"})
        self.assertEqual(seeds, [42, 42])
        self.assertEqual([x["max_tokens"] for x in calls], [32, 64])
        self.assertEqual([x["id"] for x in result["samples"]], ["first", "second"])


@unittest.skipUnless(sys.version_info >= (3, 12) and importlib.util.find_spec("pydantic") is not None,
                     "Optional Python 3.12 lab runtime is not installed.")
class EvaluationPersistenceTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(evaluation.__file__).parent / "_lab"))
        self.addCleanup(lambda: sys.path.remove(str(Path(evaluation.__file__).parent / "_lab")))
        from video_lab.database import Database
        from video_lab.artifacts import ArtifactStore
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.database = Database(self.root / "experiments.sqlite3"); self.database.migrate()
        self.artifacts = ArtifactStore(self.database, self.root / "artifacts")
        self.environment = patch("mlx_agent.compute.compute_root", return_value=self.root / "compute")
        self.environment.start(); self.addCleanup(self.environment.stop)
        self.base = self.root / "base"; self.base.mkdir()
        (self.base / "config.json").write_text('{"model_type":"llama"}')
        (self.base / "model.safetensors").write_bytes(b"base fixture")
        adapter = self.root / "adapter"; adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        (adapter / "adapters.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"1234")
        from mlx_agent.lora import adapter_identity
        from mlx_agent.jit_serve import model_fingerprint
        self.receipt = self.root / "receipt.json"
        self.receipt.write_text(json.dumps({"kind": "lora", "pid": 9, "argv": [], "repo": "org/base", "run_id": "training-run",
            "out": str(adapter), "exit_status": "done", "adapter_identity": adapter_identity(adapter),
            "base_identity": {"path": str(self.base), "fingerprint": model_fingerprint(self.base)}}))

    def run_suite(self):
        prompts = suite()
        plan = evaluation.preview(str(self.receipt), prompt_suite=prompts)
        calls = []
        def generate(plan, variant):
            calls.append(variant)
            return [{"id": item["id"], "text": variant + " " + item["text"]} for item in plan["prompt_suite"]["prompts"]]
        result = evaluation.run(self.database, self.artifacts, {"receipt": str(self.receipt),
            "prompt_suite": prompts, "preview_hash": plan["preview_hash"]}, generate=generate)
        self.assertEqual(calls, ["baseline", "adapter"])
        return result

    def test_suite_retains_all_pairs_and_reviews_without_rewriting_evidence(self):
        result = self.run_suite()
        self.assertEqual(len(result["report"]["samples"]), 2)
        self.assertIsNone(result["report"]["judgment"])
        request = {"evaluation_id": result["evaluation_id"], "artifact_sha256": result["artifact_sha256"],
                   "prompt_id": "second", "preference": "adapter", "notes": "More specific."}
        reviewed = evaluation.review(self.database, self.artifacts, "reviewer", request)
        self.assertEqual(reviewed["reviews"][0]["prompt_id"], "second")
        request.update(preference="tie", notes="Both are useful.")
        evaluation.review(self.database, self.artifacts, "reviewer", request)
        inspected = evaluation.inspect(self.database, self.artifacts, result["evaluation_id"])
        self.assertEqual(inspected["artifact_sha256"], result["artifact_sha256"])
        self.assertEqual(inspected["report"], result["report"])
        self.assertEqual(inspected["reviews"][0]["preference"], "tie")
        with self.database.connect() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM text_training_evaluation_reviews").fetchone()[0], 2)

    def test_changed_prompts_partial_outputs_and_wrong_review_hash_are_rejected(self):
        result = self.run_suite()
        request = {"evaluation_id": result["evaluation_id"], "artifact_sha256": "0" * 64,
                   "prompt_id": "first", "preference": "baseline"}
        with self.assertRaises(ValueError): evaluation.review(self.database, self.artifacts, "reviewer", request)
        request.update(artifact_sha256=result["artifact_sha256"], prompt_id="unknown")
        with self.assertRaises(ValueError): evaluation.review(self.database, self.artifacts, "reviewer", request)
        prompts = suite(); plan = evaluation.preview(str(self.receipt), prompt_suite=prompts)
        prompts["prompts"][0]["text"] = "Changed"
        with self.assertRaises(ValueError):
            evaluation.run(self.database, self.artifacts, {"receipt": str(self.receipt), "prompt_suite": prompts,
                "preview_hash": plan["preview_hash"]}, generate=lambda *_: self.fail("Stale plan generated"))
        prompts = suite()
        with self.assertRaises(ValueError):
            evaluation.run(self.database, self.artifacts, {"receipt": str(self.receipt), "prompt_suite": prompts,
                "preview_hash": plan["preview_hash"]}, generate=lambda *_: [{"id": "first", "text": "Only one"}])
        self.assertEqual(len(evaluation.history(self.database)), 1)

    def test_legacy_single_prompt_reports_remain_inspectable_and_reviewable(self):
        plan = evaluation.preview(str(self.receipt), "Legacy prompt")
        result = evaluation.run(self.database, self.artifacts, {"receipt": str(self.receipt), "prompt": "Legacy prompt",
            "preview_hash": plan["preview_hash"]}, generate=lambda _, variant: variant + " answer")
        self.assertEqual(result["report"]["schema_version"], "text-training-evaluation/1")
        evaluation.review(self.database, self.artifacts, "reviewer", {"evaluation_id": result["evaluation_id"],
            "artifact_sha256": result["artifact_sha256"], "prompt_id": "prompt", "preference": "tie"})
        self.assertEqual(evaluation.inspect(self.database, self.artifacts, result["evaluation_id"])["reviews"][0]["preference"], "tie")

    def test_workspace_accepts_suite_requests_but_refuses_mixed_inputs(self):
        from mlx_agent.lab_workspace import validate_request, execute
        request = {"operation": "training-preview", "receipt": str(self.receipt), "prompt_suite": suite()}
        result = execute(self.database.path, self.artifacts.root, "reviewer", request)
        self.assertEqual(result["plan"]["prompt_suite"], suite())
        for extra in ({"prompt": "Ambiguous"}, {"max_tokens": 128}):
            with self.assertRaises(ValueError): validate_request(dict(request, **extra))

    def test_changed_retained_artifact_cannot_receive_a_review(self):
        from video_lab.artifacts import ArtifactError
        result = self.run_suite()
        path = Path(result["path"])
        path.chmod(0o600); path.write_text("changed artifact")
        with self.assertRaises((ValueError, ArtifactError)):
            evaluation.review(self.database, self.artifacts, "reviewer", {"evaluation_id": result["evaluation_id"],
                "artifact_sha256": result["artifact_sha256"], "prompt_id": "first", "preference": "tie"})

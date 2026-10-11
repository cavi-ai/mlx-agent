import json
import importlib.util
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

@unittest.skipUnless(sys.version_info >= (3, 12) and importlib.util.find_spec("typer") is not None
    and importlib.util.find_spec("pydantic") is not None, "Optional Python 3.12 lab runtime is not installed.")
class LabWorkspaceTests(unittest.TestCase):
    def invoke(self, directory, request, reviewer="reviewer"):
        directory = str(Path(directory).resolve())
        request_path = Path(directory) / "request.json"
        request_path.write_text(json.dumps(request))
        result = subprocess.run([sys.executable, str(ROOT / "scripts/mlx-agent"), "lab",
            "--database", str(Path(directory) / "experiments.sqlite3"),
            "--artifacts", str(Path(directory) / "artifacts"), "--reviewer-id", reviewer,
            "workspace", "--request", str(request_path), "--json"],
            capture_output=True, text=True, timeout=60)
        return result, json.loads(result.stdout)

    def test_native_adaptive_progression_and_reviewer_binding(self):
        manifest = dict(model_id="video-lab/fake", model_revision="r1", model_checksum="sha256:fake",
            adapter_id="video-lab.fake.v1", adapter_version="1", runtime_versions={"fake":"1"},
            prompt="test fixture", seed=42, width=64, height=48, num_frames=17, fps=8,
            inference_steps=4, guidance_scale=1.0, scheduler="fake", dtype="uint8",
            quantization="none", backend="cpu", device="cpu", memory_strategy="test",
            hardware_fingerprint="fake", os_fingerprint="fake")
        with tempfile.TemporaryDirectory() as directory:
            result, payload = self.invoke(directory, {"operation":"study-create","manifest":manifest})
            self.assertEqual(result.returncode,0,payload)
            identity = payload["data"]["study"]["study_id"]
            result, payload = self.invoke(directory, {"operation":"study-show","study_id":identity}, reviewer="other")
            self.assertNotEqual(result.returncode,0,payload)
            result, payload = self.invoke(directory, {"operation":"study-resume","study_id":identity})
            self.assertEqual(result.returncode,0,payload)
            self.assertEqual(payload["data"]["study"]["next_action"], "complete_generation")
            result, payload = self.invoke(directory, {"operation":"study-run","study_id":identity})
            self.assertEqual(result.returncode,0,payload)
            session = payload["data"]["study"]["review_id"]
            self.assertEqual(payload["data"]["study"]["next_action"], "record_preference")
            result, media = self.invoke(directory, {"operation":"review-media","session_id":session})
            self.assertEqual(result.returncode,0,media)
            self.assertTrue(Path(media["data"]["left"]).is_file())
            result, payload = self.invoke(directory, {"operation":"study-submit","study_id":identity,
                                                     "session_id":session,"preference":"tie"})
            self.assertEqual(result.returncode,0,payload)
            self.assertEqual(payload["data"]["study"]["comparison_count"],1)

    def test_empty_native_snapshot_is_sealed_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            result, payload = self.invoke(directory, {"operation": "snapshot"})
            self.assertEqual(result.returncode, 0, payload)
            self.assertEqual(payload["data"]["history"]["items"], [])
            self.assertEqual(payload["data"]["queue"], [])

    def test_unknown_request_is_rejected_before_state_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            result, payload = self.invoke(directory, {"operation": "erase", "unexpected": True})
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(payload["status"], "error")
            self.assertFalse((Path(directory) / "experiments.sqlite3").exists())

    def test_only_registered_capabilities_can_create_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            result, payload = self.invoke(directory, {"operation": "create", "draft": {}})
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(payload["status"], "error")

    def test_native_and_agent_share_generation_and_blind_review(self):
        baseline = dict(model_id="video-lab/fake", model_revision="r1",
            model_checksum="sha256:fake", adapter_id="video-lab.fake.v1",
            adapter_version="1", runtime_versions={"fake": "1"}, prompt="a deterministic test",
            seed=42, width=64, height=48, num_frames=17, fps=8, inference_steps=4,
            guidance_scale=1.0, scheduler="fake", dtype="uint8", quantization="none",
            backend="cpu", device="cpu", memory_strategy="test",
            hardware_fingerprint="fake", os_fingerprint="fake")
        draft = dict(baseline=baseline, variable_path="inference_steps", alternate_value=8)
        with tempfile.TemporaryDirectory() as directory:
            result, preview = self.invoke(directory, {"operation": "preview", "draft": draft})
            self.assertEqual(result.returncode, 0, preview)
            self.assertEqual(preview["data"]["changed_paths"], ["inference_steps"])
            result, created = self.invoke(directory, {"operation": "create", "draft": draft})
            self.assertEqual(result.returncode, 0, created)
            identifier = created["data"]["experiment_id"]
            result, snapshot = self.invoke(directory, {"operation": "snapshot"})
            self.assertEqual(result.returncode, 0, snapshot)
            self.assertTrue(snapshot["data"]["history"]["items"][0]["sealed"])
            for job in snapshot["data"]["queue"]:
                result, output = self.invoke(directory, {"operation": "run", "job_id": job["id"]})
                self.assertEqual(result.returncode, 0, output)
                self.assertEqual(output["data"]["status"], "completed", output)
            result = subprocess.run([sys.executable, str(ROOT / "scripts/mlx-agent"), "lab",
                "--database", str(Path(directory) / "experiments.sqlite3"),
                "--artifacts", str(Path(directory) / "artifacts"),
                "experiment", "list", "--json"], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn(identifier, result.stdout)
            _, session = self.invoke(directory, {"operation": "review-start", "experiment_id": identifier})
            session_id = session["data"]["session_id"]
            result, reveal = self.invoke(directory, {"operation": "reveal", "session_id": session_id})
            self.assertNotEqual(result.returncode, 0, reveal)
            _, media = self.invoke(directory, {"operation": "review-media", "session_id": session_id})
            for position in ("left", "right"):
                video = Path(media["data"][position])
                self.assertTrue(video.is_file(), media)
                self.assertEqual(video.name, position + ".mp4")
                proof = subprocess.run([shutil.which("ffprobe") or "ffprobe", "-v", "error", "-show_entries",
                    "stream=codec_name,width,height", "-of", "json", str(video)],
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(proof.returncode, 0, proof.stderr)
                self.assertEqual(json.loads(proof.stdout)["streams"][0]["width"], 64)
            result, repeated = self.invoke(directory, {"operation": "review-media", "session_id": session_id})
            self.assertEqual(result.returncode, 0, repeated)
            self.assertEqual(repeated["data"], media["data"])
            result, incomplete = self.invoke(directory, {"operation": "review-submit",
                "session_id": session_id, "preference": "tie", "scores": []})
            self.assertNotEqual(result.returncode, 0, incomplete)
            dimensions = ["prompt_adherence", "visual_quality", "motion_quality",
                          "temporal_consistency", "character_consistency"]
            scores = [{"display_position": p, "dimension": d, "score": 3}
                      for p in ("left", "right") for d in dimensions]
            result, review = self.invoke(directory, {"operation": "review-submit",
                "session_id": session_id, "preference": "tie", "scores": scores})
            self.assertEqual(result.returncode, 0, review)
            result, reveal = self.invoke(directory, {"operation": "reveal", "session_id": session_id})
            self.assertEqual(result.returncode, 0, reveal)
            self.assertNotEqual(reveal["data"], media["data"])
            result, snapshot = self.invoke(directory, {"operation": "snapshot"})
            self.assertEqual(result.returncode, 0, snapshot)
            self.assertFalse(snapshot["data"]["history"]["items"][0]["sealed"])

    def test_path_identifiers_and_missing_fields_create_no_store(self):
        for request in ({"operation": "review-media", "session_id": "../outside"},
                        {"operation": "run"}, {"operation": "cancel"}):
            with tempfile.TemporaryDirectory() as directory:
                result, output = self.invoke(directory, request)
                self.assertNotEqual(result.returncode, 0, output)
                self.assertFalse((Path(directory) / "experiments.sqlite3").exists())

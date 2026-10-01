import http.client
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from jsonschema import Draft202012Validator

from mlx_agent import cli
from mlx_agent.backends import load_manifests, load_registries
from mlx_agent.huggingface import HuggingFaceHTTPError
from mlx_agent.intake import resolve

from .backend_fixtures import synthetic_manifests, synthetic_registries

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
SCHEMA = json.loads((ROOT / "schemas" / "intake.schema.json").read_text(encoding="utf-8"))


class FakeClient:
    def __init__(self, info=None, files=None, info_error=None):
        self.info = info
        self.files = files or {}
        self.info_error = info_error
        self.raw_requests = []

    def fetch_model_info(self, repo, revision="main", timeout=8):
        if self.info_error is not None:
            raise self.info_error
        return self.info

    def fetch_raw_text(self, repo, revision, filename, timeout=8):
        self.raw_requests.append(filename)
        if filename not in self.files:
            raise HuggingFaceHTTPError(404, "missing")
        return self.files[filename]


def audio8_client():
    info = json.loads((FIXTURES / "hf" / "audio8-api.json").read_text(encoding="utf-8"))
    config = (FIXTURES / "hf" / "audio8-config.json").read_text(encoding="utf-8")
    return FakeClient(info=info, files={"config.json": config})


def golden_audio8():
    manifests = load_manifests()
    registries = load_registries(manifests, root=Path("/nonexistent"), find_spec=lambda name: None)
    return resolve(
        "https://huggingface.co/Edge0/Audio8-ASR-Infinite",
        client=audio8_client(), manifests=manifests, registries=registries,
    )


def repo_info(model_type=None, tags=(), pipeline_tag=None, library_name="transformers", files=("config.json", "model.safetensors"), gated=False, sizes=True):
    siblings = []
    for name in files:
        sibling = {"rfilename": name}
        if sizes:
            sibling["size"] = 1000
        siblings.append(sibling)
    return {
        "id": "org/name", "pipeline_tag": pipeline_tag, "library_name": library_name,
        "tags": list(tags), "gated": gated, "config": {"model_type": model_type} if model_type else {},
        "siblings": siblings,
    }


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.manifests = synthetic_manifests()
        self.validator = Draft202012Validator(SCHEMA)

    def run_resolve(self, client, installed=(), text="org/name"):
        payload = resolve(text, client=client, manifests=self.manifests,
                          registries=synthetic_registries(self.manifests, installed))
        self.validator.validate(payload)
        return payload

    def test_audio8_is_unsupported_with_components_and_no_download(self):
        payload = golden_audio8()
        self.validator.validate(payload)
        self.assertEqual(payload["verdict"], "unsupported")
        self.assertEqual(payload["reasons"], ["arch_not_in_registry", "custom_code"])
        self.assertEqual(payload["task"]["type"], "speech_to_text")
        self.assertEqual(payload["task"]["use_cases"], ["realtime_transcription", "transcription"])
        roles = {c["role"]: c for c in payload["components"]}
        self.assertEqual(sorted(roles), ["audio", "model", "text"])
        self.assertEqual(roles["model"]["matches"], [])
        self.assertIn("mlx_audio.stt.models.voxtral_realtime", [m["module"] for m in roles["audio"]["matches"]])
        self.assertIn("mlx_lm.models.qwen2", [m["module"] for m in roles["text"]["matches"]])
        self.assertGreater(payload["bytes"], 8_000_000_000)

    def test_golden_fixture_matches(self):
        expected = json.loads((FIXTURES / "intake-resolve-audio8.json").read_text(encoding="utf-8"))
        self.assertEqual(golden_audio8(), expected)

    def test_text_model_is_convertible_with_installed_builtin(self):
        client = FakeClient(info=repo_info(pipeline_tag="text-generation"), files={"config.json": json.dumps({"model_type": "qwen2"})})
        payload = self.run_resolve(client, installed=("mlx-lm",))
        self.assertEqual((payload["verdict"], payload["backend"], payload["backend_installed"]), ("convertible", "mlx-lm", True))

    def test_speech_model_needs_audio_backend_install(self):
        client = FakeClient(info=repo_info(pipeline_tag="automatic-speech-recognition"), files={"config.json": json.dumps({"model_type": "whisper"})})
        payload = self.run_resolve(client, installed=("mlx-lm",))
        self.assertEqual((payload["verdict"], payload["backend"], payload["backend_installed"]), ("convertible_after_install", "mlx-audio", False))

    def test_vision_config_routes_to_vlm(self):
        client = FakeClient(info=repo_info(pipeline_tag="image-text-to-text"), files={"config.json": json.dumps({"model_type": "qwen2_vl", "vision_config": {"model_type": "qwen2_vl"}})})
        payload = self.run_resolve(client)
        self.assertEqual((payload["verdict"], payload["backend"]), ("convertible_after_install", "mlx-vlm"))

    def test_already_mlx_and_gguf_and_gated(self):
        mlx = self.run_resolve(FakeClient(info=repo_info(library_name="mlx", tags=["mlx"]), files={"config.json": "{}"}))
        self.assertEqual(mlx["verdict"], "already_mlx")
        gguf = self.run_resolve(FakeClient(info=repo_info(files=("a-Q4_K_M.gguf", "a-Q8_0.gguf"))))
        self.assertEqual(gguf["verdict"], "gguf")
        self.assertEqual([f["name"] for f in gguf["files"]["gguf"]], ["a-Q4_K_M.gguf", "a-Q8_0.gguf"])
        gated_client = FakeClient(info=repo_info(gated="manual"))
        gated = self.run_resolve(gated_client)
        self.assertEqual((gated["verdict"], gated["reasons"]), ("blocked", ["gated"]))
        self.assertEqual(gated_client.raw_requests, [])

    def test_hub_failures(self):
        missing = self.run_resolve(FakeClient(info_error=HuggingFaceHTTPError(401, "HTTP 401")))
        self.assertEqual((missing["verdict"], missing["reasons"]), ("blocked", ["not_found_or_private"]))
        offline = self.run_resolve(FakeClient(info_error=TimeoutError("deadline")))
        self.assertEqual((offline["verdict"], offline["reasons"]), ("unknown", ["hub_unreachable"]))
        broken = self.run_resolve(FakeClient(info_error=http.client.HTTPException("boom")))
        self.assertEqual(broken["verdict"], "unknown")

    def test_missing_config_is_unsupported_no_config(self):
        payload = self.run_resolve(FakeClient(info=repo_info(files=("model.safetensors",))))
        self.assertEqual((payload["verdict"], payload["reasons"]), ("unsupported", ["no_config"]))

    def test_missing_sizes_count_as_zero(self):
        client = FakeClient(info=repo_info(sizes=False), files={"config.json": json.dumps({"model_type": "qwen2"})})
        payload = self.run_resolve(client, installed=("mlx-lm",))
        self.assertEqual(payload["bytes"], 0)
        self.assertEqual(payload["verdict"], "convertible")

    def test_unreadable_config_warns(self):
        client = FakeClient(info=repo_info(), files={"config.json": "{oops"})
        payload = self.run_resolve(client)
        self.assertEqual(payload["verdict"], "unsupported")
        self.assertTrue(any("config.json" in warning for warning in payload["warnings"]))


class IntakeCliTests(unittest.TestCase):
    def test_resolve_cli_emits_envelope(self):
        with mock.patch("mlx_agent.intake.HuggingFaceClient", return_value=audio8_client()), \
             mock.patch("mlx_agent.intake.load_registries", side_effect=lambda manifests: load_registries(manifests, root=Path("/nonexistent"), find_spec=lambda name: None)):
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(["intake", "resolve", "Edge0/Audio8-ASR-Infinite", "--json"])
        envelope = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(envelope["operation"], "intake-resolve")
        self.assertEqual(envelope["data"]["verdict"], "unsupported")

    def test_invalid_source_fails_cleanly(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["intake", "resolve", "https://github.com/x/y", "--json"])
        envelope = json.loads(out.getvalue())
        self.assertEqual(code, 2)
        self.assertEqual(envelope["error"]["code"], "invalid_source")


if __name__ == "__main__":
    unittest.main()

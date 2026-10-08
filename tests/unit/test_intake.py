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
    def __init__(self, info=None, files=None, info_error=None, headers=None):
        self.info = info
        self.files = files or {}
        self.info_error = info_error
        self.headers = headers or {}
        self.raw_requests = []
        self.header_requests = []

    def fetch_model_info(self, repo, revision="main", timeout=8):
        if self.info_error is not None:
            raise self.info_error
        return self.info

    def fetch_raw_text(self, repo, revision, filename, timeout=8):
        self.raw_requests.append(filename)
        if filename not in self.files:
            raise HuggingFaceHTTPError(404, "missing")
        return self.files[filename]

    def fetch_safetensors_header(self, repo, revision, filename, timeout=8):
        self.header_requests.append(filename)
        if filename not in self.headers:
            raise HuggingFaceHTTPError(404, "missing")
        return self.headers[filename]


def audio8_client():
    info = json.loads((FIXTURES / "hf" / "audio8-api.json").read_text(encoding="utf-8"))
    config = (FIXTURES / "hf" / "audio8-config.json").read_text(encoding="utf-8")
    headers = json.loads((FIXTURES / "hf" / "audio8-safetensors-headers.json").read_text(encoding="utf-8"))
    return FakeClient(info=info, files={"config.json": config}, headers=headers)


def golden_audio8():
    manifests = load_manifests()
    registries = load_registries(manifests, root=Path("/nonexistent"), find_spec=lambda name: None)
    return resolve(
        "https://huggingface.co/Edge0/Audio8-ASR-Infinite",
        client=audio8_client(), manifests=manifests, registries=registries,
    )


def golden_laya(text="convaiinnovations/laya", config=None):
    info = json.loads((FIXTURES / "hf" / "laya-api.json").read_text(encoding="utf-8"))
    if config is not None:
        info = dict(info, config=config)
    headers = json.loads((FIXTURES / "hf" / "laya-safetensors-headers.json").read_text(encoding="utf-8"))
    client = FakeClient(info=info, headers=headers)
    manifests = load_manifests()
    registries = load_registries(manifests, root=Path("/nonexistent"), find_spec=lambda name: None)
    return client, resolve(text, client=client, manifests=manifests, registries=registries)


class ReposClient(FakeClient):
    """Model info per repository (a recipe reads its base repository's listing too)."""

    def __init__(self, infos, **kwargs):
        super().__init__(**kwargs)
        self.infos = infos
        self.info_requests = []

    def fetch_model_info(self, repo, revision="main", timeout=8):
        self.info_requests.append((repo, revision))
        if repo not in self.infos:
            raise HuggingFaceHTTPError(404, "missing")
        return self.infos[repo]


def qwen_image(repo):
    uc = json.loads((FIXTURES / "hf" / "qwen-image-uc-api.json").read_text(encoding="utf-8"))
    base = json.loads((FIXTURES / "hf" / "qwen-image-base-api.json").read_text(encoding="utf-8"))
    index = (FIXTURES / "hf" / "qwen-image-model-index.json").read_text(encoding="utf-8")
    client = ReposClient({"abenzerps/Qwen-Image-2.1-Uncensored-GGUF": uc, "Qwen/Qwen-Image-2.1": base},
                         files={"model_index.json": index})
    manifests = load_manifests()
    registries = load_registries(manifests, root=Path("/nonexistent"), find_spec=lambda name: None)
    return client, resolve(repo, client=client, manifests=manifests, registries=registries)


def wan_t2v(repo="Wan-AI/Wan2.1-T2V-1.3B"):
    info = json.loads((FIXTURES / "hf" / "wan21-t2v-api.json").read_text(encoding="utf-8"))
    config = (FIXTURES / "hf" / "wan21-t2v-config.json").read_text(encoding="utf-8")
    headers = json.loads((FIXTURES / "hf" / "wan21-t2v-safetensors-headers.json").read_text(encoding="utf-8"))
    client = FakeClient(info=info, files={"config.json": config}, headers=headers)
    manifests = load_manifests()
    registries = load_registries(manifests, root=Path("/nonexistent"), find_spec=lambda name: None)
    return client, resolve(repo, client=client, manifests=manifests, registries=registries)


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

    def test_audio8_converts_through_the_mlx_audio_port_with_no_download(self):
        payload = golden_audio8()
        self.validator.validate(payload)
        self.assertEqual(payload["verdict"], "convertible_after_install")
        self.assertEqual((payload["backend"], payload["reasons"]), ("mlx-audio", []))
        self.assertTrue(payload["custom_code"])
        self.assertEqual(payload["task"]["type"], "speech_to_text")
        self.assertEqual(payload["task"]["use_cases"], ["realtime_transcription", "transcription"])
        roles = {c["role"]: c for c in payload["components"]}
        self.assertEqual(sorted(roles), ["audio", "model", "text"])
        self.assertEqual([m["module"] for m in roles["model"]["matches"]], ["mlx_audio.stt.models.audio8_asr_infinite"])
        self.assertIn("mlx_audio.stt.models.voxtral_realtime", [m["module"] for m in roles["audio"]["matches"]])
        self.assertIn("mlx_lm.models.qwen2", [m["module"] for m in roles["text"]["matches"]])
        self.assertGreater(payload["bytes"], 8_000_000_000)

    def test_audio8_estimate_counts_only_the_ported_decoder_as_quantized(self):
        payload = golden_audio8()
        self.assertEqual(payload["estimated_output_bytes"], {"4": 3748321085, "8": 5291169597})

    def test_a_configless_repo_matching_a_port_signature_converts_through_that_port(self):
        client, payload = golden_laya()
        self.validator.validate(payload)
        self.assertEqual((payload["verdict"], payload["backend"], payload["model_type"]),
                         ("convertible_after_install", "mlx-embeddings", "laya"))
        self.assertEqual(payload["task"]["type"], "classification")
        self.assertEqual(client.raw_requests, [])
        self.assertEqual(client.header_requests, ["model.safetensors"])
        self.assertEqual([m["module"] for m in payload["components"][0]["matches"]], ["mlx_embeddings.classifiers.laya"])

    def test_laya_estimate_uses_the_port_group_size_and_its_signature_files(self):
        _, payload = golden_laya()
        self.assertEqual(payload["estimated_output_bytes"], {"4": 339744590, "8": 523900750})

    def test_download_bytes_count_what_a_snapshot_fetch_takes(self):
        _, laya = golden_laya()
        self.assertEqual(laya["download_bytes"], 846195574)
        self.assertGreater(laya["bytes"], 2_000_000_000)
        info = repo_info(pipeline_tag="text-generation",
                         files=("config.json", "model.safetensors", "model.onnx", "onnx/decoder.onnx", "flax_model.msgpack"))
        client = FakeClient(info=info, files={"config.json": json.dumps({"model_type": "qwen2"})})
        payload = self.run_resolve(client, installed=("mlx-lm",))
        self.assertEqual((payload["bytes"], payload["download_bytes"]), (5000, 2000))

    def test_a_subfolder_checkpoint_resolves_against_its_own_files(self):
        client, payload = golden_laya("https://huggingface.co/convaiinnovations/laya/tree/main/multilingual",
                                      config={"model_type": "bert"})
        self.validator.validate(payload)
        self.assertEqual(payload["source"]["subfolder"], "multilingual")
        self.assertEqual(payload["source"]["url"], "https://huggingface.co/convaiinnovations/laya/tree/main/multilingual")
        self.assertEqual((payload["verdict"], payload["backend"], payload["model_type"]),
                         ("convertible_after_install", "mlx-embeddings", "laya"))
        self.assertEqual(client.header_requests, ["multilingual/model.safetensors"])
        self.assertEqual(payload["bytes"], payload["download_bytes"])
        self.assertEqual(payload["download_bytes"], 678201636)
        self.assertEqual(payload["estimated_output_bytes"], {"4": 507061436, "8": 569287868})
        self.assertEqual(payload["q_bits"], [8])
        self.assertEqual(payload["files"]["python"], [])

    def test_a_missing_subfolder_is_unsupported_and_a_subfolder_config_is_read_there(self):
        _, payload = golden_laya("convaiinnovations/laya/nope")
        self.validator.validate(payload)
        self.assertEqual((payload["verdict"], payload["reasons"], payload["bytes"]), ("unsupported", ["subfolder_not_found"], 0))
        info = repo_info(pipeline_tag="text-generation", model_type="llama",
                         files=("README.md", "chat/config.json", "chat/model.safetensors", "chat/model.onnx"))
        client = FakeClient(info=info, files={"chat/config.json": json.dumps({"model_type": "qwen2"})},
                            headers={"chat/model.safetensors": {}})
        payload = self.run_resolve(client, installed=("mlx-lm",), text="org/name/chat")
        self.assertEqual((payload["model_type"], payload["verdict"], payload["q_bits"]), ("qwen2", "convertible", [4, 8]))
        self.assertEqual(client.raw_requests, ["chat/config.json"])
        self.assertEqual((payload["bytes"], payload["download_bytes"]), (3000, 2000))

    def test_a_recipe_repo_converts_from_its_pinned_base_and_lora(self):
        client, payload = qwen_image("abenzerps/Qwen-Image-2.1-Uncensored-GGUF")
        self.validator.validate(payload)
        self.assertEqual((payload["verdict"], payload["backend"], payload["model_type"]),
                         ("convertible_after_install", "mflux", "qwen_image_21"))
        self.assertEqual(payload["task"]["type"], "image_generation")
        self.assertEqual(payload["recipe"], {"base": "Qwen/Qwen-Image-2.1", "base_revision": "d26bb61231c349cf6b7896fa83353113880e1ba3",
                                             "lora": "qwen-image-2.1-uncensored-lora.safetensors", "lora_scale": 1.0})
        self.assertEqual(client.info_requests[-1], ("Qwen/Qwen-Image-2.1", "d26bb61231c349cf6b7896fa83353113880e1ba3"))
        self.assertEqual(payload["download_bytes"], 33134949561 + 33586704)
        self.assertIsNone(payload["estimated_output_bytes"])
        self.assertEqual(client.header_requests, [])

    def test_a_wan_text_to_video_repo_converts_through_the_mlx_video_port(self):
        client, payload = wan_t2v()
        self.validator.validate(payload)
        self.assertEqual((payload["verdict"], payload["backend"], payload["model_type"], payload["reasons"]),
                         ("convertible_after_install", "mlx-video", "t2v", []))
        self.assertEqual((payload["task"]["type"], payload["task"]["use_cases"], payload["task"]["source"]),
                         ("video_generation", ["video_generation"], "pipeline_tag"))
        self.assertEqual((payload["q_bits"], payload["download_bytes"], payload["recipe"]), ([4, 8], 17573837064, None))
        self.assertEqual(client.raw_requests, ["config.json"])
        self.assertIsNone(payload["estimated_output_bytes"])
        self.assertEqual(client.header_requests, [])

    def test_a_diffusers_pipeline_repo_converts_through_its_port(self):
        client, payload = qwen_image("Qwen/Qwen-Image-2.1")
        self.validator.validate(payload)
        self.assertEqual((payload["verdict"], payload["model_type"], payload["recipe"]), ("convertible_after_install", "qwen_image_21", None))
        self.assertEqual(client.raw_requests, ["model_index.json"])
        self.assertIsNone(payload["estimated_output_bytes"])

    def test_laya_golden_fixture_matches(self):
        expected = json.loads((FIXTURES / "intake-resolve-laya.json").read_text(encoding="utf-8"))
        self.assertEqual(golden_laya()[1], expected)

    def test_estimate_quantizes_matrix_weights_and_keeps_the_rest(self):
        header = {
            "__metadata__": {"format": "pt"},
            "layer.weight": {"dtype": "BF16", "shape": [128, 64], "data_offsets": [0, 0]},
            "layer.bias": {"dtype": "BF16", "shape": [128], "data_offsets": [0, 0]},
            "norm.weight": {"dtype": "BF16", "shape": [64], "data_offsets": [0, 0]},
            "odd.weight": {"dtype": "F32", "shape": [10, 30], "data_offsets": [0, 0]},
        }
        info = repo_info(pipeline_tag="text-generation", files=("config.json", "model.safetensors", "tokenizer.json"))
        client = FakeClient(info=info, files={"config.json": json.dumps({"model_type": "qwen2"})}, headers={"model.safetensors": header})
        payload = self.run_resolve(client, installed=("mlx-lm",))
        kept = 128 * 2 + 64 * 2 + 300 * 4
        quantized = {4: 8192 // 2 + 128 * 2 * 2, 8: 8192 + 128 * 2 * 2}
        self.assertEqual(payload["estimated_output_bytes"], {str(bits): quantized[bits] + kept + 2000 for bits in (4, 8)})
        self.assertEqual(client.header_requests, ["model.safetensors"])

    def test_estimate_survives_a_nested_pth_checkpoint_beside_root_safetensors_shards(self):
        header = {
            "__metadata__": {"format": "pt"},
            "layer.weight": {"dtype": "BF16", "shape": [128, 64], "data_offsets": [0, 0]},
        }
        shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        info = repo_info(
            pipeline_tag="text-generation", model_type="llama",
            files=("config.json", "tokenizer.json", "model.safetensors.index.json", *shards, "original/consolidated.00.pth"),
        )
        client = FakeClient(info=info, files={"config.json": json.dumps({"model_type": "llama"})},
                            headers={shard: header for shard in shards})
        payload = self.run_resolve(client, installed=("mlx-lm",))
        quantized = {4: 8192 // 2 + 128 * 2 * 2, 8: 8192 + 128 * 2 * 2}
        self.assertEqual(payload["estimated_output_bytes"], {str(bits): quantized[bits] * 2 + 2000 for bits in (4, 8)})
        self.assertEqual(client.header_requests, list(shards))

    def test_estimate_is_null_when_a_header_is_unreadable_or_the_verdict_is_not_a_conversion(self):
        client = FakeClient(info=repo_info(pipeline_tag="text-generation"), files={"config.json": json.dumps({"model_type": "qwen2"})})
        payload = self.run_resolve(client, installed=("mlx-lm",))
        self.assertIsNone(payload["estimated_output_bytes"])
        self.assertTrue(any("output size estimate unavailable" in warning for warning in payload["warnings"]))
        gguf_client = FakeClient(info=repo_info(files=("a-Q4_K_M.gguf",)))
        self.assertIsNone(self.run_resolve(gguf_client)["estimated_output_bytes"])
        self.assertEqual(gguf_client.header_requests, [])

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

    def test_already_mlx_keeps_its_required_backend_and_install_state(self):
        client = FakeClient(info=repo_info(library_name="mlx", tags=["mlx"]),
                            files={"config.json": json.dumps({"model_type": "glm4_moe_lite"})})
        for installed in ((), ("mlx-vlm",)):
            payload = self.run_resolve(client, installed=installed)
            self.assertEqual(payload["verdict"], "already_mlx")
            self.assertEqual(payload["backend"], "mlx-vlm")
            self.assertEqual(payload["backend_installed"], bool(installed))
            self.assertIsNone(payload["estimated_output_bytes"])

    def test_hub_failures(self):
        missing = self.run_resolve(FakeClient(info_error=HuggingFaceHTTPError(401, "HTTP 401")))
        self.assertEqual((missing["verdict"], missing["reasons"]), ("blocked", ["not_found_or_private"]))
        offline = self.run_resolve(FakeClient(info_error=TimeoutError("deadline")))
        self.assertEqual((offline["verdict"], offline["reasons"]), ("unknown", ["hub_unreachable"]))
        broken = self.run_resolve(FakeClient(info_error=http.client.HTTPException("boom")))
        self.assertEqual(broken["verdict"], "unknown")

    def test_unknown_architecture_with_custom_code_is_unsupported(self):
        config = {"model_type": "made_up_asr", "auto_map": {"AutoConfig": "configuration_made_up.Config"}}
        client = FakeClient(info=repo_info(pipeline_tag="automatic-speech-recognition"), files={"config.json": json.dumps(config)})
        payload = self.run_resolve(client, installed=("mlx-audio",))
        self.assertEqual((payload["verdict"], payload["reasons"], payload["backend"]),
                         ("unsupported", ["arch_not_in_registry", "custom_code"], None))

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
        self.assertEqual(envelope["data"]["verdict"], "convertible_after_install")

    def test_invalid_source_fails_cleanly(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["intake", "resolve", "https://github.com/x/y", "--json"])
        envelope = json.loads(out.getvalue())
        self.assertEqual(code, 2)
        self.assertEqual(envelope["error"]["code"], "invalid_source")


if __name__ == "__main__":
    unittest.main()

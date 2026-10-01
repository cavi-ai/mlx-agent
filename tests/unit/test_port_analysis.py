import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from mlx_agent.huggingface import HuggingFaceHTTPError
from mlx_agent.port_analysis import PortAnalysisError, analyze, analyze_with_sources, weight_prefixes

from .backend_fixtures import synthetic_manifests, synthetic_registries
from .test_intake import FakeClient

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "schemas" / "port-analysis.schema.json").read_text(encoding="utf-8"))

TOY_MODELING = '''
import torch.nn as nn


class ToyProjector(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x


class ToyAsrForConditionalGeneration(PreTrainedModel, GenerationMixin):
    def forward(self, input_features, input_ids):
        return None
'''


def toy_client():
    info = {
        "id": "org/toy-asr", "pipeline_tag": "automatic-speech-recognition", "library_name": "transformers",
        "tags": ["custom_code"], "gated": False, "config": {"model_type": "toy_asr"},
        "siblings": [
            {"rfilename": "config.json", "size": 10},
            {"rfilename": "model.safetensors.index.json", "size": 10},
            {"rfilename": "model-00001-of-00001.safetensors", "size": 100},
            {"rfilename": "vad_heads.safetensors", "size": 5},
            {"rfilename": "modeling_toy_asr.py", "size": 10},
            {"rfilename": "configuration_toy_asr.py", "size": 10},
        ],
    }
    config = {
        "model_type": "toy_asr", "architectures": ["ToyAsrForConditionalGeneration"],
        "processor_class": "ToyFeatureExtractor", "transformers_version": "5.13.0",
        "audio_config": {"model_type": "whisper_encoder"}, "text_config": {"model_type": "qwen2"},
    }
    index = {"weight_map": {
        "audio_tower.layers.0.w": "model-00001-of-00001.safetensors",
        "audio_tower.layers.1.w": "model-00001-of-00001.safetensors",
        "language_model.model.layers.0.w": "model-00001-of-00001.safetensors",
        "multi_modal_projector.linear.w": "model-00001-of-00001.safetensors",
    }}
    return FakeClient(info=info, files={
        "config.json": json.dumps(config),
        "model.safetensors.index.json": json.dumps(index),
        "modeling_toy_asr.py": TOY_MODELING,
        "configuration_toy_asr.py": "class ToyConfig(PretrainedConfig):\n    model_type = 'toy_asr'\n",
    })


def golden_toy():
    manifests = synthetic_manifests()
    return analyze("org/toy-asr", client=toy_client(), manifests=manifests, registries=synthetic_registries(manifests))


class PortAnalysisTests(unittest.TestCase):
    def test_toy_analysis(self):
        payload = golden_toy()
        Draft202012Validator(SCHEMA).validate(payload)
        status = {c["role"]: c["status"] for c in payload["components"]}
        self.assertEqual(status, {"model": "missing", "audio": "exists", "text": "exists"})
        prefixes = {p["prefix"]: p["component"] for p in payload["weights"]["prefixes"]}
        self.assertEqual(prefixes, {"audio_tower": "audio", "language_model": "text", "multi_modal_projector": None})
        self.assertEqual(payload["weights"]["extra_files"], ["vad_heads.safetensors"])
        self.assertEqual([f["name"] for f in payload["code"]["files"]], ["modeling_toy_asr.py", "configuration_toy_asr.py"])
        classes = [c["name"] for c in payload["code"]["files"][0]["classes"]]
        self.assertEqual(classes, ["ToyProjector", "ToyAsrForConditionalGeneration"])
        self.assertIn({"kind": "weights", "name": "multi_modal_projector"}, payload["missing"])
        self.assertIn({"kind": "component", "name": "model"}, payload["missing"])
        self.assertIn({"kind": "file", "name": "vad_heads.safetensors"}, payload["missing"])
        self.assertEqual(payload["processor_class"], "ToyFeatureExtractor")

    def test_golden_fixture_matches(self):
        expected = json.loads((ROOT / "tests" / "fixtures" / "port-analysis-synthetic.json").read_text(encoding="utf-8"))
        self.assertEqual(golden_toy(), expected)

    def test_sources_are_returned_for_the_draft(self):
        manifests = synthetic_manifests()
        _, sources = analyze_with_sources("org/toy-asr", client=toy_client(), manifests=manifests, registries=synthetic_registries(manifests))
        self.assertIn("ToyProjector", sources["modeling_toy_asr.py"])

    def test_failures_are_classified(self):
        manifests = synthetic_manifests()
        registries = synthetic_registries(manifests)
        cases = (
            (FakeClient(info_error=HuggingFaceHTTPError(401, "x")), "not_found_or_private"),
            (FakeClient(info_error=TimeoutError("x")), "hub_unreachable"),
            (FakeClient(info={"siblings": [], "gated": "auto"}), "gated"),
            (FakeClient(info={"siblings": [{"rfilename": "model.safetensors"}]}), "no_config"),
        )
        for client, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(PortAnalysisError) as caught:
                    analyze("org/x", client=client, manifests=manifests, registries=registries)
                self.assertEqual(caught.exception.code, code)

    def test_weight_prefixes_without_index(self):
        self.assertEqual(weight_prefixes(None, {"text"}), [])


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from mlx_agent import study_recipes as recipes


def settings(revision="a" * 40):
    return dict(revision=revision, prompt="A boat on a quiet lake", negative_prompt="", seed=42,
        width=256, height=144, num_frames=17, fps=8, inference_steps=20, guidance_scale=5.0)


def snapshot(root):
    folder = root / recipes.REPOSITORY / "snapshots" / ("a" * 40)
    folder.mkdir(parents=True)
    (folder / "model_index.json").write_text(json.dumps({"_class_name":"WanPipeline",
        "scheduler":["diffusers","UniPCMultistepScheduler"], "transformer":["diffusers","WanTransformer3DModel"],
        "text_encoder":["transformers","UMT5EncoderModel"], "tokenizer":["transformers","T5TokenizerFast"],
        "vae":["diffusers","AutoencoderKLWan"]}))
    for component in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler"):
        (folder / component).mkdir()
    for name in ("transformer/config.json", "vae/config.json", "text_encoder/config.json",
                 "tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json", "scheduler/scheduler_config.json"):
        (folder / name).write_text("{}")
    for name in ("transformer/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors", "text_encoder/model.safetensors"):
        (folder / name).write_bytes(b"weight fixture")
    return folder


@unittest.skipUnless(sys.version_info >= (3,12) and importlib.util.find_spec("pydantic"), "Optional lab runtime requires Python 3.12.")
class StudyRecipeTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0,str(Path(recipes.__file__).parent / "_lab"))
        self.addCleanup(lambda:sys.path.remove(str(Path(recipes.__file__).parent / "_lab")))
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve(); self.folder=snapshot(self.root)
        self.versions={"python":"3.12", "torch":"2", "diffusers":"1", "study-recipes":"sha256:fixture"}
        stub=patch.object(recipes,"runtime_versions",return_value=self.versions)
        stub.start(); self.addCleanup(stub.stop)

    def test_catalog_and_capture_are_offline_and_do_not_change_cache(self):
        before={p.relative_to(self.root):p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        catalog=recipes.catalog(self.root)
        self.assertEqual(catalog[0]["revisions"],["a"*40])
        result=recipes.prepare(settings(),self.root)
        self.assertEqual(result["model_revision"],"a"*40)
        self.assertEqual(result["runtime_versions"],self.versions)
        self.assertEqual(result["prompt"],settings()["prompt"])
        self.assertIn("tokenizer/tokenizer.json",result["source_checksums"])
        self.assertTrue(result["model_checksum"].startswith("sha256:"))
        recipes.verify(result,self.root)
        self.assertEqual(before,{p.relative_to(self.root):p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_settings_reject_invalid_values_before_cache_work(self):
        for change in ({"revision":"main"},{"width":257},{"num_frames":18},{"seed":True},
                       {"prompt":"  "},{"inference_steps":0},{"guidance_scale":float("nan")},{"extra":1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                recipes.prepare(dict(settings(),**change),self.root)

    def test_incomplete_weights_and_missing_runtime_are_refused(self):
        index=self.folder/"transformer/diffusion_pytorch_model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map":{"w":"missing.safetensors"}}))
        with self.assertRaisesRegex(ValueError,"missing|incomplete"):
            recipes.prepare(settings(),self.root)
        index.unlink()
        with patch.object(recipes,"runtime_versions",side_effect=ValueError("Install video runtime")):
            self.assertIn("Install video runtime",recipes.catalog(self.root)[0]["unavailable_reason"])
            with self.assertRaisesRegex(ValueError,"Install video runtime"):
                recipes.prepare(settings(),self.root)

    def test_hf_blob_links_are_allowed_and_external_links_refused(self):
        weight=self.folder/"vae/diffusion_pytorch_model.safetensors"
        blob=self.root/recipes.REPOSITORY/"blobs"/"hash"
        blob.parent.mkdir(); weight.rename(blob); weight.symlink_to(blob)
        recipes.prepare(settings(),self.root)
        external=self.root/"unrelated"; external.write_bytes(b"not a cache blob")
        weight.unlink(); weight.symlink_to(external)
        with self.assertRaisesRegex(ValueError,"outside|cache"):
            recipes.prepare(settings(),self.root)

    def test_changed_weights_assets_and_runtime_invalidate_captured_recipe(self):
        result=recipes.prepare(settings(),self.root)
        with patch.object(recipes,"runtime_versions",return_value=dict(self.versions,torch="changed")):
            with self.assertRaisesRegex(ValueError,"runtime"):
                recipes.verify(result,self.root)
        for name in ("tokenizer/tokenizer.json","transformer/diffusion_pytorch_model.safetensors"):
            path=self.folder/name; original=path.read_bytes(); path.write_bytes(b"{\"changed\":true}" if name.endswith(".json") else b"changed")
            with self.assertRaisesRegex(ValueError,"changed"):
                recipes.verify(result,self.root)
            path.write_bytes(original)

    def test_capture_refuses_a_file_changed_after_it_was_hashed(self):
        read = recipes._read
        def mutate(path, *args):
            result = read(path, *args)
            if path.name == "tokenizer.json":
                (self.folder / "model_index.json").write_text("{}")
            return result
        with patch.object(recipes, "_read", side_effect=mutate), self.assertRaisesRegex(ValueError, "changed"):
            recipes.prepare(settings(), self.root)

    def test_changed_runtime_is_rejected_before_network_or_model_load(self):
        from video_lab.domain import Manifest
        from video_lab.adapter_protocol import GenerationRequest, AdapterExecutionError
        from video_lab.adapters.wan_mps import WanMpsAdapter
        manifest = recipes.prepare(settings(), self.root)
        adapter = WanMpsAdapter(_cache_root=self.root)
        request = GenerationRequest(job_id="job", attempt_id="attempt", manifest=Manifest.model_validate(manifest), scratch_root=self.root/"scratch")
        with patch.object(recipes,"runtime_versions",return_value=dict(self.versions,torch="changed")), \
             patch.object(adapter,"_resolve_metadata",side_effect=AssertionError("Network metadata requested")), \
             patch.object(adapter,"_load",side_effect=AssertionError("Model loaded")), \
             self.assertRaisesRegex(AdapterExecutionError,"runtime changed"):
            adapter.generate(request, lambda _:None, lambda:False)

    def test_captured_recipe_execution_is_offline_and_rechecks_assets_after_generation(self):
        from video_lab.domain import Manifest
        from video_lab.adapter_protocol import GenerationRequest, AdapterExecutionError
        from video_lab.adapters.wan_mps import WanMpsAdapter
        manifest = recipes.prepare(settings(), self.root)
        loads = []
        class Pipeline:
            def to(self, device): self.device = device
            def __call__(self, **kwargs):
                if len(loads) == 2:
                    (self.root / "tokenizer/tokenizer.json").write_text("{\"changed\":true}")
                return SimpleNamespace(frames=[[b"frame"]])
        pipe = Pipeline(); pipe.root = self.folder
        def load(model_id, **kwargs):
            loads.append(kwargs)
            return pipe
        torch = SimpleNamespace(bfloat16="bf16", Generator=lambda **_:SimpleNamespace(manual_seed=lambda _:None),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:True,is_built=lambda:True)), __version__="2")
        adapter = WanMpsAdapter(_cache_root=self.root, _torch=torch, _pipeline_loader=load,
            _exporter=lambda frames,path,fps:Path(path).write_bytes(b"video fixture"), _device_inspector=lambda _: {"pipeline":"mps"})
        request = GenerationRequest(job_id="job", attempt_id="attempt", manifest=Manifest.model_validate(manifest), scratch_root=self.root/"scratch")
        with patch.object(adapter,"_resolve_metadata",side_effect=AssertionError("Network metadata requested")):
            result = adapter.generate(request, lambda _:None, lambda:False)
            self.assertTrue(result.video_path.is_file())
            self.assertEqual(loads[0]["revision"],"a"*40)
            self.assertTrue(loads[0]["local_files_only"])
            with self.assertRaisesRegex(AdapterExecutionError,"changed"):
                adapter.generate(request, lambda _:None, lambda:False)
            self.assertFalse((self.root/"scratch/output.mp4").exists())

    def test_workspace_captures_then_validates_and_queues_without_generation(self):
        from mlx_agent.lab_workspace import execute
        with patch.object(recipes,"cache_root",return_value=self.root):
            result=execute(self.root/"lab.sqlite3",self.root/"artifacts","reviewer",{"operation":"recipe-prepare","settings":settings()})
            draft={"baseline":result["manifest"],"variable_path":"inference_steps","alternate_value":8}
            preview=execute(self.root/"lab.sqlite3",self.root/"artifacts","reviewer",{"operation":"preview","draft":draft})
            self.assertEqual(preview["changed_paths"],["inference_steps"])
            created=execute(self.root/"lab.sqlite3",self.root/"artifacts","reviewer",{"operation":"create","draft":draft})
            self.assertEqual(len(created["job_ids"]),2)
            listed=execute(self.root/"lab.sqlite3",self.root/"artifacts","reviewer",{"operation":"snapshot"})
            self.assertEqual([job["state"] for job in listed["queue"]],["queued","queued"])

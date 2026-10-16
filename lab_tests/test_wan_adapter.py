from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg  # type: ignore[import-untyped]
import pytest

from video_lab.adapter_protocol import (
    AdapterExecutionError,
    GenerationCancelled,
    GenerationRequest,
    ProbeRequest,
)
from video_lab.adapters.wan_mps import (
    MODEL_ID,
    REQUESTED_REVISION,
    WanMpsAdapter,
    _count_video_frames,
    _sanitize_diagnostic,
)
from video_lab.domain import LoraBinding, Manifest


class TorchStub:
    __version__ = "2.7.0"
    bfloat16 = "bf16"

    class backends:
        class mps:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def is_built() -> bool:
                return True

    class Generator:
        def __init__(self, *, device: str) -> None:
            self.device, self.seed = device, None

        def manual_seed(self, seed: int) -> "TorchStub.Generator":
            self.seed = seed
            return self


class Pipeline:
    def __init__(self) -> None:
        self.device = "cpu"
        self.calls: list[dict[str, object]] = []

    def to(self, device: str) -> "Pipeline":
        self.device = device
        return self

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        callback = kwargs["callback_on_step_end"]
        callback(self, 0, None, {})  # type: ignore[operator]
        return SimpleNamespace(frames=[[b"frame"]])


def manifest(**changes: object) -> Manifest:
    values: dict[str, object] = dict(
        model_id=MODEL_ID,
        model_revision=REQUESTED_REVISION,
        model_checksum="sha256:x",
        adapter_id="diffusers.wan2.1-t2v-1.3b.mps.v1",
        adapter_version="1",
        runtime_versions={},
        prompt="a paper boat",
        negative_prompt="blur",
        seed=42,
        width=832,
        height=480,
        num_frames=17,
        fps=8,
        inference_steps=4,
        guidance_scale=5.0,
        scheduler="unipc",
        dtype="bfloat16",
        quantization="none",
        backend="mps",
        device="mps",
        memory_strategy="default",
        hardware_fingerprint="apple",
        os_fingerprint="macos",
    )
    values.update(changes)
    return Manifest(**values)  # type: ignore[arg-type]


def test_legacy_source_checksums_do_not_opt_into_recipe_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mlx_agent import study_recipes

    runtime, _, loads = adapter(tmp_path)
    monkeypatch.setattr(study_recipes, "verify", lambda *_: pytest.fail("Legacy recipe used capture checks"))
    request = GenerationRequest(job_id="job_legacy", attempt_id="attempt_legacy",
        manifest=manifest(source_checksums={"source": "sha256:legacy"}), scratch_root=tmp_path / "legacy")
    runtime.generate(request, lambda _: None, lambda: False)
    assert loads[0]["revision"] == "resolved-sha"


def adapter(
    tmp_path: Path,
    pipe: Pipeline | None = None,
    *,
    mps: bool = True,
    cache_root: Path | None = None,
) -> tuple[WanMpsAdapter, Pipeline, list[dict[str, object]]]:
    pipeline = pipe or Pipeline()
    loads: list[dict[str, object]] = []
    torch = TorchStub()
    torch.backends.mps.is_available = staticmethod(lambda: mps)

    def load(model_id: str, **kwargs: object) -> Pipeline:
        loads.append({"model_id": model_id, **kwargs})
        return pipeline

    def export(_frames: object, path: str, fps: int) -> None:
        Path(path).write_bytes(b"mp4")

    item = SimpleNamespace(rfilename="transformer/model.safetensors", size=123)
    api = SimpleNamespace(
        model_info=lambda *args, **kwargs: SimpleNamespace(sha="resolved-sha", siblings=[item])
    )

    def inspect(_path: Path) -> dict[str, object]:
        return {"size": (832, 480), "nframes": 17, "fps": 8.0}

    def count(_path: Path, _limit: int) -> int:
        return 17

    return (
        WanMpsAdapter(
            _torch=torch,
            _pipeline_loader=load,
            _exporter=export,
            _hf_api=api,
            _cache_root=cache_root,
            _video_inspector=inspect,
            _frame_counter=count,
        ),
        pipeline,
        loads,
    )


def test_probe_is_staged_and_metadata_does_not_load(tmp_path: Path) -> None:
    item = SimpleNamespace(rfilename="transformer/model.safetensors", size=123)
    api = SimpleNamespace(
        model_info=lambda model, revision, files_metadata: SimpleNamespace(
            sha="resolved", siblings=[item]
        )
    )
    subject, _, loads = adapter(tmp_path)
    subject._hf_api = api
    environment = subject.probe_phase("environment")
    metadata = subject.probe_phase("metadata")
    cache = subject.probe_phase("cache")
    assert environment.phase == "environment" and environment.ok
    assert metadata.diagnostics["requested_revision"] == REQUESTED_REVISION
    assert metadata.diagnostics["resolved_revision"] == "resolved"
    assert metadata.diagnostics["download_bytes"] == 123
    assert cache.phase == "cache"
    assert not loads


def test_generate_maps_manifest_exactly_and_uses_strict_local_mps(tmp_path: Path) -> None:
    subject, pipeline, loads = adapter(tmp_path)
    result = subject.generate(
        GenerationRequest(job_id="j", attempt_id="a", manifest=manifest(), scratch_root=tmp_path),
        lambda _: None,
        lambda: False,
    )
    assert loads == [
        {
            "model_id": MODEL_ID,
            "revision": "resolved-sha",
            "torch_dtype": "bf16",
            "local_files_only": True,
            "trust_remote_code": False,
        }
    ]
    assert pipeline.device == "mps"
    call = pipeline.calls[0]
    assert {
        key: call[key]
        for key in (
            "prompt",
            "negative_prompt",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
        )
    } == {
        "prompt": "a paper boat",
        "negative_prompt": "blur",
        "height": 480,
        "width": 832,
        "num_frames": 17,
        "num_inference_steps": 4,
        "guidance_scale": 5.0,
    }
    generator = call["generator"]
    assert generator.device == "cpu" and generator.seed == 42
    assert result.video_path == tmp_path / "output.mp4"
    assert result.adapter_telemetry["execution_device"] == "mps"
    assert result.adapter_telemetry["rng_device"] == "cpu"
    assert result.adapter_telemetry["requested_revision"] == REQUESTED_REVISION
    assert result.adapter_telemetry["resolved_revision"] == "resolved-sha"


def test_generate_accepts_array_like_frame_batch_without_boolean_coercion(tmp_path: Path) -> None:
    class ArrayLikeFrames:
        def __len__(self) -> int:
            return 17

        def __bool__(self) -> bool:
            raise ValueError("array truth value is ambiguous")

    class ArrayPipeline(Pipeline):
        def __call__(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(frames=[ArrayLikeFrames()])

    pipeline = ArrayPipeline()
    subject, _, _ = adapter(tmp_path, pipeline)

    result = subject.generate(
        GenerationRequest(job_id="j", attempt_id="a", manifest=manifest(), scratch_root=tmp_path),
        lambda _: None,
        lambda: False,
    )

    assert result.frame_count == 17


def test_generate_probe_requires_context_and_runs_real_generation(tmp_path: Path) -> None:
    subject, pipeline, _ = adapter(tmp_path)
    missing = subject.probe_phase("generate")
    assert not missing.ok and missing.diagnostics["error_type"] == "RuntimeError"
    request = GenerationRequest(
        job_id="j", attempt_id="a", manifest=manifest(), scratch_root=tmp_path
    )
    report = subject.probe_phase("generate", ProbeRequest(request, lambda _: None, lambda: False))
    assert report.ok and pipeline.calls
    assert report.diagnostics["frame_count"] == 17
    assert (tmp_path / "output.mp4").read_bytes() == b"mp4"


def test_generate_probe_rejects_bad_container_metadata_and_cleans_final(tmp_path: Path) -> None:
    subject, _, _ = adapter(tmp_path)
    subject._video_inspector = lambda _path: {"size": (1, 1), "nframes": 1, "fps": 1}
    request = GenerationRequest(
        job_id="j", attempt_id="a", manifest=manifest(), scratch_root=tmp_path
    )
    report = subject.probe_phase("generate", ProbeRequest(request, lambda _: None, lambda: False))
    assert not report.ok
    assert "invalid MP4 metadata" in str(report.diagnostics["error"])
    assert not (tmp_path / "output.mp4").exists()


def test_generate_probe_cleans_output_when_inspector_throws(tmp_path: Path) -> None:
    subject, _, _ = adapter(tmp_path)

    def throwing(_path: Path) -> dict[str, object]:
        raise RuntimeError("inspector failed")

    subject._video_inspector = throwing
    request = GenerationRequest(
        job_id="j", attempt_id="a", manifest=manifest(), scratch_root=tmp_path
    )
    report = subject.probe_phase("generate", ProbeRequest(request, lambda _: None, lambda: False))
    assert not report.ok
    assert not (tmp_path / "output.partial.mp4").exists()
    assert not (tmp_path / "output.mp4").exists()


@pytest.mark.parametrize("reported", [None, float("inf")])
def test_generate_probe_bounded_frame_count_fallback(tmp_path: Path, reported: object) -> None:
    subject, _, _ = adapter(tmp_path)
    subject._video_inspector = lambda _path: {"size": (832, 480), "nframes": reported, "fps": 8.0}
    limits: list[int] = []

    def count(_path: Path, limit: int) -> int:
        limits.append(limit)
        return 17

    subject._frame_counter = count
    request = GenerationRequest(
        job_id="j", attempt_id="a", manifest=manifest(), scratch_root=tmp_path
    )
    report = subject.probe_phase("generate", ProbeRequest(request, lambda _: None, lambda: False))
    assert report.ok
    assert limits == [18]


def test_default_frame_counter_counts_real_small_mp4_with_bound(tmp_path: Path) -> None:
    path = tmp_path / "small.mp4"
    writer = imageio_ffmpeg.write_frames(path, (2, 2), fps=8, macro_block_size=1)
    writer.send(None)
    for _ in range(3):
        writer.send(bytes((0, 0, 0)) * 4)
    writer.close()
    assert _count_video_frames(path, 4) == 3
    assert _count_video_frames(path, 2) == 2


def test_load_probe_resolves_alias_and_loads_exact_sha(tmp_path: Path) -> None:
    subject, _, loads = adapter(tmp_path)
    report = subject.probe_phase("load")
    assert report.ok
    assert report.diagnostics["requested_revision"] == REQUESTED_REVISION
    assert report.diagnostics["resolved_revision"] == "resolved-sha"
    assert loads[0]["revision"] == "resolved-sha"


def test_cache_probe_resolves_metadata_and_reports_missing_files(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    subject, _, loads = adapter(tmp_path, cache_root=cache)
    report = subject.probe_phase("cache")
    assert not report.ok
    assert report.diagnostics["missing_files"] == ("transformer/model.safetensors",)
    assert report.diagnostics["missing_bytes"] == 123
    assert report.diagnostics["cache_root"] == str(cache)
    assert not loads


def test_cache_root_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub-env"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home-env"))
    subject, _, _ = adapter(tmp_path, cache_root=explicit)
    assert subject._effective_cache_root() == explicit
    subject._cache_root = None
    assert subject._effective_cache_root() == tmp_path / "hub-env"
    monkeypatch.delenv("HF_HUB_CACHE")
    assert subject._effective_cache_root() == tmp_path / "home-env" / "hub"


def test_device_inspection_rejects_hidden_cpu_parameter(tmp_path: Path) -> None:
    class Parameter:
        device = "cpu"

    class Transformer:
        device = "mps"

        def parameters(self) -> list[Parameter]:
            return [Parameter()]

        def buffers(self) -> list[object]:
            return []

    pipeline = Pipeline()
    pipeline.components = {"transformer": Transformer()}  # type: ignore[attr-defined]
    subject, _, _ = adapter(tmp_path, pipeline)
    with pytest.raises(AdapterExecutionError, match=r"transformer\.parameters\[0\].*cpu"):
        subject.generate(
            GenerationRequest(
                job_id="j", attempt_id="a", manifest=manifest(), scratch_root=tmp_path
            ),
            lambda _: None,
            lambda: False,
        )


def test_device_inspection_ignores_null_optional_component() -> None:
    pipeline = Pipeline()
    pipeline.components = {"transformer_2": None}  # type: ignore[attr-defined]

    assert WanMpsAdapter._inspect_devices(pipeline) == {
        "pipeline.device": "cpu",
        "transformer_2": "non-executing",
    }


@pytest.mark.parametrize(
    "leak",
    [
        "https://user:password@example.test/file?token=secret",
        "Bearer abc.def.secret",
        "hf_supersecrettoken",
        f"{Path.home()}/.cache/huggingface/hub/private",
    ],
)
def test_diagnostic_sanitizer_removes_credentials_tokens_and_home_paths(leak: str) -> None:
    sanitized = _sanitize_diagnostic(RuntimeError(f"operator failed at {leak}"))
    assert "operator failed" in sanitized
    assert "user" not in sanitized and "password" not in sanitized and "secret" not in sanitized
    assert "token=" not in sanitized
    assert str(Path.home()) not in sanitized
    assert len(sanitized) <= 1000


@pytest.mark.parametrize(
    "change",
    [
        {"scheduler": "ddim"},
        {"dtype": "float16"},
        {"quantization": "int8"},
        {"loras": (LoraBinding(model_id="x", revision="r", checksum="c", weight=1),)},
    ],
)
def test_unsupported_settings_are_rejected_before_load(
    tmp_path: Path, change: dict[str, object]
) -> None:
    subject, _, loads = adapter(tmp_path)
    with pytest.raises(AdapterExecutionError, match="unsupported manifest"):
        subject.generate(
            GenerationRequest(
                job_id="j", attempt_id="a", manifest=manifest(**change), scratch_root=tmp_path
            ),
            lambda _: None,
            lambda: False,
        )
    assert not loads


def test_mps_is_required_without_cpu_fallback(tmp_path: Path) -> None:
    subject, _, loads = adapter(tmp_path, mps=False)
    with pytest.raises(AdapterExecutionError, match="MPS is unavailable"):
        subject.generate(
            GenerationRequest(
                job_id="j", attempt_id="a", manifest=manifest(), scratch_root=tmp_path
            ),
            lambda _: None,
            lambda: False,
        )
    assert not loads


def test_cancellation_propagates_and_removes_partial_files(tmp_path: Path) -> None:
    subject, _, _ = adapter(tmp_path)
    with pytest.raises(GenerationCancelled):
        subject.generate(
            GenerationRequest(
                job_id="j", attempt_id="a", manifest=manifest(), scratch_root=tmp_path
            ),
            lambda _: None,
            lambda: True,
        )
    assert not (tmp_path / "output.partial.mp4").exists()
    assert not (tmp_path / "output.mp4").exists()


def test_execution_error_is_typed_and_removes_partial_files(tmp_path: Path) -> None:
    class Broken(Pipeline):
        def __call__(self, **kwargs: object) -> object:
            (tmp_path / "output.partial.mp4").write_bytes(b"partial")
            raise RuntimeError("MPS allocation failed")

    subject, _, _ = adapter(tmp_path, Broken())
    with pytest.raises(AdapterExecutionError) as raised:
        subject.generate(
            GenerationRequest(
                job_id="j", attempt_id="a", manifest=manifest(), scratch_root=tmp_path
            ),
            lambda _: None,
            lambda: False,
        )
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert not (tmp_path / "output.partial.mp4").exists()


def test_caused_generation_error_is_sanitized_and_preserves_cause(tmp_path: Path) -> None:
    leaked = f"https://user:secret@example.test/run?token=hf_secret {Path.home()}/private"

    class Broken(Pipeline):
        def __call__(self, **kwargs: object) -> object:
            raise RuntimeError(leaked)

    subject, _, _ = adapter(tmp_path, Broken())
    with pytest.raises(AdapterExecutionError) as raised:
        subject.generate(
            GenerationRequest(
                job_id="j", attempt_id="a", manifest=manifest(), scratch_root=tmp_path
            ),
            lambda _: None,
            lambda: False,
        )
    assert isinstance(raised.value.__cause__, RuntimeError)
    message = str(raised.value)
    assert "RuntimeError" in message and "run" in message
    assert "secret" not in message and "token=" not in message and str(Path.home()) not in message

def test_pinned_weight_content_is_verified(tmp_path: Path) -> None:
    import hashlib
    from video_lab.adapters.wan_mps import _Metadata
    revision = "a" * 40
    cache = tmp_path / "hub"
    root = cache / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / revision
    weight = root / "transformer" / "model.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"fixture-weights")
    checksum = hashlib.sha256(b"fixture-weights").hexdigest()
    aggregate = hashlib.sha256(b"transformer/model.safetensors" + b"\0" + checksum.encode()).hexdigest()
    subject, _, _ = adapter(tmp_path, cache_root=cache)
    recipe = manifest(model_revision=revision, model_checksum="sha256:" + aggregate)
    metadata = _Metadata(revision, revision, {})
    subject._verify_pinned_weights(recipe, metadata)
    weight.write_bytes(b"modified-weights")
    with pytest.raises(AdapterExecutionError, match="content changed"):
        subject._verify_pinned_weights(recipe, metadata)

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from video_lab.adapter_protocol import (
    AdapterExecutionError,
    GenerationCancelled,
    GenerationRequest,
    GenerationResult,
)
from video_lab.adapters.fake import FakeVideoAdapter
from video_lab.domain import Manifest


@pytest.fixture
def manifest() -> Manifest:
    return Manifest(
        model_id="video-lab/fake",
        model_revision="r1",
        model_checksum="sha256:fake",
        adapter_id="video-lab.fake.v1",
        adapter_version="1",
        runtime_versions={"fake": "1"},
        prompt="a deterministic test",
        seed=42,
        width=64,
        height=48,
        num_frames=17,
        fps=8,
        inference_steps=4,
        guidance_scale=1.0,
        scheduler="fake",
        dtype="uint8",
        quantization="none",
        backend="cpu",
        device="cpu",
        memory_strategy="test",
        hardware_fingerprint="fake",
        os_fingerprint="fake",
    )


def make_request(manifest: Manifest, scratch_root: Path) -> GenerationRequest:
    return GenerationRequest(
        job_id="job-1", attempt_id="attempt-1", manifest=manifest, scratch_root=scratch_root
    )


def test_fake_adapter_is_deterministic(tmp_path: Path, manifest: Manifest) -> None:
    adapter = FakeVideoAdapter()
    first = adapter.generate(make_request(manifest, tmp_path / "a"), lambda _: None, lambda: False)
    second = adapter.generate(make_request(manifest, tmp_path / "b"), lambda _: None, lambda: False)

    assert first.video_path.read_bytes() == second.video_path.read_bytes()
    assert first.video_path.read_bytes()[4:8] == b"ftyp"
    assert first.effective_manifest == manifest
    assert (first.frame_count, first.fps, first.width, first.height) == (17, 8, 64, 48)


def test_fake_adapter_changes_bytes_when_inference_steps_change(
    tmp_path: Path, manifest: Manifest
) -> None:
    adapter = FakeVideoAdapter()
    first = adapter.generate(make_request(manifest, tmp_path / "a"), lambda _: None, lambda: False)
    changed = manifest.model_copy(update={"inference_steps": 5})
    second = adapter.generate(make_request(changed, tmp_path / "b"), lambda _: None, lambda: False)

    assert first.video_path.read_bytes() != second.video_path.read_bytes()


def test_generation_result_mapping_payloads_are_immutable(
    tmp_path: Path, manifest: Manifest
) -> None:
    result = FakeVideoAdapter().generate(
        make_request(manifest, tmp_path), lambda _: None, lambda: False
    )

    with pytest.raises(TypeError):
        result.runtime_versions["changed"] = "yes"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.adapter_telemetry["changed"] = True  # type: ignore[index]

    nested = result.model_copy(
        update={"adapter_telemetry": {"memory": {"samples": [1, {"rss": 2}]}}}
    )
    assert isinstance(nested.adapter_telemetry["memory"], MappingProxyType)
    memory = nested.adapter_telemetry["memory"]
    assert isinstance(memory, MappingProxyType)
    samples = memory["samples"]
    assert isinstance(samples, tuple)
    with pytest.raises(TypeError):
        memory["extra"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        samples[1]["rss"] = 4  # type: ignore[index]
    assert nested.model_dump(mode="json")["adapter_telemetry"] == {
        "memory": {"samples": [1, {"rss": 2}]}
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_generation_result_rejects_non_finite_telemetry(
    tmp_path: Path, manifest: Manifest, value: float
) -> None:
    with pytest.raises(ValidationError, match="unsupported non-finite JSON number"):
        GenerationResult(
            video_path=tmp_path / "output.mp4",
            effective_manifest=manifest,
            frame_count=17,
            fps=8,
            width=64,
            height=48,
            adapter_version="1",
            runtime_versions={},
            adapter_telemetry={"value": value},
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_generation_result_serialization_rejects_bypassed_non_finite_telemetry(
    tmp_path: Path, manifest: Manifest, value: float
) -> None:
    result = GenerationResult.model_construct(
        video_path=tmp_path / "output.mp4",
        effective_manifest=manifest,
        frame_count=17,
        fps=8,
        width=64,
        height=48,
        adapter_version="1",
        runtime_versions={},
        adapter_telemetry={"value": value},
    )

    with pytest.raises(PydanticSerializationError, match="unsupported non-finite JSON number"):
        result.model_dump(mode="json")


def test_fake_adapter_reports_one_ordered_progress_event_per_frame(
    tmp_path: Path, manifest: Manifest
) -> None:
    progress = []

    FakeVideoAdapter().generate(make_request(manifest, tmp_path), progress.append, lambda: False)

    assert [event.completed for event in progress] == list(range(1, 18))
    assert all(event.phase == "generating" and event.total == 17 for event in progress)
    assert [event.elapsed_seconds for event in progress] == sorted(
        event.elapsed_seconds for event in progress
    )


def test_fake_adapter_cancellation_removes_partial_and_final_artifacts(
    tmp_path: Path, manifest: Manifest
) -> None:
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 5

    with pytest.raises(GenerationCancelled):
        FakeVideoAdapter().generate(make_request(manifest, tmp_path), lambda _: None, cancelled)

    assert not (tmp_path / "output.partial.mp4").exists()
    assert not (tmp_path / "output.mp4").exists()


def assert_no_outputs(root: Path) -> None:
    assert not (root / "output.partial.mp4").exists()
    assert not (root / "output.mp4").exists()


class WriterDouble:
    def __init__(self, path: Path, *, fail_close: bool = False) -> None:
        self.path = path
        self.path.write_bytes(b"partial")
        self.fail_close = fail_close

    def send(self, frame: bytes | None) -> None:
        return None

    def close(self) -> None:
        if self.fail_close:
            raise RuntimeError("close failed")


def writer_factory(*, fail_close: bool = False) -> Callable[..., WriterDouble]:
    def create(path: Path, *_args: object, **_kwargs: object) -> WriterDouble:
        return WriterDouble(path, fail_close=fail_close)

    return create


def test_immediate_cancellation_cleans_outputs(tmp_path: Path, manifest: Manifest) -> None:
    adapter = FakeVideoAdapter(_writer_factory=writer_factory())

    with pytest.raises(GenerationCancelled):
        adapter.generate(make_request(manifest, tmp_path), lambda _: None, lambda: True)

    assert_no_outputs(tmp_path)


def test_final_pre_rename_cancellation_cleans_outputs(tmp_path: Path, manifest: Manifest) -> None:
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks == 18

    adapter = FakeVideoAdapter(_writer_factory=writer_factory())
    with pytest.raises(GenerationCancelled):
        adapter.generate(make_request(manifest, tmp_path), lambda _: None, cancelled)

    assert checks == 18
    assert_no_outputs(tmp_path)


def test_progress_callback_failure_preserves_primary_and_cleans_outputs(
    tmp_path: Path, manifest: Manifest
) -> None:
    def fail_progress(_event: object) -> None:
        raise LookupError("callback failed")

    adapter = FakeVideoAdapter(_writer_factory=writer_factory())
    with pytest.raises(AdapterExecutionError) as raised:
        adapter.generate(make_request(manifest, tmp_path), fail_progress, lambda: False)

    assert isinstance(raised.value.__cause__, LookupError)
    assert str(raised.value.__cause__) == "callback failed"
    assert_no_outputs(tmp_path)


def test_close_failure_does_not_mask_cancellation_and_cleanup_is_unconditional(
    tmp_path: Path, manifest: Manifest
) -> None:
    adapter = FakeVideoAdapter(_writer_factory=writer_factory(fail_close=True))

    with pytest.raises(GenerationCancelled) as raised:
        adapter.generate(make_request(manifest, tmp_path), lambda _: None, lambda: True)

    assert any("close failed" in note for note in raised.value.__notes__)
    assert_no_outputs(tmp_path)


def test_writer_close_failure_is_reported_and_cleans_outputs(
    tmp_path: Path, manifest: Manifest
) -> None:
    adapter = FakeVideoAdapter(_writer_factory=writer_factory(fail_close=True))

    with pytest.raises(AdapterExecutionError) as raised:
        adapter.generate(make_request(manifest, tmp_path), lambda _: None, lambda: False)

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "close failed"
    assert_no_outputs(tmp_path)


def test_close_failure_preserves_callback_failure_as_primary(
    tmp_path: Path, manifest: Manifest
) -> None:
    def fail_progress(_event: object) -> None:
        raise LookupError("callback failed")

    adapter = FakeVideoAdapter(_writer_factory=writer_factory(fail_close=True))
    with pytest.raises(AdapterExecutionError) as raised:
        adapter.generate(make_request(manifest, tmp_path), fail_progress, lambda: False)

    assert isinstance(raised.value.__cause__, LookupError)
    assert any("close failed" in note for note in raised.value.__cause__.__notes__)
    assert_no_outputs(tmp_path)


def _subprocess_digest(scratch: Path) -> str:
    code = """
import hashlib
import sys
from pathlib import Path
from video_lab.adapter_protocol import GenerationRequest
from video_lab.adapters.fake import FakeVideoAdapter
from video_lab.domain import Manifest
manifest = Manifest.model_validate_json(sys.argv[2])
request = GenerationRequest(
    job_id="job", attempt_id="attempt", manifest=manifest, scratch_root=Path(sys.argv[1])
)
result = FakeVideoAdapter().generate(request, lambda _: None, lambda: False)
print(hashlib.sha256(result.video_path.read_bytes()).hexdigest())
"""
    payload = manifest_payload()
    completed = subprocess.run(
        [sys.executable, "-c", code, str(scratch), json.dumps(payload)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def manifest_payload() -> dict[str, Any]:
    return {
        "model_id": "video-lab/fake",
        "model_revision": "r1",
        "model_checksum": "sha256:fake",
        "adapter_id": "video-lab.fake.v1",
        "adapter_version": "1",
        "runtime_versions": {"fake": "1"},
        "prompt": "a deterministic test",
        "seed": 42,
        "width": 64,
        "height": 48,
        "num_frames": 17,
        "fps": 8,
        "inference_steps": 4,
        "guidance_scale": 1.0,
        "scheduler": "fake",
        "dtype": "uint8",
        "quantization": "none",
        "backend": "cpu",
        "device": "cpu",
        "memory_strategy": "test",
        "hardware_fingerprint": "fake",
        "os_fingerprint": "fake",
    }


def test_fake_adapter_is_deterministic_across_spawned_processes(tmp_path: Path) -> None:
    assert _subprocess_digest(tmp_path / "one") == _subprocess_digest(tmp_path / "two")


def test_fake_mp4_is_decodable_with_expected_stream(tmp_path: Path, manifest: Manifest) -> None:
    import imageio_ffmpeg  # type: ignore[import-untyped]

    result = FakeVideoAdapter().generate(
        make_request(manifest, tmp_path), lambda _: None, lambda: False
    )
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-i",
        str(result.video_path),
        "-map",
        "0:v:0",
        "-f",
        "framemd5",
        "-",
    ]
    decoded = subprocess.run(command, check=False, capture_output=True, text=True)

    assert decoded.returncode == 0, decoded.stderr
    assert "Video: h264" in decoded.stderr
    frame_lines = [
        line for line in decoded.stdout.splitlines() if line and not line.startswith("#")
    ]
    assert len(frame_lines) == 17
    assert "#dimensions 0: 64x48" in decoded.stdout
    assert "#tb 0: 1/8" in decoded.stdout

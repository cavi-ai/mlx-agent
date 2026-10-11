import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Protocol, cast

import imageio_ffmpeg  # type: ignore[import-untyped]
import psutil  # type: ignore[import-untyped]

from video_lab.adapter_protocol import (
    AdapterExecutionError,
    CancellationCheck,
    GenerationCancelled,
    GenerationProgress,
    GenerationRequest,
    GenerationResult,
    ProgressCallback,
)
from video_lab.capabilities import AdapterCapabilities
from video_lab.domain import canonical_json


class _Writer(Protocol):
    def send(self, frame: bytes | None) -> None: ...

    def close(self) -> None: ...


_WriterFactory = Callable[..., _Writer]


def _default_writer_factory(*args: object, **kwargs: object) -> _Writer:
    return cast(_Writer, imageio_ffmpeg.write_frames(*args, **kwargs))


class FakeVideoAdapter:
    adapter_id = "video-lab.fake.v1"
    adapter_version = "1"

    def __init__(self, *, _writer_factory: _WriterFactory = _default_writer_factory) -> None:
        self._writer_factory = _writer_factory

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_id=self.adapter_id,
            model_patterns=("*",),
            backends=("cpu",),
            dtypes=("uint8",),
            quantizations=("none",),
            schedulers=("fake",),
            min_width=1,
            max_width=4096,
            width_multiple=1,
            min_height=1,
            max_height=4096,
            height_multiple=1,
            min_frames=17,
            max_frames=17,
            frame_rule="16n+1",
            supports_lora=False,
        )

    def probe(self) -> bool:
        return bool(imageio_ffmpeg.get_ffmpeg_exe())

    def generate(
        self,
        request: GenerationRequest,
        progress: ProgressCallback,
        cancelled: CancellationCheck,
    ) -> GenerationResult:
        scratch = request.scratch_root
        partial_path = scratch / "output.partial.mp4"
        final_path = scratch / "output.mp4"
        scratch.mkdir(parents=True, exist_ok=True)
        self._remove(partial_path, final_path)
        started = monotonic()
        manifest = request.manifest
        digest = hashlib.sha256(canonical_json(manifest).encode()).digest()
        frame = bytes(digest[:3]) * (manifest.width * manifest.height)
        writer: _Writer | None = None
        primary_error: Exception | None = None
        try:
            writer = self._writer_factory(
                partial_path,
                (manifest.width, manifest.height),
                fps=manifest.fps,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                macro_block_size=1,
                ffmpeg_log_level="error",
                output_params=[
                    "-map_metadata",
                    "-1",
                    "-fflags",
                    "+bitexact",
                    "-flags:v",
                    "+bitexact",
                ],
            )
            writer.send(None)
            for index in range(17):
                if cancelled():
                    raise GenerationCancelled("generation cancelled")
                writer.send(frame)
                progress(
                    GenerationProgress(
                        phase="generating",
                        completed=index + 1,
                        total=17,
                        message=f"rendered frame {index + 1} of 17",
                        elapsed_seconds=monotonic() - started,
                    )
                )
        except Exception as error:
            primary_error = error
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception as close_error:
                    if primary_error is None:
                        primary_error = close_error
                    else:
                        primary_error.add_note(f"writer close failed: {close_error}")

        if primary_error is not None:
            self._remove(partial_path, final_path)
            if isinstance(primary_error, GenerationCancelled):
                raise primary_error
            raise AdapterExecutionError("fake adapter generation failed") from primary_error

        try:
            with partial_path.open("rb") as artifact:
                os.fsync(artifact.fileno())
            if cancelled():
                raise GenerationCancelled("generation cancelled")
            partial_path.replace(final_path)
        except GenerationCancelled:
            self._remove(partial_path, final_path)
            raise
        except Exception as error:
            self._remove(partial_path, final_path)
            raise AdapterExecutionError("fake adapter generation failed") from error

        process = psutil.Process()
        return GenerationResult(
            video_path=final_path,
            effective_manifest=manifest,
            frame_count=17,
            fps=manifest.fps,
            width=manifest.width,
            height=manifest.height,
            adapter_version=self.adapter_version,
            runtime_versions={"imageio-ffmpeg": imageio_ffmpeg.__version__},
            adapter_telemetry={"process_rss_bytes": process.memory_info().rss},
        )

    @staticmethod
    def _remove(*paths: Path) -> None:
        for path in paths:
            path.unlink(missing_ok=True)

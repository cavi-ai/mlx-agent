from __future__ import annotations

import hashlib
import inspect
import os
import time
import zlib
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from PIL import Image

import video_lab.training_media as training_media
from video_lab.training_media import (
    TrainingMediaInspectionError,
    TrainingMediaSnapshot,
    _inspect_training_media,
    _open_training_media_snapshot,
    inspect_training_media,
    open_training_media_collection_snapshot,
    open_training_media_snapshot,
)


def test_collection_snapshot_rejects_root_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _image(root / "a.png", "PNG")
    moved = tmp_path / "moved"
    original = training_media._snapshot_collection_directory

    def replace_after_snapshot(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)  # type: ignore[arg-type]
        if args[1] == "":
            root.rename(moved)
            root.mkdir()

    monkeypatch.setattr(training_media, "_snapshot_collection_directory", replace_after_snapshot)
    with pytest.raises(TrainingMediaInspectionError, match="source_changed"):
        with open_training_media_collection_snapshot(root):
            pytest.fail("a replaced collection root must not be yielded")


def test_collection_snapshot_rejects_multi_item_ancestor_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    for name in ("a", "b"):
        (root / name).mkdir(parents=True)
        _image(root / name / f"{name}.png", "PNG")
    original = training_media._snapshot_collection_directory

    def swap_after_snapshot(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)  # type: ignore[arg-type]
        if args[1] == "":
            temporary = root / "temporary"
            (root / "a").rename(temporary)
            (root / "b").rename(root / "a")
            temporary.rename(root / "b")

    monkeypatch.setattr(training_media, "_snapshot_collection_directory", swap_after_snapshot)
    with pytest.raises(TrainingMediaInspectionError, match="source_changed"):
        with open_training_media_collection_snapshot(root):
            pytest.fail("a swapped collection ancestor must not be yielded")


def _image(path: Path, format: str) -> None:
    Image.new("RGB", (6, 4), color=(20, 40, 60)).save(path, format=format)


def _probe_result_code(
    *, width: int, height: int, fps: float, duration: float, frame_count: int
) -> str:
    metadata = {
        "fps": fps,
        "duration": duration,
        "size": [width, height],
        "frame_count": frame_count,
    }
    return f"import json;print(json.dumps({metadata!r}))"


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_inspection_public_api_accepts_only_a_path() -> None:
    """Exposing probe code or timeout injection would turn a private seam into public API."""
    signature = inspect.signature(inspect_training_media)

    assert tuple(signature.parameters) == ("path",)
    assert signature.parameters["path"].annotation in {Path, "Path"}


def test_snapshot_public_api_accepts_only_a_path() -> None:
    """Public snapshot acquisition must not expose race or probe test seams."""
    signature = inspect.signature(open_training_media_snapshot)

    assert tuple(signature.parameters) == ("path",)
    assert signature.parameters["path"].annotation in {Path, "Path"}


def test_snapshot_exposes_exact_immutable_bytes_only_inside_context(tmp_path: Path) -> None:
    """Consumers must train from the same immutable bytes that produced the metadata digest."""
    source = tmp_path / "still.png"
    _image(source, "PNG")
    expected = source.read_bytes()

    with open_training_media_snapshot(source) as snapshot:
        assert isinstance(snapshot, TrainingMediaSnapshot)
        assert snapshot.path.read_bytes() == expected
        assert snapshot.path.stat().st_mode & 0o222 == 0
        assert snapshot.media.sha256 == f"sha256:{hashlib.sha256(expected).hexdigest()}"
        snapshot_path = snapshot.path
        with pytest.raises((AttributeError, TypeError)):
            snapshot.path = source  # type: ignore[misc]

    assert not snapshot_path.exists()


def test_snapshot_cleanup_and_errors_do_not_expose_private_paths(tmp_path: Path) -> None:
    """Failed decoding must clean the private snapshot and reveal neither path nor cause."""
    source = tmp_path / "broken.png"
    source.write_bytes(b"not an image")
    observed: list[Path] = []

    with pytest.raises(TrainingMediaInspectionError) as raised:
        with _open_training_media_snapshot(source, after_snapshot=observed.append):
            pytest.fail("a corrupt snapshot must not be yielded")

    assert raised.value.code == "corrupt_media"
    assert raised.value.__cause__ is None
    assert str(source) not in str(raised.value)
    assert observed and all(not path.exists() for path in observed)


def test_snapshot_rejects_leaf_replacement_and_cleans_exact_snapshot(tmp_path: Path) -> None:
    """A leaf swap after copying must fail closed and remove the untrusted snapshot."""
    source = tmp_path / "still.png"
    moved = tmp_path / "moved.png"
    _image(source, "PNG")
    observed: list[Path] = []

    def replace_leaf() -> None:
        source.rename(moved)
        _image(source, "PNG")

    with pytest.raises(TrainingMediaInspectionError) as raised:
        with _open_training_media_snapshot(
            source,
            before_entry_stat=replace_leaf,
            after_snapshot=observed.append,
        ):
            pytest.fail("a replaced source must not be yielded")

    assert raised.value.code == "source_changed"
    assert observed and all(not path.exists() for path in observed)


@pytest.mark.parametrize(
    ("suffix", "format", "media_type"),
    [("jpg", "JPEG", "image/jpeg"), ("png", "PNG", "image/png"), ("webp", "WEBP", "image/webp")],
)
def test_inspection_decodes_supported_stills_and_hashes_exact_bytes(
    tmp_path: Path, suffix: str, format: str, media_type: str
) -> None:
    """Changing a still's bytes or parser format must change its bounded inspection result."""
    path = tmp_path / f"still.{suffix}"
    _image(path, format)

    inspected = inspect_training_media(path)

    assert inspected.media_type == media_type
    assert (inspected.width, inspected.height) == (6, 4)
    assert inspected.sha256 == f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    assert inspected.duration_seconds is None


def test_inspection_reports_valid_mp4_frame_rate_and_duration(tmp_path: Path) -> None:
    """Accepting a video without its 2–5 second measurable stream metadata is a bug."""
    path = tmp_path / "clip.mp4"
    frames = np.zeros((16, 4, 6, 3), dtype=np.uint8)
    iio.imwrite(path, frames, fps=8)

    inspected = inspect_training_media(path)

    assert inspected.media_type == "video/mp4"
    assert inspected.frame_count == 16
    assert inspected.fps == pytest.approx(8)
    assert inspected.duration_seconds == pytest.approx(2)


def test_inspection_rejects_symlinks_unsupported_and_corrupt_sources(tmp_path: Path) -> None:
    """Following a link or accepting corrupt/unsupported source bytes is a safety bug."""
    valid = tmp_path / "valid.png"
    _image(valid, "PNG")
    link = tmp_path / "linked.png"
    link.symlink_to(valid)
    unsupported = tmp_path / "still.gif"
    unsupported.write_bytes(b"GIF89a")
    corrupt = tmp_path / "broken.png"
    corrupt.write_bytes(b"not an image")

    for path, code in (
        (link, "symlink"),
        (unsupported, "unsupported_media"),
        (corrupt, "corrupt_media"),
    ):
        with pytest.raises(TrainingMediaInspectionError) as raised:
            inspect_training_media(path)
        assert raised.value.code == code
        assert str(path) not in str(raised.value)


def test_inspection_rejects_actual_source_byte_mutation_during_read(tmp_path: Path) -> None:
    """Hashing one byte stream while its source is being mutated is a TOCTOU bug."""
    path = tmp_path / "still.png"
    _image(path, "PNG")
    mutated = False

    def mutate_source(_: int) -> None:
        nonlocal mutated
        if mutated:
            return
        mutated = True
        with path.open("r+b") as source:
            source.seek(0)
            source.write(b"X")
            source.flush()
            os.fsync(source.fileno())

    with pytest.raises(TrainingMediaInspectionError) as raised:
        _inspect_training_media(path, after_chunk=mutate_source)

    assert raised.value.code == "source_changed"


def test_inspection_rejects_actual_leaf_replacement_after_snapshot(tmp_path: Path) -> None:
    """Returning the old inode after the named leaf was replaced is a TOCTOU bug."""
    path = tmp_path / "still.png"
    moved = tmp_path / "old.png"
    _image(path, "PNG")

    def replace_leaf() -> None:
        path.rename(moved)
        _image(path, "PNG")

    with pytest.raises(TrainingMediaInspectionError) as raised:
        _inspect_training_media(path, before_entry_stat=replace_leaf)

    assert raised.value.code == "source_changed"


def test_inspection_rejects_renamed_non_mp4_and_out_of_range_video_metadata(tmp_path: Path) -> None:
    """Treating a suffix as a container or accepting non-v1 stream limits is a safety bug."""
    renamed = tmp_path / "renamed.mp4"
    _image(renamed, "PNG")
    too_short = tmp_path / "short.mp4"
    iio.imwrite(too_short, np.zeros((8, 16, 16, 3), dtype=np.uint8), fps=8)

    for path, code in ((renamed, "unsupported_container"), (too_short, "media_out_of_bounds")):
        with pytest.raises(TrainingMediaInspectionError) as raised:
            inspect_training_media(path)
        assert raised.value.code == code


def test_inspection_checks_declared_size_before_buffering(tmp_path: Path) -> None:
    """Reading an already-oversized source before rejecting it would permit memory exhaustion."""
    path = tmp_path / "oversized.png"
    with path.open("wb") as source:
        source.truncate(training_media.MAX_STILL_BYTES + 1)

    started = time.monotonic()
    with pytest.raises(TrainingMediaInspectionError) as raised:
        inspect_training_media(path)

    assert raised.value.code == "media_out_of_bounds"
    assert time.monotonic() - started < 1


def test_still_byte_limit_is_inclusive_and_growth_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rejecting the exact byte limit or buffering growth past it is an off-by-one safety bug."""
    path = tmp_path / "still.png"
    _image(path, "PNG")
    original_size = path.stat().st_size
    monkeypatch.setattr(training_media, "MAX_STILL_BYTES", original_size)

    assert inspect_training_media(path).byte_size == original_size

    grew = False

    def grow_source(_: int) -> None:
        nonlocal grew
        if grew:
            return
        grew = True
        with path.open("ab") as source:
            source.write(b"growth")
            source.flush()
            os.fsync(source.fileno())

    with pytest.raises(TrainingMediaInspectionError) as raised:
        _inspect_training_media(path, after_chunk=grow_source)

    assert raised.value.code == "media_out_of_bounds"


def test_mp4_container_is_rejected_from_bounded_header_before_full_snapshot(
    tmp_path: Path,
) -> None:
    """Reading an entire non-MP4 before checking its leading box would waste bounded capacity."""
    path = tmp_path / "renamed.mp4"
    path.write_bytes(b"not-an-ftyp-box" + b"x" * (2 * 1024 * 1024))
    observed_bytes: list[int] = []

    with pytest.raises(TrainingMediaInspectionError) as raised:
        _inspect_training_media(path, after_chunk=observed_bytes.append)

    assert raised.value.code == "unsupported_container"
    assert observed_bytes
    assert max(observed_bytes) <= 4096


def test_video_byte_limit_is_inclusive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact video byte ceiling must pass and one byte below its size must reject."""
    path = tmp_path / "clip.mp4"
    iio.imwrite(path, np.zeros((16, 16, 16, 3), dtype=np.uint8), fps=8)
    byte_size = path.stat().st_size
    monkeypatch.setattr(training_media, "MAX_VIDEO_BYTES", byte_size)

    assert inspect_training_media(path).byte_size == byte_size

    monkeypatch.setattr(training_media, "MAX_VIDEO_BYTES", byte_size - 1)
    with pytest.raises(TrainingMediaInspectionError) as raised:
        inspect_training_media(path)
    assert raised.value.code == "media_out_of_bounds"


def test_still_edge_and_pixel_limits_are_inclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Boundary-sized images must pass while one edge or pixel beyond must fail."""
    path = tmp_path / "still.png"
    _image(path, "PNG")
    monkeypatch.setattr(training_media, "MAX_STILL_EDGE", 6)
    monkeypatch.setattr(training_media, "MAX_STILL_PIXELS", 24)

    assert (inspect_training_media(path).width, inspect_training_media(path).height) == (6, 4)

    monkeypatch.setattr(training_media, "MAX_STILL_EDGE", 5)
    with pytest.raises(TrainingMediaInspectionError) as edge_error:
        inspect_training_media(path)
    assert edge_error.value.code == "media_out_of_bounds"

    monkeypatch.setattr(training_media, "MAX_STILL_EDGE", 6)
    monkeypatch.setattr(training_media, "MAX_STILL_PIXELS", 23)
    with pytest.raises(TrainingMediaInspectionError) as pixel_error:
        inspect_training_media(path)
    assert pixel_error.value.code == "media_out_of_bounds"


@pytest.mark.parametrize("pillow_pixel_limit", [20, 10])
def test_inspection_converts_pillow_decompression_bomb_warning_and_error_to_stable_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pillow_pixel_limit: int
) -> None:
    """Pillow's warning or error must not leak decoder details or reach full decode."""
    path = tmp_path / "still.png"
    _image(path, "PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", pillow_pixel_limit)

    with pytest.raises(TrainingMediaInspectionError) as raised:
        inspect_training_media(path)

    assert raised.value.code == "media_out_of_bounds"
    assert raised.value.__cause__ is None


def test_inspection_rejects_oversized_png_header_before_pixel_decode(tmp_path: Path) -> None:
    """Loading pixels before enforcing header dimensions would enable decompression exhaustion."""
    path = tmp_path / "oversized-header.png"
    _image(path, "PNG")
    data = bytearray(path.read_bytes())
    data[16:20] = (training_media.MAX_STILL_EDGE + 1).to_bytes(4, "big")
    data[20:24] = (1).to_bytes(4, "big")
    data[29:33] = zlib.crc32(data[12:29]).to_bytes(4, "big")
    path.write_bytes(data)

    with pytest.raises(TrainingMediaInspectionError) as raised:
        inspect_training_media(path)

    assert raised.value.code == "media_out_of_bounds"


@pytest.mark.parametrize(
    ("width", "height", "fps", "duration", "frame_count", "accepted"),
    [
        (16, 16, 1.0, 2.0, 2, True),
        (3840, 2160, 120.0, 5.0, 600, True),
        (3841, 16, 1.0, 2.0, 2, False),
        (3840, 2161, 1.0, 2.0, 2, False),
        (16, 16, 0.999, 2.0, 2, False),
        (16, 16, 120.001, 2.0, 2, False),
        (16, 16, 1.0, 1.999, 2, False),
        (16, 16, 1.0, 5.001, 2, False),
        (16, 16, 1.0, 2.0, 1, False),
        (16, 16, 1.0, 2.0, 601, False),
    ],
)
def test_video_edge_pixel_fps_duration_and_frame_count_bounds_are_inclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    height: int,
    fps: float,
    duration: float,
    frame_count: int,
    accepted: bool,
) -> None:
    """A one-unit video metadata boundary error would accept unsafe work or reject valid clips."""
    path = tmp_path / "clip.mp4"
    iio.imwrite(path, np.zeros((16, 16, 16, 3), dtype=np.uint8), fps=8)
    monkeypatch.setattr(
        training_media,
        "_PROBE_CODE",
        _probe_result_code(
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            frame_count=frame_count,
        ),
    )

    if accepted:
        inspected = inspect_training_media(path)
        assert (inspected.width, inspected.height) == (width, height)
        assert inspected.fps == fps
        assert inspected.duration_seconds == duration
        assert inspected.frame_count == frame_count
    else:
        with pytest.raises(TrainingMediaInspectionError) as raised:
            inspect_training_media(path)
        assert raised.value.code == "media_out_of_bounds"


def test_fixed_probe_rejects_header_metadata_before_counting_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counting packets before rejecting oversized metadata would enable decoder exhaustion."""
    path = tmp_path / "clip.mp4"
    iio.imwrite(path, np.zeros((16, 16, 16, 3), dtype=np.uint8), fps=8)
    marker = tmp_path / "counted"
    fake_module = tmp_path / "imageio_ffmpeg.py"
    fake_module.write_text(
        "from pathlib import Path\n"
        "class Reader:\n"
        "    def __iter__(self): return self\n"
        "    def __next__(self):\n"
        "        return {'fps': 8.0, 'duration': 2.0, 'size': (5000, 16)}\n"
        "    def close(self): pass\n"
        "def read_frames(path): return Reader()\n"
        "def count_frames_and_secs(path):\n"
        f"    Path({str(marker)!r}).write_text('called')\n"
        "    return 16, 2.0\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    with pytest.raises(TrainingMediaInspectionError) as raised:
        inspect_training_media(path)

    assert raised.value.code == "media_out_of_bounds"
    assert not marker.exists()


def test_inspection_rejects_ancestor_swapped_to_outside_symlink(tmp_path: Path) -> None:
    """Reading outside bytes after an ancestor swap is a containment failure."""
    inside = tmp_path / "inside" / "nested"
    outside = tmp_path / "outside"
    inside.mkdir(parents=True)
    outside.mkdir()
    _image(inside / "still.png", "PNG")
    _image(outside / "still.png", "PNG")

    def swap(component: str) -> None:
        if component == "inside":
            (tmp_path / "inside").rename(tmp_path / "moved")
            (tmp_path / "inside").symlink_to(outside, target_is_directory=True)

    with pytest.raises(TrainingMediaInspectionError) as raised:
        _inspect_training_media(inside / "still.png", before_component=swap)

    assert raised.value.code == "unsafe_path"


def test_inspection_timeout_kills_and_reaps_real_child_and_grandchild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Timing out only the direct probe would leak a long-lived decoder grandchild."""
    path = tmp_path / "clip.mp4"
    iio.imwrite(path, np.zeros((16, 16, 16, 3), dtype=np.uint8), fps=8)
    child_pid_path = tmp_path / "child.pid"
    grandchild_pid_path = tmp_path / "grandchild.pid"
    grandchild_code = (
        "import os,time;from pathlib import Path;"
        f"Path({str(grandchild_pid_path)!r}).write_text(str(os.getpid()));"
        "time.sleep(30)"
    )
    probe_code = (
        "import os,subprocess,sys,time;from pathlib import Path;"
        f"Path({str(child_pid_path)!r}).write_text(str(os.getpid()));"
        f"subprocess.Popen([sys.executable,'-c',{grandchild_code!r}]);"
        "time.sleep(30)"
    )
    monkeypatch.setattr(training_media, "_PROBE_CODE", probe_code)
    started = time.monotonic()

    with pytest.raises(TrainingMediaInspectionError) as raised:
        _inspect_training_media(path, probe_timeout_seconds=0.3)

    assert raised.value.code == "probe_timeout"
    assert raised.value.__cause__ is None
    assert time.monotonic() - started < 2
    child_pid = int(child_pid_path.read_text())
    grandchild_pid = int(grandchild_pid_path.read_text())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and (_pid_exists(child_pid) or _pid_exists(grandchild_pid)):
        time.sleep(0.01)
    assert not _pid_exists(child_pid)
    assert not _pid_exists(grandchild_pid)

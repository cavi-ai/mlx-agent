"""Safe, bounded inspection of local training-media bytes."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import unicodedata
import warnings
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from PIL import Image, ImageOps, UnidentifiedImageError

from video_lab.training_models import TrainingMedia

_CHUNK_SIZE: Final = 1024 * 1024
_MP4_HEADER_BYTES: Final = 4096
MAX_STILL_BYTES: Final = 100 * 1024 * 1024
MAX_VIDEO_BYTES: Final = 2 * 1024 * 1024 * 1024
MAX_STILL_EDGE: Final = 8192
MAX_STILL_PIXELS: Final = 40_000_000
MAX_VIDEO_EDGE: Final = 3840
MAX_VIDEO_PIXELS: Final = 8_294_400
_MIN_VIDEO_EDGE: Final = 16
_MIN_VIDEO_FPS: Final = 1.0
_MAX_VIDEO_FPS: Final = 120.0
_MIN_VIDEO_DURATION: Final = 2.0
_MAX_VIDEO_DURATION: Final = 5.0
_MIN_VIDEO_FRAMES: Final = 2
_MAX_VIDEO_FRAMES: Final = 600
_PROBE_TIMEOUT_SECONDS: Final = 30
_PROBE_CODE: Final = """
import json
import sys

import imageio_ffmpeg

reader = imageio_ffmpeg.read_frames(sys.argv[1])
try:
    metadata = next(reader)
finally:
    reader.close()
fps = float(metadata["fps"])
duration = float(metadata["duration"])
width, height = (int(value) for value in metadata["size"])
if (
    not 16 <= width <= 3840
    or not 16 <= height <= 3840
    or width * height > 8294400
    or not 1.0 <= fps <= 120.0
    or not 2.0 <= duration <= 5.0
):
    print(json.dumps({"status": "out_of_bounds"}))
    raise SystemExit(0)
frame_count, _ = imageio_ffmpeg.count_frames_and_secs(sys.argv[1])
frame_count = int(frame_count)
if not 2 <= frame_count <= 600:
    print(json.dumps({"status": "out_of_bounds"}))
    raise SystemExit(0)
print(
    json.dumps(
        {
            "status": "ok",
            "fps": fps,
            "duration": duration,
            "size": [width, height],
            "frame_count": frame_count,
        }
    )
)
"""
_STILL_FORMATS: Final[dict[str, Literal["image/jpeg", "image/png", "image/webp"]]] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
_MP4_BRANDS: Final = {b"isom", b"iso2", b"avc1", b"mp41", b"mp42", b"M4V ", b"hvc1"}


class TrainingMediaInspectionError(ValueError):
    """Path-free media safety failure with a stable reason code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class TrainingMediaSnapshot:
    """Inspected metadata paired with its private, context-bound exact bytes."""

    media: TrainingMedia
    path: Path


@dataclass(frozen=True, slots=True)
class TrainingMediaCollectionSnapshot:
    """One exact, normalized collection observed through a single root descriptor."""

    entries: tuple[tuple[str, TrainingMediaSnapshot], ...]


def inspect_training_media(path: Path) -> TrainingMedia:
    """Safely inspect one regular local file without retaining its source location."""
    with open_training_media_snapshot(path) as snapshot:
        return snapshot.media


def open_training_media_snapshot(path: Path) -> AbstractContextManager[TrainingMediaSnapshot]:
    """Open verified exact bytes whose path remains valid only inside the context."""
    return _open_training_media_snapshot(path)


def open_training_media_collection_snapshot(
    root: Path,
) -> AbstractContextManager[TrainingMediaCollectionSnapshot]:
    """Snapshot a media tree without ever resolving discovered entries by pathname."""
    return _open_training_media_collection_snapshot(root)


@contextmanager
def _open_training_media_collection_snapshot(
    root: Path,
) -> Iterator[TrainingMediaCollectionSnapshot]:
    flags = (
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    root_fd = -1
    try:
        root_fd = os.open(root, flags)
        root_identity = _identity(os.fstat(root_fd))
        with tempfile.TemporaryDirectory(prefix="video-lab-collection-") as directory:
            snapshots: list[tuple[str, TrainingMediaSnapshot]] = []
            _snapshot_collection_directory(root_fd, "", Path(directory), snapshots)
            if not snapshots:
                raise TrainingMediaInspectionError("empty_media_folder")
            current = os.stat(root, follow_symlinks=False)
            if _identity(current) != root_identity or not stat.S_ISDIR(current.st_mode):
                raise TrainingMediaInspectionError("source_changed")
            yield TrainingMediaCollectionSnapshot(entries=tuple(snapshots))
    except TrainingMediaInspectionError:
        raise
    except OSError:
        raise TrainingMediaInspectionError("unsafe_path") from None
    finally:
        if root_fd >= 0:
            os.close(root_fd)


def _snapshot_collection_directory(
    directory_fd: int,
    prefix: str,
    temporary_root: Path,
    snapshots: list[tuple[str, TrainingMediaSnapshot]],
) -> None:
    before = _identity(os.fstat(directory_fd))
    names = sorted(os.listdir(directory_fd), key=lambda value: _normalized_entry_key(prefix, value))
    if len({_normalized_entry_key(prefix, name)[1] for name in names}) != len(names):
        raise TrainingMediaInspectionError("normalization_collision")
    for name in names:
        if name in {"", ".", ".."} or os.sep in name:
            raise TrainingMediaInspectionError("unsafe_path")
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        relative = f"{prefix}/{name}" if prefix else name
        normalized = unicodedata.normalize("NFC", relative)
        if stat.S_ISDIR(entry.st_mode):
            child = os.open(
                name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                _snapshot_collection_directory(child, normalized, temporary_root, snapshots)
                if _identity(os.fstat(child)) != _identity(entry):
                    raise TrainingMediaInspectionError("source_changed")
            finally:
                os.close(child)
        elif stat.S_ISREG(entry.st_mode):
            suffix = Path(name).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".mp4"}:
                raise TrainingMediaInspectionError("unsupported_media")
            fd = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                snapshot = temporary_root / f"{len(snapshots):08d}{suffix}"
                digest, size = _snapshot_open_descriptor(fd, snapshot, suffix == ".mp4")
                if _identity(os.fstat(fd)) != _identity(entry):
                    raise TrainingMediaInspectionError("source_changed")
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if _identity(current) != _identity(entry):
                    raise TrainingMediaInspectionError("source_changed")
                media = (
                    _inspect_mp4(snapshot, digest, size, timeout_seconds=_PROBE_TIMEOUT_SECONDS)
                    if suffix == ".mp4"
                    else _inspect_still(snapshot, digest, size)
                )
                os.chmod(snapshot, 0o400)
                snapshots.append((normalized, TrainingMediaSnapshot(media=media, path=snapshot)))
            finally:
                os.close(fd)
        else:
            raise TrainingMediaInspectionError("unsafe_path")
    if (
        _identity(os.fstat(directory_fd)) != before
        or sorted(os.listdir(directory_fd), key=lambda value: _normalized_entry_key(prefix, value))
        != names
    ):
        raise TrainingMediaInspectionError("source_changed")


def _normalized_entry_key(prefix: str, name: str) -> tuple[str, str]:
    value = unicodedata.normalize("NFC", f"{prefix}/{name}" if prefix else name)
    return value.casefold(), value


def _snapshot_open_descriptor(fd: int, destination: Path, require_mp4: bool) -> tuple[str, int]:
    identity = os.fstat(fd)
    maximum = MAX_VIDEO_BYTES if require_mp4 else MAX_STILL_BYTES
    if not stat.S_ISREG(identity.st_mode) or identity.st_size > maximum:
        raise TrainingMediaInspectionError("media_out_of_bounds")
    digest = hashlib.sha256()
    total = 0
    with destination.open("xb") as output:
        while total <= maximum:
            chunk = os.read(fd, min(_CHUNK_SIZE, maximum + 1 - total))
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            total += len(chunk)
    if total > maximum:
        raise TrainingMediaInspectionError("media_out_of_bounds")
    if require_mp4:
        with destination.open("rb") as stream:
            if not _is_mp4_container_header(stream.read(_MP4_HEADER_BYTES)):
                raise TrainingMediaInspectionError("unsupported_container")
    return f"sha256:{digest.hexdigest()}", total


def _inspect_training_media(
    path: Path,
    *,
    before_component: Callable[[str], None] | None = None,
    after_chunk: Callable[[int], None] | None = None,
    before_entry_stat: Callable[[], None] | None = None,
    probe_timeout_seconds: float | None = None,
) -> TrainingMedia:
    with _open_training_media_snapshot(
        path,
        before_component=before_component,
        after_chunk=after_chunk,
        before_entry_stat=before_entry_stat,
        probe_timeout_seconds=probe_timeout_seconds,
    ) as snapshot:
        return snapshot.media


@contextmanager
def _open_training_media_snapshot(
    path: Path,
    *,
    before_component: Callable[[str], None] | None = None,
    after_chunk: Callable[[int], None] | None = None,
    before_entry_stat: Callable[[], None] | None = None,
    probe_timeout_seconds: float | None = None,
    after_snapshot: Callable[[Path], None] | None = None,
) -> Iterator[TrainingMediaSnapshot]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".mp4"}:
        raise TrainingMediaInspectionError("unsupported_media")
    video = suffix == ".mp4"
    max_bytes = MAX_VIDEO_BYTES if video else MAX_STILL_BYTES
    with tempfile.TemporaryDirectory(prefix="video-lab-media-") as directory:
        os.chmod(directory, 0o700)
        snapshot = Path(directory) / ("clip.mp4" if video else "still.bin")
        if after_snapshot is not None:
            after_snapshot(snapshot)
        digest, byte_size = _snapshot_source(
            source,
            snapshot,
            max_bytes=max_bytes,
            require_mp4=video,
            before_component=before_component,
            after_chunk=after_chunk,
            before_entry_stat=before_entry_stat,
        )
        if video:
            timeout = (
                _PROBE_TIMEOUT_SECONDS if probe_timeout_seconds is None else probe_timeout_seconds
            )
            media = _inspect_mp4(snapshot, digest, byte_size, timeout_seconds=timeout)
        else:
            media = _inspect_still(snapshot, digest, byte_size)
        try:
            os.chmod(snapshot, 0o400)
        except OSError:
            raise TrainingMediaInspectionError("unsafe_path") from None
        yield TrainingMediaSnapshot(media=media, path=snapshot)


def _inspect_still(snapshot: Path, digest: str, byte_size: int) -> TrainingMedia:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(snapshot) as image:
                media_type = _STILL_FORMATS.get(image.format or "")
                if media_type is None:
                    raise TrainingMediaInspectionError("unsupported_media")
                _require_still_bounds(*image.size)
                normalized = ImageOps.exif_transpose(image)
                try:
                    normalized.load()
                    width, height = normalized.size
                finally:
                    if normalized is not image:
                        normalized.close()
    except TrainingMediaInspectionError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise TrainingMediaInspectionError("media_out_of_bounds") from None
    except (OSError, UnidentifiedImageError, ValueError):
        raise TrainingMediaInspectionError("corrupt_media") from None
    _require_still_bounds(width, height)
    return TrainingMedia(
        sha256=digest,
        media_type=media_type,
        byte_size=byte_size,
        width=width,
        height=height,
    )


def _require_still_bounds(width: int, height: int) -> None:
    if (
        not 1 <= width <= MAX_STILL_EDGE
        or not 1 <= height <= MAX_STILL_EDGE
        or width * height > MAX_STILL_PIXELS
    ):
        raise TrainingMediaInspectionError("media_out_of_bounds")


def _inspect_mp4(
    snapshot: Path, digest: str, byte_size: int, *, timeout_seconds: float
) -> TrainingMedia:
    metadata = _run_mp4_probe(snapshot, timeout_seconds=timeout_seconds)
    try:
        if metadata.get("status") == "out_of_bounds":
            raise TrainingMediaInspectionError("media_out_of_bounds")
        fps = _as_float(metadata["fps"])
        duration = _as_float(metadata["duration"])
        size = metadata["size"]
        if not isinstance(size, list | tuple) or len(size) != 2:
            raise ValueError
        width, height = _as_int(size[0]), _as_int(size[1])
        frame_count = _as_int(metadata["frame_count"])
    except TrainingMediaInspectionError:
        raise
    except (KeyError, TypeError, ValueError):
        raise TrainingMediaInspectionError("corrupt_media") from None
    if not _video_metadata_in_bounds(width, height, fps, duration, frame_count):
        raise TrainingMediaInspectionError("media_out_of_bounds")
    return TrainingMedia(
        sha256=digest,
        media_type="video/mp4",
        byte_size=byte_size,
        width=width,
        height=height,
        duration_seconds=duration,
        frame_count=frame_count,
        fps=fps,
    )


def _run_mp4_probe(snapshot: Path, *, timeout_seconds: float) -> dict[str, object]:
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", _PROBE_CODE, os.fspath(snapshot)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
    except OSError:
        raise TrainingMediaInspectionError("corrupt_media") from None
    try:
        stdout, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise TrainingMediaInspectionError("probe_timeout") from None
    if process.returncode != 0:
        raise TrainingMediaInspectionError("corrupt_media")
    try:
        metadata = json.loads(stdout)
    except (TypeError, ValueError, json.JSONDecodeError):
        raise TrainingMediaInspectionError("corrupt_media") from None
    if not isinstance(metadata, dict):
        raise TrainingMediaInspectionError("corrupt_media")
    return metadata


def _video_metadata_in_bounds(
    width: int, height: int, fps: float, duration: float, frame_count: int
) -> bool:
    return (
        _MIN_VIDEO_EDGE <= width <= MAX_VIDEO_EDGE
        and _MIN_VIDEO_EDGE <= height <= MAX_VIDEO_EDGE
        and width * height <= MAX_VIDEO_PIXELS
        and _MIN_VIDEO_FPS <= fps <= _MAX_VIDEO_FPS
        and _MIN_VIDEO_DURATION <= duration <= _MAX_VIDEO_DURATION
        and _MIN_VIDEO_FRAMES <= frame_count <= _MAX_VIDEO_FRAMES
    )


def _as_float(value: object) -> float:
    if not isinstance(value, str | int | float):
        raise TypeError
    return float(value)


def _as_int(value: object) -> int:
    if not isinstance(value, str | int):
        raise TypeError
    return int(value)


def _snapshot_source(
    path: Path,
    snapshot: Path,
    *,
    max_bytes: int,
    require_mp4: bool,
    before_component: Callable[[str], None] | None,
    after_chunk: Callable[[int], None] | None,
    before_entry_stat: Callable[[], None] | None,
) -> tuple[str, int]:
    directory_fd, name = _open_parent_fd(path, before_component=before_component)
    file_fd = -1
    snapshot_fd = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            file_fd = os.open(name, flags, dir_fd=directory_fd)
        except OSError:
            raise TrainingMediaInspectionError(
                "symlink" if path.is_symlink() else "unsafe_path"
            ) from None
        before_stat = os.fstat(file_fd)
        if not stat.S_ISREG(before_stat.st_mode):
            raise TrainingMediaInspectionError("unsafe_path")
        if before_stat.st_size > max_bytes:
            raise TrainingMediaInspectionError("media_out_of_bounds")
        snapshot_fd = os.open(
            snapshot,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        digest = hashlib.sha256()
        total = 0
        if require_mp4:
            header = os.read(file_fd, min(_MP4_HEADER_BYTES, max_bytes))
            total = _write_snapshot_chunk(snapshot_fd, digest, header, total)
            if after_chunk is not None:
                after_chunk(total)
            if not _is_mp4_container_header(header):
                raise TrainingMediaInspectionError("unsupported_container")
        while total < max_bytes:
            chunk = os.read(file_fd, min(_CHUNK_SIZE, max_bytes - total))
            if not chunk:
                break
            total = _write_snapshot_chunk(snapshot_fd, digest, chunk, total)
            if after_chunk is not None:
                after_chunk(total)
        after_stat = os.fstat(file_fd)
        if after_stat.st_size > max_bytes:
            raise TrainingMediaInspectionError("media_out_of_bounds")
        if before_entry_stat is not None:
            before_entry_stat()
        entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _identity(before_stat) != _identity(after_stat) or _identity(entry_stat) != _identity(
            after_stat
        ):
            raise TrainingMediaInspectionError("source_changed")
        os.fsync(snapshot_fd)
        return f"sha256:{digest.hexdigest()}", total
    except OSError:
        raise TrainingMediaInspectionError("unsafe_path") from None
    finally:
        if snapshot_fd >= 0:
            os.close(snapshot_fd)
        if file_fd >= 0:
            os.close(file_fd)
        os.close(directory_fd)


def _write_snapshot_chunk(snapshot_fd: int, digest: hashlib._Hash, chunk: bytes, total: int) -> int:
    view = memoryview(chunk)
    while view:
        written = os.write(snapshot_fd, view)
        if written <= 0:
            raise OSError("snapshot write failed")
        view = view[written:]
    digest.update(chunk)
    return total + len(chunk)


def _open_parent_fd(
    path: Path, *, before_component: Callable[[str], None] | None
) -> tuple[int, str]:
    absolute = os.path.abspath(path)
    parts = absolute.split(os.sep)
    if not absolute.startswith(os.sep) or any(part in {"", ".", ".."} for part in parts[1:]):
        raise TrainingMediaInspectionError("unsafe_path")
    root = os.open(os.sep, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0))
    current = root
    try:
        for part in parts[1:-1]:
            if before_component is not None:
                before_component(part)
            next_fd = os.open(
                part,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current,
            )
            if current != root:
                os.close(current)
            current = next_fd
    except OSError:
        if current != root:
            os.close(current)
        os.close(root)
        raise TrainingMediaInspectionError("unsafe_path") from None
    if current != root:
        os.close(root)
    return current, parts[-1]


def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _is_mp4_container_header(data: bytes) -> bool:
    offset = 0
    while offset + 8 <= len(data):
        box_size = int.from_bytes(data[offset : offset + 4], "big")
        box_type = data[offset + 4 : offset + 8]
        header_size = 8
        if box_size == 1:
            if offset + 16 > len(data):
                return False
            box_size = int.from_bytes(data[offset + 8 : offset + 16], "big")
            header_size = 16
        if box_size < header_size or offset + box_size > len(data):
            return False
        if box_type == b"ftyp":
            brands = {
                data[brand_offset : brand_offset + 4]
                for brand_offset in range(offset + header_size, offset + box_size, 4)
            }
            return bool(brands & _MP4_BRANDS)
        if box_size == 0:
            return False
        offset += box_size
    return False

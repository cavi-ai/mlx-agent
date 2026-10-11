from __future__ import annotations

import hashlib
import io
import os
import stat
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from video_lab.artifacts import Artifact
from video_lab.curated_models import CuratedFrameV1

_HASH_CHUNK_SIZE = 1024 * 1024
_DIRECTORY_FLAGS = (
    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_Identity = tuple[int, int, int, int, int, int, int]


class CuratedSourceError(Exception):
    """A curated source failed closed at its filesystem or decode boundary."""


@dataclass(frozen=True, slots=True)
class CuratedSourceLimits:
    max_width: int = 16_384
    max_height: int = 16_384
    max_pixels: int = 64 * 1024 * 1024
    max_source_bytes: int = 256 * 1024 * 1024
    max_decoded_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.max_width,
                self.max_height,
                self.max_pixels,
                self.max_source_bytes,
                self.max_decoded_bytes,
            )
        ):
            raise ValueError("curated source limits must be positive")


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    width: int
    height: int
    channels: int
    bit_depth: int
    color_space: str
    pixel_digest: str
    pixels: bytes


def decoded_pixel_digest(frame: DecodedFrame, limits: CuratedSourceLimits | None = None) -> str:
    """Return the sole canonical digest for a bounded immutable decoded frame."""
    source_limits = limits if limits is not None else CuratedSourceLimits()
    if any(type(value) is not int for value in (frame.width, frame.height, frame.channels)):
        raise CuratedSourceError("decoded frame dimensions must be exact integers")
    if frame.width <= 0 or frame.height <= 0:
        raise CuratedSourceError("decoded frame dimensions must be positive")
    expected_color_space = {3: "RGB", 4: "RGBA"}.get(frame.channels)
    if expected_color_space is None or frame.color_space != expected_color_space:
        raise CuratedSourceError("decoded frame channel and color-space metadata disagree")
    if type(frame.bit_depth) is not int or frame.bit_depth != 8:
        raise CuratedSourceError("decoded frame bit depth must be 8")
    if not isinstance(frame.pixels, bytes):
        raise CuratedSourceError("decoded frame pixels must be immutable bytes")
    pixel_count = frame.width * frame.height
    decoded_bytes = pixel_count * frame.channels
    if (
        frame.width > source_limits.max_width
        or frame.height > source_limits.max_height
        or pixel_count > source_limits.max_pixels
        or decoded_bytes > source_limits.max_decoded_bytes
    ):
        raise CuratedSourceError("decoded frame exceeds its resource bounds")
    if len(frame.pixels) != decoded_bytes:
        raise CuratedSourceError("decoded frame pixel length does not match its metadata")
    header = (f"{frame.width}x{frame.height}x{frame.channels}|uint8|{frame.color_space}").encode(
        "ascii"
    )
    return hashlib.sha256(b"curated-pixels-v1\0" + header + b"\0" + frame.pixels).hexdigest()


def validate_decoded_frame(frame: DecodedFrame, limits: CuratedSourceLimits | None = None) -> None:
    """Fail closed unless metadata, bytes, and the canonical digest agree exactly."""
    if decoded_pixel_digest(frame, limits) != frame.pixel_digest:
        raise CuratedSourceError("decoded frame canonical pixel digest mismatch")


@dataclass(frozen=True, slots=True)
class _SourceBinding:
    artifact: Artifact
    descriptor: int
    file_identity: _Identity
    ancestor_descriptors: tuple[int, ...]
    ancestor_identities: tuple[_Identity, ...]


class CuratedSourceLease:
    """Retain descriptor-bound PNG sources and revalidate them around decoding."""

    def __init__(
        self,
        *,
        root: Path,
        root_descriptor: int,
        root_identity: _Identity,
        root_component_descriptors: tuple[int, ...],
        root_component_identities: tuple[_Identity, ...],
        bindings: tuple[_SourceBinding, ...],
        descriptors: list[int],
        limits: CuratedSourceLimits,
    ) -> None:
        self._root = root
        self._root_descriptor = root_descriptor
        self._root_identity = root_identity
        self._root_component_descriptors = root_component_descriptors
        self._root_component_identities = root_component_identities
        self._bindings = {binding.artifact.relative_path: binding for binding in bindings}
        self._descriptors = descriptors
        self._limits = limits
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        root: Path,
        expected: Artifact | Iterable[Artifact],
        limits: CuratedSourceLimits | None = None,
    ) -> CuratedSourceLease:
        if not hasattr(os, "O_NOFOLLOW"):
            raise CuratedSourceError("O_NOFOLLOW is required for curated sources")
        artifacts = (expected,) if isinstance(expected, Artifact) else tuple(expected)
        if not artifacts:
            raise CuratedSourceError("at least one source descriptor is required")
        source_limits = limits if limits is not None else CuratedSourceLimits()
        root_path = Path(os.path.abspath(os.fspath(root)))
        descriptors: list[int] = []
        try:
            root_descriptor, root_component_descriptors, root_component_identities = (
                _open_absolute_directory(root_path, descriptors)
            )
            root_status = os.fstat(root_descriptor)
            if not stat.S_ISDIR(root_status.st_mode):
                raise CuratedSourceError("curated source root is not a directory")
            root_identity = _identity(root_status)
            bindings: list[_SourceBinding] = []
            seen_paths: set[str] = set()
            seen_inodes: set[tuple[int, int]] = set()
            for artifact in artifacts:
                if artifact.relative_path in seen_paths:
                    raise CuratedSourceError("duplicate curated source path")
                seen_paths.add(artifact.relative_path)
                binding = _open_binding(
                    root_descriptor=root_descriptor,
                    artifact=artifact,
                    limits=source_limits,
                    descriptors=descriptors,
                )
                inode = (binding.file_identity[0], binding.file_identity[1])
                if inode in seen_inodes:
                    raise CuratedSourceError("duplicate curated source inode")
                seen_inodes.add(inode)
                bindings.append(binding)
            lease = cls(
                root=root_path,
                root_descriptor=root_descriptor,
                root_identity=root_identity,
                root_component_descriptors=root_component_descriptors,
                root_component_identities=root_component_identities,
                bindings=tuple(bindings),
                descriptors=descriptors,
                limits=source_limits,
            )
            lease.reverify()
            return lease
        except CuratedSourceError:
            _close_descriptors(descriptors)
            raise
        except (OSError, ValueError, TypeError):
            _close_descriptors(descriptors)
            raise CuratedSourceError("curated sources cannot be opened safely") from None

    def read_frame(self, frame: CuratedFrameV1) -> DecodedFrame:
        self._require_open()
        binding = self._bindings.get(frame.source_path)
        if binding is None or binding.artifact.sha256 != frame.source_digest:
            raise CuratedSourceError("frame does not match its retained source descriptor")
        self.reverify()
        try:
            source = os.pread(
                binding.descriptor,
                binding.artifact.byte_size + 1,
                0,
            )
            if len(source) != binding.artifact.byte_size:
                raise CuratedSourceError("curated source size changed")
            decoded = _decode_png(source, self._limits)
        except CuratedSourceError:
            raise
        except Exception:
            raise CuratedSourceError("curated PNG decoding failed") from None
        self.reverify()
        if decoded.pixel_digest != frame.pixel_digest:
            raise CuratedSourceError("decoded pixels do not match the curated frame")
        return decoded

    def reverify(self) -> None:
        self._require_open()
        try:
            current_root_components = tuple(
                _identity(os.fstat(descriptor)) for descriptor in self._root_component_descriptors
            )
            if current_root_components != self._root_component_identities:
                raise CuratedSourceError("curated source root ancestry changed")
            if _identity(os.fstat(self._root_descriptor)) != self._root_identity:
                raise CuratedSourceError("curated source root changed")
            for binding in self._bindings.values():
                self._reverify_retained_binding(binding)
                self._reverify_canonical_binding(binding)
        except CuratedSourceError:
            raise
        except OSError:
            raise CuratedSourceError("curated source changed during verification") from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        descriptors = self._descriptors
        self._descriptors = []
        errors = _close_descriptors(descriptors)
        if errors:
            raise CuratedSourceError("curated source descriptor cleanup failed")

    def _require_open(self) -> None:
        if self._closed:
            raise CuratedSourceError("curated source lease is closed")

    def _reverify_retained_binding(self, binding: _SourceBinding) -> None:
        if tuple(_identity(os.fstat(fd)) for fd in binding.ancestor_descriptors) != (
            binding.ancestor_identities
        ):
            raise CuratedSourceError("curated source ancestor changed")
        if _identity(os.fstat(binding.descriptor)) != binding.file_identity:
            raise CuratedSourceError("curated source file changed")
        digest, size = _hash_descriptor(binding.descriptor, self._limits.max_source_bytes)
        if digest != binding.artifact.sha256 or size != binding.artifact.byte_size:
            raise CuratedSourceError("curated source content changed")

    def _reverify_canonical_binding(self, binding: _SourceBinding) -> None:
        temporary: list[int] = []
        try:
            root_descriptor, _, root_component_identities = _open_absolute_directory(
                self._root, temporary
            )
            if root_component_identities != self._root_component_identities:
                raise CuratedSourceError("curated source root ancestry path changed")
            if _identity(os.fstat(root_descriptor)) != self._root_identity:
                raise CuratedSourceError("curated source root path changed")
            current = root_descriptor
            parts = _relative_parts(binding.artifact.relative_path)
            canonical_ancestors: list[_Identity] = []
            for part in parts[:-1]:
                current = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
                temporary.append(current)
                canonical_ancestors.append(_identity(os.fstat(current)))
            leaf = os.open(parts[-1], _FILE_FLAGS, dir_fd=current)
            temporary.append(leaf)
            if tuple(canonical_ancestors) != binding.ancestor_identities:
                raise CuratedSourceError("curated source ancestor path changed")
            if _identity(os.fstat(leaf)) != binding.file_identity:
                raise CuratedSourceError("curated source file path changed")
            digest, size = _hash_descriptor(leaf, self._limits.max_source_bytes)
            if digest != binding.artifact.sha256 or size != binding.artifact.byte_size:
                raise CuratedSourceError("curated source canonical content changed")
        finally:
            if _close_descriptors(temporary):
                raise CuratedSourceError("curated source verification cleanup failed")


def _open_absolute_directory(
    path: Path, descriptors: list[int]
) -> tuple[int, tuple[int, ...], tuple[_Identity, ...]]:
    if not path.is_absolute():
        raise CuratedSourceError("curated source root must be absolute")
    current = os.open("/", _DIRECTORY_FLAGS)
    descriptors.append(current)
    component_descriptors = [current]
    component_identities = [_identity(os.fstat(current))]
    for part in path.parts[1:]:
        if part in {"", ".", ".."}:
            raise CuratedSourceError("curated source root path is invalid")
        current = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
        descriptors.append(current)
        component_descriptors.append(current)
        component_identities.append(_identity(os.fstat(current)))
    return current, tuple(component_descriptors), tuple(component_identities)


def _open_binding(
    *,
    root_descriptor: int,
    artifact: Artifact,
    limits: CuratedSourceLimits,
    descriptors: list[int],
) -> _SourceBinding:
    if artifact.media_type != "image/png":
        raise CuratedSourceError("curated source must be described as image/png")
    if artifact.byte_size <= 0 or artifact.byte_size > limits.max_source_bytes:
        raise CuratedSourceError("curated source byte size exceeds its bound")
    parts = _relative_parts(artifact.relative_path)
    current = root_descriptor
    ancestor_descriptors: list[int] = []
    ancestor_identities: list[_Identity] = []
    for part in parts[:-1]:
        current = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
        descriptors.append(current)
        opened = os.fstat(current)
        if not stat.S_ISDIR(opened.st_mode):
            raise CuratedSourceError("curated source ancestor is not a directory")
        ancestor_descriptors.append(current)
        ancestor_identities.append(_identity(opened))
    descriptor = os.open(parts[-1], _FILE_FLAGS, dir_fd=current)
    descriptors.append(descriptor)
    opened_file = os.fstat(descriptor)
    if not stat.S_ISREG(opened_file.st_mode):
        raise CuratedSourceError("curated source is not a regular file")
    if opened_file.st_nlink != 1:
        raise CuratedSourceError("curated source must have exactly one link")
    identity = _identity(opened_file)
    digest, size = _hash_descriptor(descriptor, limits.max_source_bytes)
    if size != artifact.byte_size or digest != artifact.sha256:
        raise CuratedSourceError("curated source descriptor mismatch")
    return _SourceBinding(
        artifact=artifact,
        descriptor=descriptor,
        file_identity=identity,
        ancestor_descriptors=tuple(ancestor_descriptors),
        ancestor_identities=tuple(ancestor_identities),
    )


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    path = Path(relative_path)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise CuratedSourceError("curated source path is invalid")
    return path.parts


def _identity(status: os.stat_result) -> _Identity:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_nlink,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _hash_descriptor(descriptor: int, maximum: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while size <= maximum:
        chunk = os.pread(descriptor, min(_HASH_CHUNK_SIZE, maximum + 1 - size), size)
        if not chunk:
            return digest.hexdigest(), size
        digest.update(chunk)
        size += len(chunk)
    raise CuratedSourceError("curated source exceeds its byte bound")


def _decode_png(source: bytes, limits: CuratedSourceLimits) -> DecodedFrame:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(source)) as verified:
                if verified.format != "PNG":
                    raise CuratedSourceError("curated source is not a PNG")
                verified.verify()
            with Image.open(io.BytesIO(source)) as image:
                width, height, channels, color_space = _validate_png_metadata(image, limits)
                image.load()
                pixel_array = np.array(image, dtype=np.uint8, copy=True, order="C")
    except CuratedSourceError:
        raise
    except Exception:
        raise CuratedSourceError("curated PNG is invalid") from None
    if pixel_array.dtype != np.uint8 or pixel_array.shape != (height, width, channels):
        raise CuratedSourceError("curated PNG decoded to an unexpected pixel layout")
    if not pixel_array.flags.owndata or not pixel_array.flags.c_contiguous:
        pixel_array = np.ascontiguousarray(pixel_array).copy()
    pixels = pixel_array.tobytes(order="C")
    provisional = DecodedFrame(
        width=width,
        height=height,
        channels=channels,
        bit_depth=8,
        color_space=color_space,
        pixel_digest="0" * 64,
        pixels=pixels,
    )
    return DecodedFrame(
        width=width,
        height=height,
        channels=channels,
        bit_depth=8,
        color_space=color_space,
        pixel_digest=decoded_pixel_digest(provisional, limits),
        pixels=pixels,
    )


def _validate_png_metadata(
    image: Image.Image, limits: CuratedSourceLimits
) -> tuple[int, int, int, str]:
    if image.format != "PNG":
        raise CuratedSourceError("curated source is not a PNG")
    if bool(getattr(image, "is_animated", False)) or int(getattr(image, "n_frames", 1)) != 1:
        raise CuratedSourceError("animated PNG sources are not allowed")
    if image.mode not in {"RGB", "RGBA"}:
        raise CuratedSourceError("curated PNG pixel mode is not approved")
    if image.info.get("icc_profile"):
        raise CuratedSourceError("embedded color profiles are not approved")
    if "transparency" in image.info:
        raise CuratedSourceError("implicit transparency is not approved")
    orientation = image.getexif().get(274, 1)
    if orientation != 1:
        raise CuratedSourceError("implicit image orientation is not approved")
    width, height = image.size
    channels = 3 if image.mode == "RGB" else 4
    pixels = width * height
    if (
        width <= 0
        or height <= 0
        or width > limits.max_width
        or height > limits.max_height
        or pixels > limits.max_pixels
        or pixels * channels > limits.max_decoded_bytes
    ):
        raise CuratedSourceError("curated PNG exceeds its decoded bounds")
    return width, height, channels, image.mode


def _close_descriptors(descriptors: list[int]) -> bool:
    failed = False
    while descriptors:
        descriptor = descriptors.pop()
        try:
            os.close(descriptor)
        except OSError:
            failed = True
    return failed

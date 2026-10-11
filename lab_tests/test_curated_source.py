from __future__ import annotations

import hashlib
import os
import socket
import warnings
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngImageFile

import video_lab.curated_source as curated_source
from video_lab.artifacts import Artifact
from video_lab.curated_models import CuratedFrameV1
from video_lab.curated_source import CuratedSourceError, CuratedSourceLease, CuratedSourceLimits

PIXEL_DIGEST = "e36c1e3173d38d31f7bcdda01a72c688f3e059a2065623f23aa289862eaa0cd6"


def _write_rgb_png(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (2, 1))
    image.putdata(((255, 0, 0), (0, 255, 0)))
    image.save(path, format="PNG")
    return path.read_bytes()


def _artifact(path: Path, root: Path) -> Artifact:
    source = path.read_bytes()
    return Artifact(
        id="art_frame_0001",
        job_id=None,
        sha256=hashlib.sha256(source).hexdigest(),
        byte_size=len(source),
        media_type="image/png",
        relative_path=path.relative_to(root).as_posix(),
        created_at="2026-07-29T00:00:00+00:00",
    )


def _frame(descriptor: Artifact, *, pixel_digest: str = PIXEL_DIGEST) -> CuratedFrameV1:
    return CuratedFrameV1(
        frame_id="frame_0001",
        frame_index=0,
        timestamp_numerator=0,
        timestamp_denominator=24,
        source_path=descriptor.relative_path,
        source_digest=descriptor.sha256,
        pixel_digest=pixel_digest,
        review_state="accepted",
        reviewer_evidence=("review_001",),
    )


def test_read_frame_binds_file_and_decoded_pixels(tmp_path: Path) -> None:
    source = tmp_path / "frames" / "0001.png"
    _write_rgb_png(source)
    descriptor = _artifact(source, tmp_path)

    lease = CuratedSourceLease.open(root=tmp_path, expected=descriptor)
    try:
        decoded = lease.read_frame(_frame(descriptor))
        assert decoded.pixel_digest == PIXEL_DIGEST
        assert (decoded.width, decoded.height, decoded.channels) == (2, 1, 3)
        assert (decoded.bit_depth, decoded.color_space) == (8, "RGB")
        assert decoded.pixels == bytes((255, 0, 0, 0, 255, 0))
        assert isinstance(decoded.pixels, bytes)
        lease.reverify()
    finally:
        lease.close()


def test_read_frame_preserves_explicit_rgba_alpha(tmp_path: Path) -> None:
    source = tmp_path / "rgba.png"
    Image.new("RGBA", (1, 1), (10, 20, 30, 40)).save(source, format="PNG")
    descriptor = _artifact(source, tmp_path)
    pixel_digest = "bd8bba63a36d6e564ffcef299c103836f608e97400e36d9025d58a62c6b74d8e"
    lease = CuratedSourceLease.open(root=tmp_path, expected=descriptor)
    try:
        decoded = lease.read_frame(_frame(descriptor, pixel_digest=pixel_digest))
        assert (decoded.channels, decoded.color_space) == (4, "RGBA")
        assert decoded.pixels == bytes((10, 20, 30, 40))
    finally:
        lease.close()


def test_limits_reject_nonpositive_configuration() -> None:
    with pytest.raises(ValueError):
        CuratedSourceLimits(max_width=0)


def test_open_rejects_source_bytes_above_configured_limit(tmp_path: Path) -> None:
    source = tmp_path / "0001.png"
    _write_rgb_png(source)
    descriptor = _artifact(source, tmp_path)
    limits = CuratedSourceLimits(max_source_bytes=descriptor.byte_size - 1)

    with pytest.raises(CuratedSourceError):
        CuratedSourceLease.open(
            root=tmp_path,
            expected=descriptor,
            limits=limits,
        )


@pytest.mark.parametrize("component", ["root", "ancestor", "file"])
def test_open_rejects_symlinked_path_components(component: str, tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    source = real_root / "frames" / "0001.png"
    _write_rgb_png(source)
    descriptor = _artifact(source, real_root)
    root = real_root
    if component == "root":
        root = tmp_path / "linked-root"
        root.symlink_to(real_root, target_is_directory=True)
    elif component == "ancestor":
        detached = tmp_path / "detached-frames"
        (real_root / "frames").rename(detached)
        (real_root / "frames").symlink_to(detached, target_is_directory=True)
    else:
        detached = tmp_path / "detached.png"
        source.rename(detached)
        source.symlink_to(detached)

    with pytest.raises(CuratedSourceError):
        CuratedSourceLease.open(root=root, expected=descriptor)


def test_open_rejects_intermediate_symlink_in_supplied_root(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_root = real_parent / "source-root"
    source = real_root / "0001.png"
    _write_rgb_png(source)
    descriptor = _artifact(source, real_root)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    lease: CuratedSourceLease | None = None

    try:
        with pytest.raises(CuratedSourceError):
            lease = CuratedSourceLease.open(
                root=linked_parent / "source-root",
                expected=descriptor,
            )
    finally:
        if lease is not None:
            lease.close()


@pytest.mark.parametrize("operation", ["reverify", "decode"])
def test_intermediate_root_symlink_swap_fails_closed(
    operation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_parent = tmp_path / "root-parent"
    root = root_parent / "source-root"
    source = root / "0001.png"
    _write_rgb_png(source)
    descriptor = _artifact(source, root)
    lease = CuratedSourceLease.open(root=root, expected=descriptor)
    detached_parent = tmp_path / "detached-parent"
    swapped = False

    def swap_intermediate_component() -> None:
        nonlocal swapped
        if swapped:
            return
        root_parent.rename(detached_parent)
        root_parent.symlink_to(detached_parent, target_is_directory=True)
        swapped = True

    try:
        if operation == "reverify":
            swap_intermediate_component()
            with pytest.raises(CuratedSourceError):
                lease.reverify()
        else:
            original_array = curated_source.np.array

            def array_then_swap(*args: Any, **kwargs: Any) -> Any:
                array = original_array(*args, **kwargs)
                swap_intermediate_component()
                return array

            monkeypatch.setattr(curated_source.np, "array", array_then_swap)
            with pytest.raises(CuratedSourceError):
                lease.read_frame(_frame(descriptor))
    finally:
        lease.close()
        if swapped:
            root_parent.unlink()
            detached_parent.rename(root_parent)


@pytest.mark.parametrize("kind", ["fifo", "device", "socket"])
def test_open_rejects_special_files_without_blocking(kind: str, tmp_path: Path) -> None:
    root = tmp_path
    special = root / "special.png"
    bound_socket: socket.socket | None = None
    if kind == "fifo":
        os.mkfifo(special)
    elif kind == "device":
        root = Path("/dev")
        special = root / "null"
    else:
        root = Path("/tmp")
        special = root / f"curated-{os.getpid()}-{tmp_path.name}.sock"
        special.unlink(missing_ok=True)
        bound_socket = socket.socket(socket.AF_UNIX)
        bound_socket.bind(str(special))
    descriptor = Artifact(
        id="art_special",
        job_id=None,
        sha256="0" * 64,
        byte_size=1,
        media_type="image/png",
        relative_path=special.name,
        created_at="2026-07-29T00:00:00+00:00",
    )
    try:
        with pytest.raises(CuratedSourceError):
            CuratedSourceLease.open(root=root, expected=descriptor)
    finally:
        if bound_socket is not None:
            bound_socket.close()
            special.unlink(missing_ok=True)


def test_open_rejects_duplicate_inode_and_hard_links(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_rgb_png(first)
    os.link(first, second)

    with pytest.raises(CuratedSourceError):
        CuratedSourceLease.open(
            root=tmp_path,
            expected=(_artifact(first, tmp_path), _artifact(second, tmp_path)),
        )


def test_open_rejects_source_replaced_after_descriptor_creation(tmp_path: Path) -> None:
    source = tmp_path / "0001.png"
    _write_rgb_png(source)
    descriptor = _artifact(source, tmp_path)
    Image.new("RGB", (2, 1), (0, 0, 255)).save(source, format="PNG")

    with pytest.raises(CuratedSourceError):
        CuratedSourceLease.open(root=tmp_path, expected=descriptor)


@pytest.mark.parametrize("mutation", ["ancestor", "root", "same_inode"])
def test_reverify_rejects_retained_identity_mutation(mutation: str, tmp_path: Path) -> None:
    source = tmp_path / "frames" / "0001.png"
    _write_rgb_png(source)
    descriptor = _artifact(source, tmp_path)
    lease = CuratedSourceLease.open(root=tmp_path, expected=descriptor)
    original_root_mode: int | None = None
    try:
        if mutation == "ancestor":
            detached = tmp_path / "detached"
            source.parent.rename(detached)
            source.parent.mkdir()
            (source.parent / source.name).write_bytes((detached / source.name).read_bytes())
        elif mutation == "root":
            root_stat = tmp_path.stat()
            original_root_mode = root_stat.st_mode
            tmp_path.chmod(root_stat.st_mode ^ 0o100)
        else:
            original_stat = source.stat()
            mutated = bytearray(source.read_bytes())
            mutated[-12] ^= 1
            source.write_bytes(mutated)
            os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

        with pytest.raises(CuratedSourceError):
            lease.reverify()
    finally:
        if original_root_mode is not None:
            tmp_path.chmod(original_root_mode)
        lease.close()


def test_read_rejects_path_replacement_before_decode(tmp_path: Path) -> None:
    source = tmp_path / "0001.png"
    _write_rgb_png(source)
    descriptor = _artifact(source, tmp_path)
    lease = CuratedSourceLease.open(root=tmp_path, expected=descriptor)
    replacement = tmp_path / "replacement.png"
    replacement.write_bytes(source.read_bytes())
    os.replace(replacement, source)
    try:
        with pytest.raises(CuratedSourceError):
            lease.read_frame(_frame(descriptor))
    finally:
        lease.close()


@pytest.mark.parametrize("when", ["during", "after"])
def test_read_rejects_replacement_during_decode(
    when: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "0001.png"
    _write_rgb_png(source)
    descriptor = _artifact(source, tmp_path)
    lease = CuratedSourceLease.open(root=tmp_path, expected=descriptor)

    def replace_path() -> None:
        replacement = tmp_path / f"{when}.png"
        replacement.write_bytes(source.read_bytes())
        os.replace(replacement, source)

    if when == "during":
        original_load = PngImageFile.load

        def load_then_replace(self: PngImageFile, *args: Any, **kwargs: Any) -> Any:
            loaded = original_load(self, *args, **kwargs)
            replace_path()
            return loaded

        monkeypatch.setattr(PngImageFile, "load", load_then_replace)
    else:
        original_array = curated_source.np.array

        def array_then_replace(*args: Any, **kwargs: Any) -> Any:
            array = original_array(*args, **kwargs)
            replace_path()
            return array

        monkeypatch.setattr(curated_source.np, "array", array_then_replace)
    try:
        with pytest.raises(CuratedSourceError):
            lease.read_frame(_frame(descriptor))
    finally:
        lease.close()


def test_read_rejects_oversized_png_before_decompression(tmp_path: Path) -> None:
    source = tmp_path / "wide.png"
    Image.new("RGB", (3, 1), (0, 0, 0)).save(source, format="PNG")
    descriptor = _artifact(source, tmp_path)
    limits = CuratedSourceLimits(
        max_width=2,
        max_height=2,
        max_pixels=4,
        max_source_bytes=1_024,
        max_decoded_bytes=12,
    )
    lease = CuratedSourceLease.open(
        root=tmp_path, expected=descriptor, limits=limits
    )
    try:
        with pytest.raises(CuratedSourceError):
            lease.read_frame(_frame(descriptor))
    finally:
        lease.close()


def test_read_rejects_decompressed_byte_limit(tmp_path: Path) -> None:
    source = tmp_path / "0001.png"
    _write_rgb_png(source)
    descriptor = _artifact(source, tmp_path)
    limits = CuratedSourceLimits(
        max_width=2,
        max_height=1,
        max_pixels=2,
        max_source_bytes=1_024,
        max_decoded_bytes=5,
    )
    lease = CuratedSourceLease.open(
        root=tmp_path, expected=descriptor, limits=limits
    )
    try:
        with pytest.raises(CuratedSourceError):
            lease.read_frame(_frame(descriptor))
    finally:
        lease.close()


def test_read_rejects_apng(tmp_path: Path) -> None:
    source = tmp_path / "animated.png"
    Image.new("RGB", (2, 1), (255, 0, 0)).save(
        source,
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (2, 1), (0, 255, 0))],
        duration=10,
        loop=0,
    )
    descriptor = _artifact(source, tmp_path)
    lease = CuratedSourceLease.open(root=tmp_path, expected=descriptor)
    try:
        with pytest.raises(CuratedSourceError):
            lease.read_frame(_frame(descriptor))
    finally:
        lease.close()


@pytest.mark.parametrize("mode", ["P", "CMYK", "I;16"])
def test_read_rejects_unapproved_pixel_modes(mode: str, tmp_path: Path) -> None:
    source = tmp_path / f"{mode.replace(';', '-')}.png"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        Image.new(mode, (2, 1)).save(source, format="JPEG" if mode == "CMYK" else "PNG")
    descriptor = _artifact(source, tmp_path)
    lease = CuratedSourceLease.open(root=tmp_path, expected=descriptor)
    try:
        with pytest.raises(CuratedSourceError):
            lease.read_frame(_frame(descriptor))
    finally:
        lease.close()


@pytest.mark.parametrize("metadata", ["icc", "orientation"])
def test_read_rejects_unapproved_color_and_orientation_metadata(
    metadata: str, tmp_path: Path
) -> None:
    source = tmp_path / f"{metadata}.png"
    image = Image.new("RGB", (2, 1), (255, 0, 0))
    if metadata == "icc":
        image.save(source, format="PNG", icc_profile=b"unapproved-profile")
    else:
        exif = Image.Exif()
        exif[274] = 6
        image.save(source, format="PNG", exif=exif)
    descriptor = _artifact(source, tmp_path)
    lease = CuratedSourceLease.open(root=tmp_path, expected=descriptor)
    try:
        with pytest.raises(CuratedSourceError):
            lease.read_frame(_frame(descriptor))
    finally:
        lease.close()


@pytest.mark.parametrize("damage", ["truncated", "malformed"])
def test_read_rejects_invalid_png_bytes(damage: str, tmp_path: Path) -> None:
    source = tmp_path / f"{damage}.png"
    valid = _write_rgb_png(source)
    if damage == "truncated":
        source.write_bytes(valid[: len(valid) // 2])
    else:
        source.write_bytes(valid[:8] + b"\xff\xff\xff\xffBAD!" + valid[16:])
    descriptor = _artifact(source, tmp_path)
    lease = CuratedSourceLease.open(root=tmp_path, expected=descriptor)
    try:
        with pytest.raises(CuratedSourceError):
            lease.read_frame(_frame(descriptor))
    finally:
        lease.close()


@pytest.mark.parametrize("mismatch", ["path", "source_digest", "pixel_digest"])
def test_read_requires_exact_manifest_binding(mismatch: str, tmp_path: Path) -> None:
    source = tmp_path / "0001.png"
    _write_rgb_png(source)
    descriptor = _artifact(source, tmp_path)
    frame = _frame(descriptor)
    if mismatch == "path":
        frame = frame.model_copy(update={"source_path": "other.png"})
    elif mismatch == "source_digest":
        frame = frame.model_copy(update={"source_digest": "1" * 64})
    else:
        frame = frame.model_copy(update={"pixel_digest": "1" * 64})
    lease = CuratedSourceLease.open(root=tmp_path, expected=descriptor)
    try:
        with pytest.raises(CuratedSourceError):
            lease.read_frame(frame)
    finally:
        lease.close()


def test_close_is_idempotent_and_closed_lease_cannot_be_used(tmp_path: Path) -> None:
    source = tmp_path / "0001.png"
    _write_rgb_png(source)
    descriptor = _artifact(source, tmp_path)
    lease = CuratedSourceLease.open(root=tmp_path, expected=descriptor)

    lease.close()
    lease.close()

    with pytest.raises(CuratedSourceError):
        lease.reverify()
    with pytest.raises(CuratedSourceError):
        lease.read_frame(_frame(descriptor))


def test_close_reports_cleanup_failure_after_closing_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "0001.png"
    _write_rgb_png(source)
    descriptor = _artifact(source, tmp_path)
    lease = CuratedSourceLease.open(root=tmp_path, expected=descriptor)
    original_close = curated_source.os.close
    failed = False

    def close_then_fail(descriptor_number: int) -> None:
        nonlocal failed
        original_close(descriptor_number)
        if not failed:
            failed = True
            raise OSError("injected close failure")

    monkeypatch.setattr(curated_source.os, "close", close_then_fail)
    with pytest.raises(CuratedSourceError):
        lease.close()
    lease.close()

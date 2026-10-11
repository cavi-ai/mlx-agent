from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import pytest

pytestmark = pytest.mark.wan_curated_host

_REVISION = re.compile(r"[0-9a-f]{40}")
_ROOT_ENVIRONMENTS = {
    "tokenizer": "VIDEO_LAB_WAN_TOKENIZER_ROOT",
    "text_encoder": "VIDEO_LAB_WAN_TEXT_ENCODER_ROOT",
    "vae": "VIDEO_LAB_WAN_VAE_ROOT",
    "scheduler": "VIDEO_LAB_WAN_SCHEDULER_ROOT",
    "source": "VIDEO_LAB_WAN_SOURCE_ROOT",
    "cache": "VIDEO_LAB_WAN_CACHE_ROOT",
}


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns


def _retain_snapshot_files(root: Path, stack: ExitStack) -> Callable[[], None]:
    retained: list[tuple[Path, int, tuple[int, int, int, int, int], str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("wan_curated_root_invalid")
        if not path.is_file():
            continue
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        stack.callback(os.close, descriptor)
        identity = _file_identity(os.fstat(descriptor))
        digest = hashlib.sha256(os.pread(descriptor, identity[3], 0)).hexdigest()
        retained.append((path.relative_to(root), descriptor, identity, digest))

    def reverify() -> None:
        for relative, descriptor, identity, digest in retained:
            if _file_identity(os.fstat(descriptor)) != identity:
                raise ValueError("wan_curated_root_changed")
            if hashlib.sha256(os.pread(descriptor, identity[3], 0)).hexdigest() != digest:
                raise ValueError("wan_curated_root_changed")
            current = os.open(
                root / relative,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                if _file_identity(os.fstat(current)) != identity:
                    raise ValueError("wan_curated_root_changed")
                if hashlib.sha256(os.pread(current, identity[3], 0)).hexdigest() != digest:
                    raise ValueError("wan_curated_root_changed")
            finally:
                os.close(current)

    return reverify


def _root_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    revisions = {role: f"{index + 1:040x}" for index, role in enumerate(_ROOT_ENVIRONMENTS)}
    for role, environment in _ROOT_ENVIRONMENTS.items():
        root = tmp_path / role
        root.mkdir(parents=True)
        (root / "config.json").write_text(json.dumps({"role": role}), encoding="utf-8")
        monkeypatch.setenv(environment, str(root))
    return revisions


def test_component_root_preflight_binds_exact_revisions_and_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revisions = _root_fixture(tmp_path / "sources", monkeypatch)
    with ExitStack() as stack:
        roots, bindings, reverify = _secure_component_roots(
            tmp_path / "snapshots", revisions, stack
        )
        assert set(roots) == set(_ROOT_ENVIRONMENTS.values())
        assert set(bindings) == set(_ROOT_ENVIRONMENTS)
        assert len(set(bindings.values())) == len(bindings)
        reverify()


@pytest.mark.parametrize(
    "attack", ["uppercase_revision", "symlink_root", "fifo", "root_replacement"]
)
def test_component_root_preflight_rejects_unsafe_authority(
    attack: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revisions = _root_fixture(tmp_path / "sources", monkeypatch)
    tokenizer = Path(os.environ["VIDEO_LAB_WAN_TOKENIZER_ROOT"])
    if attack == "uppercase_revision":
        revisions["tokenizer"] = "A" * 40
    elif attack == "symlink_root":
        linked = tmp_path / "linked-tokenizer"
        linked.symlink_to(tokenizer, target_is_directory=True)
        monkeypatch.setenv("VIDEO_LAB_WAN_TOKENIZER_ROOT", str(linked))
    elif attack == "fifo":
        os.mkfifo(tokenizer / "weights.fifo")
    else:
        from video_lab.trainers import wan_factory

        original = wan_factory.secure_snapshot_pinned_root

        def replace_after_snapshot(*args: object, **kwargs: object):
            result = original(*args, **kwargs)
            source = args[0]
            if source.path == tokenizer:
                detached = tmp_path / "detached-tokenizer"
                tokenizer.rename(detached)
                tokenizer.mkdir()
                (tokenizer / "config.json").write_bytes((detached / "config.json").read_bytes())
            return result

        monkeypatch.setattr(wan_factory, "secure_snapshot_pinned_root", replace_after_snapshot)
    with pytest.raises(ValueError, match="wan_curated_root_invalid"):
        with ExitStack() as stack:
            _secure_component_roots(tmp_path / "snapshots", revisions, stack)


@pytest.mark.parametrize("attack", ["same_bytes_new_inode", "ancestor_swap"])
def test_component_root_reverification_rejects_snapshot_identity_swaps(
    attack: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revisions = _root_fixture(tmp_path / "sources", monkeypatch)
    with ExitStack() as stack:
        roots, _, reverify = _secure_component_roots(tmp_path / "snapshots", revisions, stack)
        root = roots["VIDEO_LAB_WAN_TOKENIZER_ROOT"]
        target = root / "config.json"
        payload = target.read_bytes()
        if attack == "same_bytes_new_inode":
            root.chmod(0o700)
            target.chmod(0o600)
            detached = root / "detached.json"
            target.rename(detached)
            target.write_bytes(payload)
        else:
            root.chmod(0o700)
            detached_root = root.with_name(root.name + "-detached")
            root.rename(detached_root)
            root.mkdir()
            (root / "config.json").write_bytes(payload)
        with pytest.raises(ValueError, match="wan_curated_root_changed"):
            reverify()


def _secure_component_roots(
    tmp_path: Path, revisions: object, stack: ExitStack
) -> tuple[dict[str, Path], dict[str, str], Callable[[], None]]:
    if not isinstance(revisions, dict) or set(revisions) != set(_ROOT_ENVIRONMENTS):
        raise ValueError("wan_curated_root_invalid")
    try:
        from video_lab.domain import canonical_json
        from video_lab.trainers.wan_factory import (
            WanPinnedLocalRoot,
            WanSnapshotCopyLimits,
            _manifest_identity,
            _open_directory_chain,
            root_manifest_digest,
            secure_snapshot_pinned_root,
        )

        roots: dict[str, Path] = {}
        bindings: dict[str, str] = {}
        limits = WanSnapshotCopyLimits(max_files=100_000, max_bytes=64 << 30, max_depth=32)
        tmp_path.mkdir(parents=True, mode=0o700)
        snapshots = tmp_path / "qualified-component-roots"
        snapshots.mkdir(mode=0o700)
        for role, environment in _ROOT_ENVIRONMENTS.items():
            revision = revisions[role]
            value = os.environ.get(environment)
            if (
                not isinstance(revision, str)
                or _REVISION.fullmatch(revision) is None
                or value is None
            ):
                raise ValueError
            source = Path(value)
            descriptors = _open_directory_chain(source)
            identities = tuple(_manifest_identity(os.fstat(item)) for item in descriptors)
            try:
                manifest = root_manifest_digest(
                    source,
                    max_files=limits.max_files,
                    max_bytes=limits.max_bytes,
                    max_depth=limits.max_depth,
                )
                snapshot = secure_snapshot_pinned_root(
                    WanPinnedLocalRoot(source, revision, manifest),
                    snapshots / role,
                    limits=limits,
                )
                retraversed = _open_directory_chain(source)
                try:
                    retraversed_identities = tuple(
                        _manifest_identity(os.fstat(item)) for item in retraversed
                    )
                finally:
                    for descriptor in reversed(retraversed):
                        os.close(descriptor)
                if (
                    tuple(_manifest_identity(os.fstat(item)) for item in descriptors) != identities
                    or retraversed_identities != identities
                ):
                    raise ValueError
            finally:
                for descriptor in reversed(descriptors):
                    os.close(descriptor)
            roots[environment] = snapshot.path
            bindings[role] = hashlib.sha256(
                canonical_json(
                    {"role": role, "revision": revision, "manifest_digest": manifest}
                ).encode()
            ).hexdigest()
        retained: list[
            tuple[tuple[int, ...], tuple[object, ...], Path, str, Callable[[], None]]
        ] = []
        for role, environment in _ROOT_ENVIRONMENTS.items():
            descriptors = tuple(_open_directory_chain(roots[environment]))
            identities = tuple(_manifest_identity(os.fstat(item)) for item in descriptors)
            file_reverify = _retain_snapshot_files(roots[environment], stack)
            retained.append(
                (descriptors, identities, roots[environment], bindings[role], file_reverify)
            )
            for descriptor in descriptors:
                stack.callback(os.close, descriptor)

        def reverify() -> None:
            for descriptors, identities, root, binding, file_reverify in retained:
                if tuple(_manifest_identity(os.fstat(item)) for item in descriptors) != identities:
                    raise ValueError("wan_curated_root_changed")
                current = _open_directory_chain(root)
                try:
                    if tuple(_manifest_identity(os.fstat(item)) for item in current) != identities:
                        raise ValueError("wan_curated_root_changed")
                finally:
                    for descriptor in reversed(current):
                        os.close(descriptor)
                manifest = root_manifest_digest(
                    root,
                    max_files=limits.max_files,
                    max_bytes=limits.max_bytes,
                    max_depth=limits.max_depth,
                )
                matched = False
                for role, environment in _ROOT_ENVIRONMENTS.items():
                    if roots[environment] == root:
                        expected = hashlib.sha256(
                            canonical_json(
                                {
                                    "role": role,
                                    "revision": revisions[role],
                                    "manifest_digest": manifest,
                                }
                            ).encode()
                        ).hexdigest()
                        matched = expected == binding
                        break
                if not matched:
                    raise ValueError("wan_curated_root_changed")
                file_reverify()

        reverify()
        return roots, bindings, reverify
    except (OSError, TypeError, ValueError):
        raise ValueError("wan_curated_root_invalid") from None


def _preflight(
    tmp_path: Path, stack: ExitStack
) -> tuple[dict[str, Path], dict[str, str], dict[str, str], Callable[[], None]]:
    if os.environ.get("VIDEO_LAB_RUN_WAN_CURATED_CANARY") != "1":
        pytest.skip("set VIDEO_LAB_RUN_WAN_CURATED_CANARY=1 to enable")
    if platform.system() != "Darwin":
        pytest.skip("curated Wan canary requires Darwin")
    revisions_value = os.environ.get("VIDEO_LAB_WAN_COMPONENT_REVISIONS_JSON")
    if revisions_value is None:
        pytest.skip("missing exact component revisions: VIDEO_LAB_WAN_COMPONENT_REVISIONS_JSON")
    try:
        revisions = json.loads(revisions_value)
        roots, bindings, reverify = _secure_component_roots(tmp_path, revisions, stack)
    except (TypeError, ValueError):
        pytest.fail("curated Wan root authority is invalid")
    if shutil.disk_usage(tmp_path).free < 8 * 1024 * 1024 * 1024:
        pytest.skip("insufficient artifact capacity: require 8 GiB free under tmp_path")
    try:
        import torch
    except ImportError:
        pytest.skip("missing installed production adapter dependency: torch")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        pytest.skip("missing qualified production device: available MPS")
    return roots, revisions, bindings, reverify


def test_wan_curated_roots_are_retained_for_concrete_orchestrator(tmp_path: Path) -> None:
    with ExitStack() as stack:
        roots, revisions, root_bindings, reverify_roots = _preflight(tmp_path, stack)
        adapter_json = os.environ.get("VIDEO_LAB_WAN_ADAPTER_PATHS_JSON")
        targets_json = os.environ.get("VIDEO_LAB_WAN_TARGETS_JSON")
        if not all((adapter_json, targets_json)):
            pytest.skip(
                "missing installed production adapter/root: exact adapter paths, targets, "
                "and component revisions"
            )
        try:
            adapter_paths = json.loads(adapter_json)
            targets = tuple(json.loads(targets_json))
        except (TypeError, ValueError):
            pytest.fail("curated Wan canary exact local JSON inputs are invalid")
        from video_lab.host_canaries.wan_curated import (
            WanProductionCanaryExecutionV2,
            run_wan_curated_host_canary,
        )

        assert callable(run_wan_curated_host_canary)
        assert WanProductionCanaryExecutionV2.__module__ == (
            "video_lab.host_canaries.wan_curated"
        )
        assert set(roots) == set(_ROOT_ENVIRONMENTS.values())
        assert set(revisions) == set(_ROOT_ENVIRONMENTS)
        assert set(root_bindings) == set(_ROOT_ENVIRONMENTS)
        assert isinstance(adapter_paths, dict) and adapter_paths
        assert targets
        reverify_roots()

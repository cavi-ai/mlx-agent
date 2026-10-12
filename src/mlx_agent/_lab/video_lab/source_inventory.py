from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Protocol

from video_lab.feasibility_models import FeasibilityEvidenceError

GIB = 1024**3
COMMIT_LENGTH = 40
HASHED_EVIDENCE_SUFFIXES = (".json", ".safetensors", ".bin", ".pt", ".ckpt")

ANIMEGEN_REQUIRED_FILES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "animegen": ("high_noise.safetensors", "low_noise.safetensors", "LICENSE"),
        "wan-base": (
            "model_index.json",
            "transformer/config.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "scheduler/scheduler_config.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/spiece.model",
            "text_encoder/config.json",
            "text_encoder/model-00001-of-00003.safetensors",
            "text_encoder/model-00002-of-00003.safetensors",
            "text_encoder/model-00003-of-00003.safetensors",
            "text_encoder/model.safetensors.index.json",
        ),
        "lightning-high": ("Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",),
        "lightning-low": ("Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors",),
    }
)
ANIMEGEN_SOURCE_REPOS: Mapping[str, str] = MappingProxyType(
    {
        "animegen": "aidealab/AnimeGen-T2V",
        "wan-base": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "lightning-high": "lightx2v/Wan2.2-Lightning",
        "lightning-low": "lightx2v/Wan2.2-Lightning",
    }
)


class SourceInventoryError(FeasibilityEvidenceError):
    """Raised when immutable source evidence cannot be established."""


class ModelInfo(Protocol):
    def __call__(self, *, repo_id: str, revision: str, files_metadata: bool) -> object: ...


class SnapshotDownload(Protocol):
    def __call__(self, **kwargs: object) -> str: ...


class DiskUsage(Protocol):
    @property
    def free(self) -> int: ...


@dataclass(frozen=True)
class SourceSpec:
    component: str
    repo_id: str
    revision: str
    required_files: tuple[str, ...]
    allow_custom_code: bool = False

    def __init__(
        self,
        component: str,
        repo_id: str,
        revision: str,
        required_files: Sequence[str],
        allow_custom_code: bool = False,
    ) -> None:
        object.__setattr__(self, "component", _required(component, "component"))
        object.__setattr__(self, "repo_id", _required(repo_id, "repo_id"))
        object.__setattr__(self, "revision", _required(revision, "revision"))
        object.__setattr__(self, "required_files", tuple(required_files))
        object.__setattr__(self, "allow_custom_code", allow_custom_code)


@dataclass(frozen=True)
class SourceFile:
    relative_path: str
    size: int
    upstream_sha256: str | None = None
    local_sha256: str | None = None

    def export_evidence(self) -> Mapping[str, object]:
        evidence: dict[str, object] = {
            "relative_path": self.relative_path,
            "size": self.size,
        }
        if self.upstream_sha256:
            evidence["upstream_sha256"] = self.upstream_sha256
        if self.local_sha256:
            evidence["local_sha256"] = self.local_sha256
        return evidence


@dataclass(frozen=True)
class ResolvedSource:
    component: str
    repo_id: str
    requested_revision: str
    revision: str
    license: str
    files: tuple[SourceFile, ...]

    def export_evidence(self) -> Mapping[str, object]:
        return {
            "component": self.component,
            "repo_id": self.repo_id,
            "requested_revision": self.requested_revision,
            "revision": self.revision,
            "license": self.license,
            "files": tuple(item.export_evidence() for item in self.files),
        }


@dataclass(frozen=True)
class StoragePreflight:
    expected_bytes: int
    safety_margin_bytes: int
    required_bytes: int
    free_bytes: int

    def export_evidence(self) -> Mapping[str, int]:
        return {
            "expected_bytes": self.expected_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "required_bytes": self.required_bytes,
            "free_bytes": self.free_bytes,
        }


@dataclass(frozen=True)
class SourceInventory:
    sources: tuple[ResolvedSource, ...]
    expected_bytes: int
    _client: SourceInventoryClient = field(repr=False, compare=False)

    def export_evidence(self) -> Mapping[str, object]:
        return {
            "expected_bytes": self.expected_bytes,
            "sources": tuple(source.export_evidence() for source in self.sources),
        }

    def preflight(self, disk_usage: Callable[[Path], DiskUsage] | None = None) -> StoragePreflight:
        return self._client.preflight(self, disk_usage)

    def download(self) -> SourceInventory:
        return self._client.download(self)


def canonical_source_digest(source: ResolvedSource) -> str:
    files: list[dict[str, object]] = []
    for item in sorted(source.files, key=lambda value: value.relative_path):
        checksum = item.local_sha256 or item.upstream_sha256
        if checksum is None:
            raise SourceInventoryError(
                f"source file lacks a verified checksum: {item.relative_path}"
            )
        files.append({"path": item.relative_path, "size": item.size, "sha256": checksum})
    payload = {
        "repo_id": source.repo_id,
        "revision": source.revision,
        "files": files,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class SourceInventoryClient:
    def __init__(
        self,
        cache_root: Path,
        model_info: ModelInfo,
        snapshot_download: SnapshotDownload,
    ) -> None:
        self._cache_root = Path(cache_root)
        self._model_info = model_info
        self._snapshot_download = snapshot_download

    def resolve(self, specs: Iterable[SourceSpec]) -> SourceInventory:
        requested = tuple(specs)
        components = [spec.component for spec in requested]
        if len(components) != len(set(components)):
            raise SourceInventoryError("duplicate logical component")

        sources = tuple(self._resolve_one(spec) for spec in requested)
        return SourceInventory(
            sources=sources,
            expected_bytes=sum(item.size for source in sources for item in source.files),
            _client=self,
        )

    def preflight(
        self,
        inventory: SourceInventory,
        disk_usage: Callable[[Path], DiskUsage] | None = None,
    ) -> StoragePreflight:
        self._prepare_cache_root()
        try:
            usage = (disk_usage or _disk_usage)(self._cache_root)
        except OSError as exc:
            raise SourceInventoryError("storage preflight filesystem access failed") from exc
        free = usage.free
        margin = max(20 * GIB, (inventory.expected_bytes + 4) // 5)
        required = inventory.expected_bytes + margin
        result = StoragePreflight(inventory.expected_bytes, margin, required, free)
        if free < required:
            raise SourceInventoryError(
                f"insufficient free space: requires {required} bytes, found {free} bytes"
            )
        return result

    def _prepare_cache_root(self) -> None:
        root = self._cache_root
        try:
            if root.is_symlink():
                raise SourceInventoryError("cache root must not be a symlink")
            if root.exists():
                if not root.is_dir():
                    raise SourceInventoryError("cache root must be a directory")
                return
            for parent in root.parents:
                if parent.exists():
                    if parent.is_symlink():
                        raise SourceInventoryError("cache parent must not be a symlink")
                    if not parent.is_dir():
                        raise SourceInventoryError("cache parent must be a directory")
                    break
            root.mkdir(parents=True, exist_ok=False)
            if root.is_symlink() or not root.is_dir():
                raise SourceInventoryError("cache root creation was unsafe")
        except SourceInventoryError:
            raise
        except (FileExistsError, OSError) as exc:
            raise SourceInventoryError("cache root setup failed safely") from exc

    def download(self, inventory: SourceInventory) -> SourceInventory:
        for source in inventory.sources:
            self._assert_frozen_revision(source)
        completed: list[ResolvedSource] = []
        for source in inventory.sources:
            try:
                target = Path(
                    self._snapshot_download(
                        repo_id=source.repo_id,
                        revision=source.revision,
                        cache_dir=str(self._cache_root),
                        allow_patterns=tuple(item.relative_path for item in source.files),
                        local_files_only=False,
                        resume_download=True,
                    )
                )
            except OSError as exc:
                raise SourceInventoryError(
                    f"download filesystem access failed for {source.repo_id}"
                ) from exc
            self._assert_frozen_revision(source)
            completed.append(replace(source, files=self._verify_download(target, source.files)))
        sources = tuple(completed)
        return SourceInventory(sources, inventory.expected_bytes, self)

    def verify_cached(self, inventory: SourceInventory) -> SourceInventory:
        """Reverify a frozen imported inventory without metadata or acquisition calls."""
        completed: list[ResolvedSource] = []
        for source in inventory.sources:
            try:
                target = Path(
                    self._snapshot_download(
                        repo_id=source.repo_id,
                        revision=source.revision,
                        cache_dir=str(self._cache_root),
                        allow_patterns=tuple(item.relative_path for item in source.files),
                        local_files_only=True,
                    )
                )
            except OSError as exc:
                raise SourceInventoryError(
                    f"cached snapshot unavailable for {source.repo_id}"
                ) from exc
            completed.append(replace(source, files=self._verify_download(target, source.files)))
        sources = tuple(completed)
        return SourceInventory(sources, inventory.expected_bytes, self)

    def refresh_cached(
        self, inventory: SourceInventory, specs: Sequence[SourceSpec]
    ) -> SourceInventory:
        """Add newly required files from the same frozen local snapshots, then reverify."""
        if len(inventory.sources) != len(specs):
            raise SourceInventoryError("cached inventory source set does not match specification")
        if all(
            tuple(item.relative_path for item in source.files) == spec.required_files
            for source, spec in zip(inventory.sources, specs, strict=True)
        ):
            return inventory
        refreshed: list[ResolvedSource] = []
        cache_root = self._cache_root.resolve()
        for source, spec in zip(inventory.sources, specs, strict=True):
            if (
                source.component != spec.component
                or source.repo_id != spec.repo_id
                or source.requested_revision != spec.revision
                or source.revision != spec.revision
            ):
                raise SourceInventoryError(f"cached inventory identity mismatch: {spec.component}")
            target = Path(
                self._snapshot_download(
                    repo_id=source.repo_id,
                    revision=source.revision,
                    cache_dir=str(self._cache_root),
                    allow_patterns=spec.required_files,
                    local_files_only=True,
                )
            )
            existing = {item.relative_path: item for item in source.files}
            files: list[SourceFile] = []
            for name in spec.required_files:
                item = existing.get(name)
                if item is None:
                    path = (target / name).resolve(strict=True)
                    if not path.is_relative_to(cache_root) or not path.is_file():
                        raise SourceInventoryError(f"cached file escapes cache: {name}")
                    item = SourceFile(name, path.stat().st_size, local_sha256=_sha256(path))
                files.append(item)
            verified = self._verify_download(target, tuple(files))
            refreshed.append(replace(source, files=verified))
        sources = tuple(refreshed)
        expected_bytes = sum(item.size for source in sources for item in source.files)
        return SourceInventory(sources, expected_bytes, self)

    def _resolve_one(self, spec: SourceSpec) -> ResolvedSource:
        if _is_mutable_revision(spec.revision):
            raise SourceInventoryError(f"source {spec.repo_id} requires an immutable revision")
        response = self._model_info(
            repo_id=spec.repo_id, revision=spec.revision, files_metadata=True
        )
        revision = str(getattr(response, "sha", ""))
        if not _is_commit(revision):
            raise SourceInventoryError(
                f"source {spec.repo_id} did not resolve to a 40-character commit"
            )
        license_name = _license_from(response)
        if not license_name:
            raise SourceInventoryError(f"source {spec.repo_id} is missing license metadata")

        siblings = tuple(getattr(response, "siblings", ()) or ())
        available: dict[str, object] = {}
        for sibling in siblings:
            name = str(getattr(sibling, "rfilename", ""))
            _validate_relative_filename(name)
            if name.lower().endswith(".py") and not spec.allow_custom_code:
                raise SourceInventoryError(f"unexpected custom code in {spec.repo_id}: {name}")
            if name in available:
                raise SourceInventoryError(f"duplicate filename in {spec.repo_id}: {name}")
            available[name] = sibling

        missing = [name for name in spec.required_files if name not in available]
        if missing:
            raise SourceInventoryError(
                f"missing required file in {spec.repo_id}: {', '.join(missing)}"
            )
        files = tuple(_source_file(name, available[name]) for name in spec.required_files)
        return ResolvedSource(
            spec.component, spec.repo_id, spec.revision, revision, license_name, files
        )

    def _verify_download(
        self, target: Path, expected: tuple[SourceFile, ...]
    ) -> tuple[SourceFile, ...]:
        try:
            cache_root = self._cache_root.resolve()
        except OSError as exc:
            raise SourceInventoryError("filesystem verification failed for cache root") from exc
        verified: list[SourceFile] = []
        for item in expected:
            path = target / item.relative_path
            try:
                is_symlink = path.is_symlink()
                resolved = path.resolve(strict=True)
            except FileNotFoundError as exc:
                if is_symlink:
                    raise SourceInventoryError(
                        f"download symlink escapes cache: {item.relative_path}"
                    ) from exc
                raise SourceInventoryError(f"download missing file: {item.relative_path}") from exc
            except RuntimeError as exc:
                raise SourceInventoryError(
                    f"download symlink resolution failed: {item.relative_path}"
                ) from exc
            except OSError as exc:
                raise SourceInventoryError(
                    f"filesystem verification failed for {item.relative_path}"
                ) from exc
            try:
                if not resolved.is_relative_to(cache_root) or not resolved.is_file():
                    raise SourceInventoryError(
                        f"download symlink escapes cache: {item.relative_path}"
                    )
                if resolved.stat().st_size != item.size:
                    raise SourceInventoryError(f"download size mismatch: {item.relative_path}")
                digest = _sha256(resolved)
            except SourceInventoryError:
                raise
            except OSError as exc:
                raise SourceInventoryError(
                    f"filesystem verification failed for {item.relative_path}"
                ) from exc
            if item.upstream_sha256 and digest != item.upstream_sha256:
                raise SourceInventoryError(f"download hash mismatch: {item.relative_path}")
            if item.local_sha256 and digest != item.local_sha256:
                raise SourceInventoryError(f"cached hash mismatch: {item.relative_path}")
            local_hash = digest if _needs_local_hash(item.relative_path) else None
            verified.append(replace(item, local_sha256=local_hash))
        return tuple(verified)

    def _assert_frozen_revision(self, source: ResolvedSource) -> None:
        current = self._model_info(
            repo_id=source.repo_id,
            revision=source.requested_revision,
            files_metadata=True,
        )
        if getattr(current, "sha", None) != source.revision:
            raise SourceInventoryError(
                f"resolved commit changed for {source.repo_id} between inventory and download"
            )


def inventory_from_evidence(
    evidence: Mapping[str, object], client: SourceInventoryClient
) -> SourceInventory:
    """Reconstruct a frozen inventory while keeping cache/client state operational-only."""
    raw_sources = evidence.get("sources")
    expected_bytes = evidence.get("expected_bytes")
    if (
        not isinstance(raw_sources, Sequence)
        or isinstance(raw_sources, str | bytes | bytearray)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        raise SourceInventoryError("invalid persisted source inventory")
    sources: list[ResolvedSource] = []
    components: set[str] = set()
    for raw_source in raw_sources:
        if not isinstance(raw_source, Mapping):
            raise SourceInventoryError("invalid persisted source identity")
        component = _required(str(raw_source.get("component", "")), "component")
        repo_id = _required(str(raw_source.get("repo_id", "")), "repo_id")
        requested = str(raw_source.get("requested_revision", ""))
        revision = str(raw_source.get("revision", ""))
        license_name = _required(str(raw_source.get("license", "")), "license")
        if component in components or not _is_commit(requested) or not _is_commit(revision):
            raise SourceInventoryError("persisted source identity is not immutable")
        components.add(component)
        raw_files = raw_source.get("files")
        if not isinstance(raw_files, Sequence) or isinstance(raw_files, str | bytes | bytearray):
            raise SourceInventoryError("invalid persisted source files")
        files: list[SourceFile] = []
        names: set[str] = set()
        for raw_file in raw_files:
            if not isinstance(raw_file, Mapping):
                raise SourceInventoryError("invalid persisted source file")
            name = str(raw_file.get("relative_path", ""))
            _validate_relative_filename(name)
            size = raw_file.get("size")
            upstream = raw_file.get("upstream_sha256")
            local = raw_file.get("local_sha256")
            if name in names or not isinstance(size, int) or size < 0:
                raise SourceInventoryError("invalid persisted source file")
            for digest in (upstream, local):
                if digest is not None and (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                ):
                    raise SourceInventoryError("invalid persisted source checksum")
            names.add(name)
            files.append(SourceFile(name, size, upstream, local))
        sources.append(
            ResolvedSource(component, repo_id, requested, revision, license_name, tuple(files))
        )
    if sum(item.size for source in sources for item in source.files) != expected_bytes:
        raise SourceInventoryError("persisted source inventory byte total does not match")
    return SourceInventory(tuple(sources), expected_bytes, client)


def _required(value: str, label: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise SourceInventoryError(f"{label} must not be blank")
    return stripped


def _is_mutable_revision(revision: str) -> bool:
    return revision.lower() in {"main", "master", "latest", "refs/heads/main"}


def _is_commit(value: str) -> bool:
    return len(value) == COMMIT_LENGTH and all(char in "0123456789abcdef" for char in value)


def _validate_relative_filename(name: str) -> None:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise SourceInventoryError(f"unsafe filename: {name}")


def _license_from(response: object) -> str | None:
    card_data = getattr(response, "card_data", None)
    if isinstance(card_data, Mapping):
        value = card_data.get("license")
    else:
        value = getattr(card_data, "license", None)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _source_file(name: str, sibling: object) -> SourceFile:
    size = getattr(sibling, "size", None)
    if not isinstance(size, int) or size < 0:
        raise SourceInventoryError(f"missing byte size for {name}")
    lfs = getattr(sibling, "lfs", None)
    sha = getattr(lfs, "sha256", None) if lfs is not None else None
    if sha is not None and (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(character not in "0123456789abcdef" for character in sha)
    ):
        raise SourceInventoryError(f"invalid upstream hash for {name}")
    return SourceFile(name, size, sha)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _needs_local_hash(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(HASHED_EVIDENCE_SUFFIXES) or Path(lowered).name.startswith("license")


def _disk_usage(path: Path) -> DiskUsage:
    return shutil.disk_usage(path)

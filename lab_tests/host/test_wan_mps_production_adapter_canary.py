from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Self

import pytest
from pydantic import (
    ConfigDict,
    StrictBool,
    ValidationError,
    field_validator,
    model_validator,
)

from video_lab.domain import FrozenModel
from video_lab.host_canaries.wan_qualification import (
    WAN_PRODUCTION_AUTHORITY_NAMES,
    WanQualificationManifestV1,
    WanQualificationSearchV1,
    discover_wan_production_authorities,
    load_wan_qualification_manifest,
)
from video_lab.local_authority import (
    RetainedDirectoryAuthority,
    RetainedSqliteAuthority,
)
from video_lab.trainers.wan_production_models import (
    WanBranchInventoryRole,
    WanComponentInventoryV1,
    WanTransformerBranch,
)

if TYPE_CHECKING:
    from video_lab.database import Database

_HEAVY_MODULES = ("torch", "peft", "diffusers", "transformers", "safetensors.torch")
_REQUIRED_DEPENDENCIES = ("diffusers", "peft", "safetensors", "torch", "transformers")
_MODEL_FILES = {
    "model_index.json": b"{}",
    "scheduler/scheduler_config.json": b"{}",
    "text_encoder/config.json": b"{}",
    "tokenizer/tokenizer.json": b"{}",
    "transformer/config.json": b"{}",
    "weights/high.safetensors": b"high",
    "weights/low.safetensors": b"low",
    "vae/config.json": b"{}",
}
_ADAPTER_PATHS = {
    "wan_snapshot": ".",
    "scheduler": "scheduler",
    "text_encoder": "text_encoder",
    "tokenizer": "tokenizer",
    "transformer_config": "transformer/config.json",
    "high_transformer": "weights/high.safetensors",
    "low_transformer": "weights/low.safetensors",
    "vae": "vae",
}

pytestmark = pytest.mark.wan_mps_host

_REVISION = re.compile(r"[0-9a-f]{40}")
_ROOT_ENVIRONMENTS = (
    "VIDEO_LAB_WAN_ARTIFACT_DATABASE",
    "VIDEO_LAB_WAN_ARTIFACT_ROOT",
    "VIDEO_LAB_WAN_MODEL_ROOT",
    "VIDEO_LAB_WAN_CACHE_ROOT",
)
_RECEIPT_ENVIRONMENTS = (
    "VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID",
    "VIDEO_LAB_WAN_COMPILATION_OPERATION_ID",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_QUALIFICATION_MANIFEST_PATH = (
    _REPOSITORY_ROOT / ".superpowers/evidence/wan-production-qualification.json"
)
_COMBINED_EVIDENCE_NAME = "wan-mps-production-two-run-evidence.json"


_ACQUISITION_BRANCHES: tuple[WanTransformerBranch, WanTransformerBranch] = (
    "high_noise",
    "low_noise",
)
_ACQUISITION_ROLES: frozenset[WanBranchInventoryRole] = frozenset(
    (
        "high_transformer",
        "low_transformer",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer_config",
        "vae",
    )
)
_ALL_HIGH_ACQUISITION_ROLES: tuple[WanBranchInventoryRole, ...] = (
    "high_transformer",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer_config",
    "vae",
)
_ALL_LOW_ACQUISITION_ROLES: tuple[WanBranchInventoryRole, ...] = (
    "low_transformer",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer_config",
    "vae",
)


class _StrictFrozenModel(FrozenModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
        hide_input_in_errors=True,
    )


class _AcquisitionEntry(_StrictFrozenModel):
    branch: WanTransformerBranch
    missing_roles: tuple[WanBranchInventoryRole, ...]

    @field_validator("missing_roles", mode="before")
    @classmethod
    def validate_exact_logical_roles(cls, value: object) -> object:
        if (
            type(value) is not tuple
            or any(type(role) is not str or role not in _ACQUISITION_ROLES for role in value)
        ):
            raise ValueError("wan_mps_acquisition_role_invalid")
        return value

    @field_validator("missing_roles")
    @classmethod
    def validate_canonical_roles(
        cls, value: tuple[WanBranchInventoryRole, ...]
    ) -> tuple[WanBranchInventoryRole, ...]:
        if not value or value != tuple(sorted(set(value))):
            raise ValueError("wan_mps_acquisition_role_invalid")
        return value

    @model_validator(mode="after")
    def validate_branch_roles(self) -> Self:
        forbidden_role = (
            "low_transformer" if self.branch == "high_noise" else "high_transformer"
        )
        if forbidden_role in self.missing_roles:
            raise ValueError("wan_mps_acquisition_branch_role_invalid")
        return self

    def to_payload(self) -> dict[str, object]:
        return {"branch": self.branch, "missing_roles": list(self.missing_roles)}


class _AcquisitionManifest(_StrictFrozenModel):
    schema_version: Literal["wan-mps-production-acquisition.v1"] = (
        "wan-mps-production-acquisition.v1"
    )
    branches: tuple[_AcquisitionEntry, ...]

    @field_validator("branches", mode="before")
    @classmethod
    def validate_exact_entries(cls, value: object) -> object:
        if type(value) is not tuple or any(type(item) is not _AcquisitionEntry for item in value):
            raise ValueError("wan_mps_acquisition_entries_invalid")
        return value

    @model_validator(mode="after")
    def validate_canonical_entries(self) -> Self:
        observed = tuple(item.branch for item in self.branches)
        if observed not in (("high_noise",), ("low_noise",), _ACQUISITION_BRANCHES):
            raise ValueError("wan_mps_acquisition_entries_invalid")
        return self

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "branches": deepcopy(self.branches) if deep else self.branches,
        }
        if update is not None:
            payload.update(update)
        return self.__class__.model_validate(payload)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "branches": [entry.to_payload() for entry in self.branches],
        }


class _InventoryOnlyResult(_StrictFrozenModel):
    available: StrictBool
    inventories: tuple[WanComponentInventoryV1, WanComponentInventoryV1] | tuple[()]
    acquisition_manifest: _AcquisitionManifest | None

    @field_validator("available", mode="before")
    @classmethod
    def validate_exact_available(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("wan_mps_inventory_result_invalid")
        return value

    @field_validator("inventories", mode="before")
    @classmethod
    def validate_exact_inventories(cls, value: object) -> object:
        if type(value) is not tuple or len(value) not in (0, 2):
            raise ValueError("wan_mps_inventory_result_invalid")
        if any(type(item) is not WanComponentInventoryV1 for item in value):
            raise ValueError("wan_mps_inventory_result_invalid")
        try:
            validated = tuple(
                WanComponentInventoryV1.model_validate(
                    item.model_dump(round_trip=True), strict=True
                )
                for item in value
            )
        except ValidationError:
            raise ValueError("wan_mps_inventory_result_invalid") from None
        if len(validated) == 2 and dict(validated[0].component_manifests)[
            "transformer"
        ] == dict(validated[1].component_manifests)["transformer"]:
            raise ValueError("wan_mps_inventory_result_invalid")
        return validated

    @field_validator("acquisition_manifest", mode="before")
    @classmethod
    def validate_exact_manifest(cls, value: object) -> object:
        if value is not None and type(value) is not _AcquisitionManifest:
            raise ValueError("wan_mps_inventory_result_invalid")
        return value

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.available:
            valid = len(self.inventories) == 2 and self.acquisition_manifest is None
        else:
            valid = (
                self.inventories == ()
                and type(self.acquisition_manifest) is _AcquisitionManifest
            )
        if not valid:
            raise ValueError("wan_mps_inventory_result_invalid")
        return self

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        payload: dict[str, object] = {
            "available": self.available,
            "inventories": deepcopy(self.inventories) if deep else self.inventories,
            "acquisition_manifest": (
                deepcopy(self.acquisition_manifest) if deep else self.acquisition_manifest
            ),
        }
        if update is not None:
            payload.update(update)
        return self.__class__.model_validate(payload)


@dataclass(frozen=True, slots=True)
class _QualifiedProductionCanary:
    database_path: Path
    artifact_root: Path
    model_root: Path
    cache_root: Path
    model_revision: str
    cache_revision: str
    receipt_id: str
    operation_id: str
    receipt: Any
    adapter_paths: dict[str, Path]
    dependency_versions: tuple[tuple[str, str], ...]
    tolerance: Any
    inventories: tuple[Any, Any]
    tensor_envelope: tuple[Any, ...]
    required_artifact_bytes: int
    memory_release_bound_bytes: int
    run_roots: tuple[_RunRoots, _RunRoots]
    run_inputs: tuple[tuple[Path, Path], tuple[Path, Path]]
    run_databases: tuple[Database, Database]
    accept_run_database_mutation: Callable[[int], None]
    reverify: Callable[[], None]


@dataclass(frozen=True, slots=True)
class _ReceiptAuthority:
    receipt: Any
    run_inputs: tuple[tuple[Path, Path], tuple[Path, Path]]
    run_databases: tuple[Database, Database]
    accept_run_database_mutation: Callable[[int], None]
    reverify: Callable[[], None]


@dataclass(frozen=True, slots=True)
class _DatabaseFileSnapshot:
    device: int
    inode: int
    size: int
    mtime_ns: int
    digest: str


@dataclass(slots=True)
class _RetainedDatabaseAuthority:
    path: Path
    descriptor: int
    expected: _DatabaseFileSnapshot

    def _snapshot(self) -> _DatabaseFileSnapshot:
        try:
            opened_before = os.fstat(self.descriptor)
            current_before = os.stat(self.path, follow_symlinks=False)
            digest = _hash_descriptor(self.descriptor, opened_before.st_size)
            opened_after = os.fstat(self.descriptor)
            current_after = os.stat(self.path, follow_symlinks=False)
        except OSError:
            raise ValueError("wan_mps_production_canary_database_changed") from None
        opened_before_identity = (
            opened_before.st_dev,
            opened_before.st_ino,
            opened_before.st_size,
            opened_before.st_mtime_ns,
        )
        current_before_identity = (
            current_before.st_dev,
            current_before.st_ino,
            current_before.st_size,
            current_before.st_mtime_ns,
        )
        opened_identity = (
            opened_after.st_dev,
            opened_after.st_ino,
            opened_after.st_size,
            opened_after.st_mtime_ns,
        )
        current_identity = (
            current_after.st_dev,
            current_after.st_ino,
            current_after.st_size,
            current_after.st_mtime_ns,
        )
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or not stat.S_ISREG(current_before.st_mode)
            or opened_before_identity != opened_identity
            or current_before_identity != current_identity
            or opened_identity != current_identity
        ):
            raise ValueError("wan_mps_production_canary_database_changed")
        return _DatabaseFileSnapshot(*opened_identity, digest)

    def reverify(self) -> None:
        if self._snapshot() != self.expected:
            raise ValueError("wan_mps_production_canary_database_changed")

    def accept_authorized_mutation(self) -> None:
        observed = self._snapshot()
        if (observed.device, observed.inode) != (
            self.expected.device,
            self.expected.inode,
        ):
            raise ValueError("wan_mps_production_canary_database_changed")
        self.expected = observed


@dataclass(frozen=True, slots=True)
class _RetainedReadonlyDatabase:
    authority: RetainedSqliteAuthority

    @property
    def path(self) -> Path:
        return self.authority.path

    def connect(self) -> Any:
        return self.authority.connect_readonly()


@dataclass(frozen=True, slots=True)
class _StaticRows:
    rows: list[tuple[int, str, str]]

    def fetchall(self) -> list[tuple[int, str, str]]:
        return self.rows


@dataclass(frozen=True, slots=True)
class _AuthorityConnectionView:
    connection: sqlite3.Connection
    path: Path

    def execute(self, statement: str, *arguments: object) -> Any:
        if statement.strip().lower() == "pragma database_list":
            return _StaticRows([(0, "main", str(self.path))])
        return self.connection.execute(statement, *arguments)


@dataclass(frozen=True, slots=True)
class _RunRoots:
    database_root: Path
    artifact_root: Path
    snapshot_root: Path
    checkpoint_root: Path
    scratch_root: Path
    evidence_root: Path

    @property
    def database_path(self) -> Path:
        return self.database_root / "lab.sqlite3"

    def all_roots(self) -> tuple[Path, ...]:
        return (
            self.database_root,
            self.artifact_root,
            self.snapshot_root,
            self.checkpoint_root,
            self.scratch_root,
            self.evidence_root,
        )


@dataclass
class _QualificationExecutionSentinels:
    events: list[tuple[str, tuple[tuple[str, str | None], ...]]]
    micro_calls: int = 0
    production_calls: int = 0

    def run_micro(self) -> None:
        self.micro_calls += 1
        self.events.append(
            (
                "micro",
                tuple((name, os.environ.get(name)) for name in WAN_PRODUCTION_AUTHORITY_NAMES),
            )
        )

    def run_production(self) -> None:
        self.production_calls += 1
        self.events.append(
            (
                "production",
                tuple((name, os.environ.get(name)) for name in WAN_PRODUCTION_AUTHORITY_NAMES),
            )
        )


def _fail(code: str) -> NoReturn:
    pytest.fail(code, pytrace=False)


def run_qualified_production_canary(
    manifest: WanQualificationManifestV1,
    *,
    micro_canary: Callable[[], None],
    production_canary: Callable[[], None],
) -> None:
    manifest = WanQualificationManifestV1.model_validate(manifest, strict=True)
    if manifest.status != "qualified":
        _fail(f"wan_mps_production_canary_inventory_{manifest.status}")

    absent = object()
    previous: dict[str, str | object] = {
        name: os.environ.get(name, absent) for name in WAN_PRODUCTION_AUTHORITY_NAMES
    }
    try:
        for name, value in manifest.bindings:
            os.environ[name] = value
        micro_canary()
        production_canary()
    finally:
        for name, value in previous.items():
            if value is absent:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)


def _load_canonical_qualification_manifest() -> WanQualificationManifestV1:
    try:
        return load_wan_qualification_manifest(_QUALIFICATION_MANIFEST_PATH)
    except ValueError as error:
        _fail(str(error))


def _validate_production_opt_in_environment() -> None:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        _fail("wan_mps_production_canary_fallback_forbidden")


def _run_real_mps_micro_canary() -> None:
    environment = os.environ.copy()
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    environment["VIDEO_LAB_RUN_WAN_MPS_MICRO_CANARY"] = "1"
    environment.pop("VIDEO_LAB_RUN_WAN_MPS_PRODUCTION_CANARY", None)
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-m",
            "wan_mps_micro_host",
            "lab_tests/host/test_wan_mps_branch_micro_canary.py",
        ),
        cwd=_REPOSITORY_ROOT,
        env=environment,
        check=False,
    )
    if result.returncode != 0:
        _fail("wan_mps_production_canary_micro_failed")


def _parse_adapter_paths(value: str) -> dict[str, Path]:
    try:
        raw = json.loads(value)
        if type(raw) is not dict or set(raw) != set(_ADAPTER_PATHS):
            raise ValueError
        parsed = {name: Path(raw[name]) for name in sorted(raw)}
        for name, path in parsed.items():
            if name == "wan_snapshot" and path == Path("."):
                continue
            if (
                path.is_absolute()
                or "\\" in str(path)
                or not path.parts
                or any(part in {"..", ""} for part in path.parts)
            ):
                raise ValueError
        return parsed
    except (TypeError, ValueError, json.JSONDecodeError):
        _fail("wan_mps_production_canary_adapter_input_invalid")


def _parse_dependency_versions(value: str) -> tuple[tuple[str, str], ...]:
    try:
        raw = json.loads(value)
        if type(raw) is not list or any(type(item) is not list or len(item) != 2 for item in raw):
            raise ValueError
        result = tuple((item[0], item[1]) for item in raw)
        if (
            not result
            or any(
                type(name) is not str
                or type(version) is not str
                or not name
                or not version
                for name, version in result
            )
            or result != tuple(sorted(set(result)))
            or tuple(name for name, _ in result) != _REQUIRED_DEPENDENCIES
        ):
            raise ValueError
        return result
    except (TypeError, ValueError, json.JSONDecodeError):
        _fail("wan_mps_production_canary_dependency_input_invalid")


def _parse_positive_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        _fail("wan_mps_production_canary_capacity_input_missing")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        _fail("wan_mps_production_canary_capacity_input_invalid")
    if str(parsed) != value or parsed <= 0:
        _fail("wan_mps_production_canary_capacity_input_invalid")
    return parsed


def _model_request(
    *,
    model_root: Path,
    model_revision: str,
    adapter_paths: dict[str, Path],
    snapshot_parent: Path,
    branch: str,
) -> Any:
    from video_lab.trainers.wan_branch_loader import WanBranchLoadRequest
    from video_lab.trainers.wan_factory import WanPinnedLocalRoot, root_manifest_digest

    return WanBranchLoadRequest(
        model_root=WanPinnedLocalRoot(
            model_root, model_revision, root_manifest_digest(model_root)
        ),
        snapshot_parent=snapshot_parent,
        branch=branch,
        scheduler=adapter_paths["scheduler"],
        text_encoder=adapter_paths["text_encoder"],
        tokenizer=adapter_paths["tokenizer"],
        transformer_config=adapter_paths["transformer_config"],
        high_transformer=adapter_paths["high_transformer"],
        low_transformer=adapter_paths["low_transformer"],
        vae=adapter_paths["vae"],
    )


def inventory_wan_mps_production_inputs(scratch_root: Path) -> _InventoryOnlyResult:
    """Inventory explicit local model authorities without importing ML libraries."""
    inputs = {
        name: os.environ.get(name)
        for name in (
            "VIDEO_LAB_WAN_ADAPTER_PATHS_JSON",
            "VIDEO_LAB_WAN_CACHE_REVISION",
            "VIDEO_LAB_WAN_CACHE_ROOT",
            "VIDEO_LAB_WAN_MODEL_REVISION",
            "VIDEO_LAB_WAN_MODEL_ROOT",
        )
    }
    if any(not value for value in inputs.values()):
        return _InventoryOnlyResult(
            available=False,
            inventories=(),
            acquisition_manifest=_AcquisitionManifest(
                branches=(
                    _AcquisitionEntry(
                        branch="high_noise", missing_roles=_ALL_HIGH_ACQUISITION_ROLES
                    ),
                    _AcquisitionEntry(
                        branch="low_noise", missing_roles=_ALL_LOW_ACQUISITION_ROLES
                    ),
                )
            ),
        )

    from video_lab.trainers.wan_branch_loader import inventory_wan_branch
    from video_lab.trainers.wan_factory import (
        WanPinnedLocalRoot,
        retain_pinned_root,
        root_manifest_digest,
    )

    model_value = inputs["VIDEO_LAB_WAN_MODEL_ROOT"]
    cache_value = inputs["VIDEO_LAB_WAN_CACHE_ROOT"]
    model_revision = inputs["VIDEO_LAB_WAN_MODEL_REVISION"]
    cache_revision = inputs["VIDEO_LAB_WAN_CACHE_REVISION"]
    adapter_value = inputs["VIDEO_LAB_WAN_ADAPTER_PATHS_JSON"]
    if any(
        _REVISION.fullmatch(value) is None
        for value in (model_revision, cache_revision)  # type: ignore[arg-type]
    ):
        _fail("wan_mps_production_canary_revision_input_invalid")
    model_root = Path(model_value)  # type: ignore[arg-type]
    cache_root = Path(cache_value)  # type: ignore[arg-type]
    if any(
        not root.is_absolute() or not root.is_dir() or root.is_symlink()
        for root in (model_root, cache_root)
    ):
        _fail("wan_mps_production_canary_root_input_invalid")
    adapter_paths = _parse_adapter_paths(adapter_value)  # type: ignore[arg-type]
    scratch_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    model = WanPinnedLocalRoot(
        model_root,
        model_revision,  # type: ignore[arg-type]
        root_manifest_digest(model_root),
    )
    cache = WanPinnedLocalRoot(
        cache_root,
        cache_revision,  # type: ignore[arg-type]
        root_manifest_digest(cache_root),
    )
    try:
        with retain_pinned_root(model) as model_lease, retain_pinned_root(cache) as cache_lease:
            requests = tuple(
                _model_request(
                    model_root=model_root,
                    model_revision=model_revision,  # type: ignore[arg-type]
                    adapter_paths=adapter_paths,
                    snapshot_parent=scratch_root,
                    branch=branch,
                )
                for branch in ("high_noise", "low_noise")
            )
            results = tuple(inventory_wan_branch(request) for request in requests)
            model_lease.reverify()
            cache_lease.reverify()
    except ValueError:
        _fail("wan_mps_production_canary_root_inventory_invalid")
    missing: tuple[_AcquisitionEntry, ...] = tuple(
        _AcquisitionEntry(branch=branch, missing_roles=result.missing_roles)
        for branch, result in zip(("high_noise", "low_noise"), results, strict=True)
        if not result.available
    )
    if missing:
        return _InventoryOnlyResult(
            available=False,
            inventories=(),
            acquisition_manifest=_AcquisitionManifest(branches=missing),
        )
    inventories = tuple(result.inventory for result in results)
    if any(item is None for item in inventories):
        _fail("wan_mps_production_canary_inventory_invalid")
    high_inventory, low_inventory = inventories
    assert high_inventory is not None
    assert low_inventory is not None
    return _InventoryOnlyResult(
        available=True,
        inventories=(high_inventory, low_inventory),
        acquisition_manifest=None,
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Publish local canary material without exposing a partial final file."""

    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    directory_descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("wan_mps_production_canary_evidence_write_failed")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
        )
        os.fsync(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
    )


def _combined_evidence_identity(
    directory_descriptor: int,
    name: str,
    payload: bytes,
    *,
    error_code: str,
) -> tuple[int, int]:
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_descriptor,
        )
        observed = os.fstat(descriptor)
        contents = bytearray()
        while len(contents) <= len(payload):
            chunk = os.pread(
                descriptor,
                len(payload) + 1 - len(contents),
                len(contents),
            )
            if not chunk:
                break
            contents.extend(chunk)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_size != len(payload)
            or bytes(contents) != payload
        ):
            raise OSError("combined evidence integrity changed")
        return observed.st_dev, observed.st_ino
    except OSError:
        raise ValueError(error_code) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _publish_combined_evidence(root: Path, payload: bytes) -> None:
    """Install once through a retained root, or compensate only this call's files."""

    error_code = "wan_mps_production_canary_evidence_authority_changed"
    authority = RetainedDirectoryAuthority.capture(root, error_code=error_code)
    temporary_name = f".{_COMBINED_EVIDENCE_NAME}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    root_descriptor = -1
    descriptor = -1
    temporary_created = False
    final_created = False
    staged_identity: tuple[int, int] | None = None
    primary_error: BaseException | None = None
    try:
        authority.reverify()
        root_descriptor = os.dup(authority.descriptor)
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=root_descriptor,
        )
        temporary_created = True
        staged = os.fstat(descriptor)
        if not stat.S_ISREG(staged.st_mode):
            raise OSError("combined evidence staging is not regular")
        staged_identity = (staged.st_dev, staged.st_ino)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("wan_mps_production_canary_evidence_write_failed")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        authority.reverify()
        try:
            os.link(
                temporary_name,
                _COMBINED_EVIDENCE_NAME,
                src_dir_fd=root_descriptor,
                dst_dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            final_created = True
        except FileExistsError:
            _combined_evidence_identity(
                root_descriptor,
                _COMBINED_EVIDENCE_NAME,
                payload,
                error_code="wan_mps_production_canary_combined_evidence_exists",
            )
        else:
            installed_identity = _combined_evidence_identity(
                root_descriptor,
                _COMBINED_EVIDENCE_NAME,
                payload,
                error_code="wan_mps_production_canary_combined_evidence_integrity_failed",
            )
            if installed_identity != staged_identity:
                raise ValueError(
                    "wan_mps_production_canary_combined_evidence_integrity_failed"
                )
        os.unlink(temporary_name, dir_fd=root_descriptor)
        temporary_created = False
        os.fsync(root_descriptor)
        authority.reverify()
        final_identity = _combined_evidence_identity(
            root_descriptor,
            _COMBINED_EVIDENCE_NAME,
            payload,
            error_code="wan_mps_production_canary_combined_evidence_integrity_failed",
        )
        if final_created and final_identity != staged_identity:
            raise ValueError(
                "wan_mps_production_canary_combined_evidence_integrity_failed"
            )
        authority.reverify()
    except BaseException as error:
        primary_error = error
        cleanup_errors: list[BaseException] = []
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            descriptor = -1
        removed = False
        if final_created and root_descriptor >= 0:
            try:
                observed = os.stat(
                    _COMBINED_EVIDENCE_NAME,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
                if (observed.st_dev, observed.st_ino) != staged_identity:
                    raise OSError("combined evidence ownership changed")
                os.unlink(_COMBINED_EVIDENCE_NAME, dir_fd=root_descriptor)
                removed = True
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if temporary_created and root_descriptor >= 0:
            try:
                observed = os.stat(
                    temporary_name,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
                if staged_identity is not None and (
                    observed.st_dev,
                    observed.st_ino,
                ) != staged_identity:
                    raise OSError("combined evidence staging ownership changed")
                os.unlink(temporary_name, dir_fd=root_descriptor)
                removed = True
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if removed and root_descriptor >= 0:
            try:
                os.fsync(root_descriptor)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise ValueError(
                "wan_mps_production_canary_combined_evidence_rollback_failed"
            ) from BaseExceptionGroup(
                "wan_mps_production_canary_combined_evidence_rollback_errors",
                tuple(
                    ValueError(
                        "wan_mps_production_canary_combined_evidence_rollback_operation_failed"
                    )
                    for _cleanup_error in cleanup_errors
                ),
            )
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if root_descriptor >= 0:
            os.close(root_descriptor)
        try:
            authority.close()
        except ValueError:
            if primary_error is None:
                raise


def _hash_descriptor(descriptor: int, expected_size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < expected_size:
        chunk = os.pread(descriptor, min(1024 * 1024, expected_size - offset), offset)
        if not chunk:
            raise ValueError("wan_mps_production_canary_database_changed")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, expected_size):
        raise ValueError("wan_mps_production_canary_database_changed")
    return digest.hexdigest()


def _retain_database_authority(
    path: Path, stack: ExitStack
) -> _RetainedDatabaseAuthority:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError:
        raise ValueError("wan_mps_production_canary_database_changed") from None
    stack.callback(os.close, descriptor)
    opened = os.fstat(descriptor)
    placeholder = _DatabaseFileSnapshot(
        opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, ""
    )
    authority = _RetainedDatabaseAuthority(path, descriptor, placeholder)
    authority.expected = authority._snapshot()
    authority.reverify()
    return authority


def _require_run_capacity(
    run_roots: tuple[_RunRoots, _RunRoots], required_artifact_bytes: int
) -> None:
    for roots in run_roots:
        if any(
            shutil.disk_usage(root).free < required_artifact_bytes
            for root in (roots.artifact_root, roots.checkpoint_root)
        ):
            _fail("wan_mps_production_canary_artifact_capacity_insufficient")


def _copy_verified_artifact(handle: Any, destination_root: Path) -> None:
    target = destination_root.joinpath(*handle.relative_parts)
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    payload = os.pread(handle.descriptor, handle.artifact.byte_size, 0)
    if (
        len(payload) != handle.artifact.byte_size
        or hashlib.sha256(payload).hexdigest() != handle.artifact.sha256
    ):
        _fail("wan_mps_production_canary_receipt_artifact_invalid")
    _atomic_write_bytes(target, payload)


def _copy_verified_input(handle: Any, destination: Path) -> None:
    payload = os.pread(handle.descriptor, handle.artifact.byte_size, 0)
    if (
        len(payload) != handle.artifact.byte_size
        or hashlib.sha256(payload).hexdigest() != handle.artifact.sha256
    ):
        _fail("wan_mps_production_canary_receipt_artifact_invalid")
    _atomic_write_bytes(destination, payload)


def _reverify_production_database(authority: RetainedSqliteAuthority) -> None:
    try:
        authority.reverify()
    except ValueError as error:
        if str(error) == "wan_canary_database_authority_changed":
            raise ValueError("wan_mps_production_canary_database_changed") from None
        raise


def _prepare_receipt_authority(
    *,
    database_path: Path,
    artifact_root: Path,
    receipt_id: str,
    operation_id: str,
    run_roots: tuple[_RunRoots, _RunRoots],
    stack: ExitStack,
) -> _ReceiptAuthority:
    """Retain the public durable receipt and materialize verified per-run inputs."""

    from video_lab.artifacts import ArtifactStore
    from video_lab.database import Database
    from video_lab.dataset_receipts import DatasetCompilationRepository

    try:
        source_database_authority = RetainedSqliteAuthority.capture(database_path)
        stack.callback(source_database_authority.close)
        database = _RetainedReadonlyDatabase(source_database_authority)
        artifacts = ArtifactStore(database, artifact_root)
        repository = DatasetCompilationRepository(database)
        _reverify_production_database(source_database_authority)
        receipt = repository.receipt(receipt_id)
        _reverify_production_database(source_database_authority)
        if (
            receipt.receipt_id != receipt_id
            or receipt.operation_id != operation_id
            or receipt.compiler_id != "wan_curated_v1"
            or tuple(output.role for output in receipt.outputs)
            != ("batch", "manifest", "provenance")
        ):
            raise ValueError
        run_database_authorities: list[_RetainedDatabaseAuthority] = []
        run_clone_authorities: list[RetainedSqliteAuthority | None] = []
        run_read_databases: list[_RetainedReadonlyDatabase] = []
        run_databases: list[Database] = []
        for roots in run_roots:
            _reverify_production_database(source_database_authority)
            destination = RetainedDirectoryAuthority.capture(
                roots.database_root,
                error_code="wan_mps_production_canary_database_changed",
            )
            stack.callback(destination.close)
            clone_authority = source_database_authority.clone_to(
                destination, roots.database_path.name
            )
            stack.callback(clone_authority.close)
            run_clone_authorities.append(clone_authority)
            run_read_databases.append(_RetainedReadonlyDatabase(clone_authority))
            _reverify_production_database(source_database_authority)
            run_database_authority = _retain_database_authority(
                roots.database_path, stack
            )
            run_database_authorities.append(run_database_authority)
            run_databases.append(Database(roots.database_path))
        source_connection = _AuthorityConnectionView(
            stack.enter_context(database.connect()), database.path
        )
        bindings = (receipt.source, receipt.recipe, *receipt.outputs)
        source_handles = tuple(
            stack.enter_context(
                artifacts.open_verified_with_connection(
                    source_connection,
                    binding.artifact_id,
                    sha256=binding.sha256,
                    byte_size=binding.byte_size,
                    media_type=binding.media_type,
                )
            )
            for binding in bindings
        )

        retained_inputs: list[tuple[Path, int, tuple[int, int, int, int], str]] = []
        run_inputs: list[tuple[Path, Path]] = []
        for roots, run_database, run_database_authority in zip(
            run_roots,
            run_read_databases,
            run_database_authorities,
            strict=True,
        ):
            for handle in source_handles:
                _copy_verified_artifact(handle, roots.artifact_root)
            run_artifacts = ArtifactStore(run_database, roots.artifact_root)
            run_database_authority.reverify()
            run_receipt = DatasetCompilationRepository(run_database).receipt(receipt_id)
            run_database_authority.reverify()
            if run_receipt != receipt:
                raise ValueError
            compiled_root = roots.scratch_root / "compiled-receipt"
            compiled_root.mkdir(mode=0o700)
            copied_outputs: dict[str, Path] = {}
            with run_database.connect() as retained_connection:
                run_connection = _AuthorityConnectionView(
                    retained_connection, run_database.path
                )
                with ExitStack() as run_handles:
                    for binding in bindings:
                        run_handle = run_handles.enter_context(
                            run_artifacts.open_verified_with_connection(
                                run_connection,
                                binding.artifact_id,
                                sha256=binding.sha256,
                                byte_size=binding.byte_size,
                                media_type=binding.media_type,
                            )
                        )
                        role = getattr(binding, "role", None)
                        if role in {"batch", "manifest"}:
                            target = compiled_root / binding.artifact_id
                            _copy_verified_input(run_handle, target)
                            target.chmod(0o400)
                            descriptor = os.open(
                                target,
                                os.O_RDONLY
                                | getattr(os, "O_CLOEXEC", 0)
                                | getattr(os, "O_NOFOLLOW", 0),
                            )
                            stack.callback(os.close, descriptor)
                            info = os.fstat(descriptor)
                            identity = (
                                info.st_dev,
                                info.st_ino,
                                info.st_size,
                                info.st_mtime_ns,
                            )
                            retained_inputs.append(
                                (
                                    target,
                                    descriptor,
                                    identity,
                                    hashlib.sha256(
                                        os.pread(descriptor, info.st_size, 0)
                                    ).hexdigest(),
                                )
                            )
                            copied_outputs[role] = target
            compiled_root.chmod(0o500)
            run_inputs.append((copied_outputs["manifest"], copied_outputs["batch"]))

        def reverify() -> None:
            _reverify_production_database(source_database_authority)
            if repository.receipt(receipt_id) != receipt:
                raise ValueError("wan_mps_production_canary_receipt_changed")
            _reverify_production_database(source_database_authority)
            for handle in source_handles:
                artifacts.reverify_open_handle(source_connection, handle)
            for authority in run_clone_authorities:
                if authority is not None:
                    authority.reverify()
            for authority in run_database_authorities:
                authority.reverify()
            for path, descriptor, identity, digest in retained_inputs:
                current = path.stat(follow_symlinks=False)
                observed = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
                if (
                    observed != identity
                    or (
                        os.fstat(descriptor).st_dev,
                        os.fstat(descriptor).st_ino,
                        os.fstat(descriptor).st_size,
                        os.fstat(descriptor).st_mtime_ns,
                    )
                    != identity
                    or hashlib.sha256(os.pread(descriptor, identity[2], 0)).hexdigest()
                    != digest
                ):
                    raise ValueError("wan_mps_production_canary_runtime_input_changed")

        def accept_run_database_mutation(run_number: int) -> None:
            if run_number not in (1, 2):
                raise ValueError("wan_mps_production_canary_database_changed")
            _reverify_production_database(source_database_authority)
            clone_authority = run_clone_authorities[run_number - 1]
            if clone_authority is not None:
                clone_authority.close()
                run_clone_authorities[run_number - 1] = None
            run_database_authorities[run_number - 1].accept_authorized_mutation()
            for index, authority in enumerate(run_database_authorities):
                if index != run_number - 1:
                    authority.reverify()
            _reverify_production_database(source_database_authority)

        reverify()
        return _ReceiptAuthority(
            receipt=receipt,
            run_inputs=(run_inputs[0], run_inputs[1]),
            run_databases=(run_databases[0], run_databases[1]),
            accept_run_database_mutation=accept_run_database_mutation,
            reverify=reverify,
        )
    except Exception:
        _fail("wan_mps_production_canary_receipt_invalid")


@contextmanager
def _production_preflight(
    scratch_root: Path,
    *,
    importer: Callable[[str], object] = importlib.import_module,
    system: Callable[[], str] = platform.system,
    package_version: Callable[[str], str] = importlib.metadata.version,
) -> Iterator[_QualifiedProductionCanary]:
    if os.environ.get("VIDEO_LAB_RUN_WAN_MPS_PRODUCTION_CANARY") != "1":
        pytest.skip("set VIDEO_LAB_RUN_WAN_MPS_PRODUCTION_CANARY=1 to enable")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        _fail("wan_mps_production_canary_fallback_forbidden")
    if system() != "Darwin":
        _fail("wan_mps_production_canary_requires_darwin")

    dependency_value = os.environ.get("VIDEO_LAB_WAN_DEPENDENCY_VERSIONS_JSON")
    if dependency_value is None:
        _fail("wan_mps_production_canary_dependency_input_missing")
    dependencies = _parse_dependency_versions(dependency_value)
    if any(os.environ.get(name) is None for name in _ROOT_ENVIRONMENTS):
        _fail("wan_mps_production_canary_root_input_missing")
    database_path, artifact_root, model_root, cache_root = (
        Path(os.environ[name]) for name in _ROOT_ENVIRONMENTS
    )
    if (
        not database_path.is_absolute()
        or not database_path.is_file()
        or database_path.is_symlink()
        or any(
            not root.is_absolute() or not root.is_dir() or root.is_symlink()
            for root in (artifact_root, model_root, cache_root)
        )
        or artifact_root.is_relative_to(model_root)
        or model_root.is_relative_to(artifact_root)
    ):
        _fail("wan_mps_production_canary_root_input_invalid")

    model_revision = os.environ.get("VIDEO_LAB_WAN_MODEL_REVISION")
    cache_revision = os.environ.get("VIDEO_LAB_WAN_CACHE_REVISION")
    if model_revision is None or cache_revision is None:
        _fail("wan_mps_production_canary_revision_input_missing")
    if _REVISION.fullmatch(model_revision) is None or _REVISION.fullmatch(cache_revision) is None:
        _fail("wan_mps_production_canary_revision_input_invalid")

    required_artifact_bytes = _parse_positive_int(
        "VIDEO_LAB_WAN_REQUIRED_ARTIFACT_BYTES"
    )
    memory_release_bound_bytes = _parse_positive_int(
        "VIDEO_LAB_WAN_MEMORY_RELEASE_BOUND_BYTES"
    )
    envelope_value = os.environ.get("VIDEO_LAB_WAN_TENSOR_ENVELOPE_JSON")
    if envelope_value is None:
        _fail("wan_mps_production_canary_capacity_input_missing")
    tolerance_value = os.environ.get("VIDEO_LAB_WAN_TOLERANCE_JSON")
    if tolerance_value is None:
        _fail("wan_mps_production_canary_tolerance_input_missing")
    try:
        from video_lab.trainers.wan_production_models import (
            WanNumericalToleranceV1,
            WanTensorEnvelopeV2,
        )

        envelope_raw = json.loads(envelope_value)
        if type(envelope_raw) is not list:
            raise ValueError
        tensor_envelope = tuple(
            WanTensorEnvelopeV2.model_validate(
                {**item, "shape": tuple(item["shape"])}, strict=True
            )
            for item in envelope_raw
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        _fail("wan_mps_production_canary_capacity_input_invalid")
    try:
        tolerance_raw = json.loads(tolerance_value)
        tolerance = WanNumericalToleranceV1.model_validate(tolerance_raw, strict=True)
    except (TypeError, ValueError, json.JSONDecodeError):
        _fail("wan_mps_production_canary_tolerance_input_invalid")

    receipt_values = tuple(os.environ.get(name) for name in _RECEIPT_ENVIRONMENTS)
    if not all(receipt_values):
        _fail("wan_mps_production_canary_receipt_input_missing")
    adapter_value = os.environ.get("VIDEO_LAB_WAN_ADAPTER_PATHS_JSON")
    if adapter_value is None:
        _fail("wan_mps_production_canary_adapter_input_missing")
    adapter_paths = _parse_adapter_paths(adapter_value)
    run_roots = _independent_run_roots(scratch_root.parent / "production-runs")

    inventory = inventory_wan_mps_production_inputs(scratch_root / "inventory")
    if not inventory.available:
        manifest_path = scratch_root / "wan-mps-acquisition.json"
        assert inventory.acquisition_manifest is not None
        _atomic_write_json(manifest_path, inventory.acquisition_manifest.to_payload())
        _fail("wan_mps_production_canary_inventory_incomplete")

    try:
        observed_dependencies = tuple((name, package_version(name)) for name, _ in dependencies)
    except importlib.metadata.PackageNotFoundError:
        _fail("wan_mps_production_canary_dependency_unavailable")
    if observed_dependencies != dependencies:
        _fail("wan_mps_production_canary_dependency_version_stale")

    from video_lab.trainers.wan_factory import (
        WanPinnedLocalRoot,
        _is_mps_device,
        retain_pinned_root,
        root_manifest_digest,
    )

    with ExitStack() as stack:
        model_lease = stack.enter_context(
            retain_pinned_root(
                WanPinnedLocalRoot(
                    model_root, model_revision, root_manifest_digest(model_root)
                )
            )
        )
        cache_lease = stack.enter_context(
            retain_pinned_root(
                WanPinnedLocalRoot(
                    cache_root, cache_revision, root_manifest_digest(cache_root)
                )
            )
        )
        _require_run_capacity(run_roots, required_artifact_bytes)
        receipt_authority = _prepare_receipt_authority(
            database_path=database_path,
            artifact_root=artifact_root,
            receipt_id=receipt_values[0],
            operation_id=receipt_values[1],
            run_roots=run_roots,
            stack=stack,
        )
        receipt_authority.reverify()

        try:
            torch = importer("torch")
        except ImportError:
            _fail("wan_mps_production_canary_dependency_unavailable")
        try:
            mps = torch.backends.mps  # type: ignore[attr-defined]
            if not mps.is_built() or not mps.is_available():
                _fail("wan_mps_production_canary_mps_unavailable")
            probe = torch.empty(1, device="mps", dtype=torch.bfloat16)  # type: ignore[attr-defined]
            if not _is_mps_device(probe.device) or str(probe.dtype) not in {
                "bfloat16",
                "torch.bfloat16",
            }:
                _fail("wan_mps_production_canary_bf16_unavailable")
            torch.mps.synchronize()  # type: ignore[attr-defined]
            torch.mps.empty_cache()  # type: ignore[attr-defined]
        except pytest.fail.Exception:
            raise
        except BaseException:
            _fail("wan_mps_production_canary_bf16_unavailable")

        def reverify() -> None:
            model_lease.reverify()
            cache_lease.reverify()
            receipt_authority.reverify()

        qualified = _QualifiedProductionCanary(
            database_path=database_path,
            artifact_root=artifact_root,
            model_root=model_root,
            cache_root=cache_root,
            model_revision=model_revision,
            cache_revision=cache_revision,
            receipt_id=receipt_values[0],  # type: ignore[arg-type]
            operation_id=receipt_values[1],  # type: ignore[arg-type]
            receipt=receipt_authority.receipt,
            adapter_paths=adapter_paths,
            dependency_versions=dependencies,
            tolerance=tolerance,
            inventories=(inventory.inventories[0], inventory.inventories[1]),
            tensor_envelope=tensor_envelope,
            required_artifact_bytes=required_artifact_bytes,
            memory_release_bound_bytes=memory_release_bound_bytes,
            run_roots=run_roots,
            run_inputs=receipt_authority.run_inputs,
            run_databases=receipt_authority.run_databases,
            accept_run_database_mutation=(
                receipt_authority.accept_run_database_mutation
            ),
            reverify=reverify,
        )
        reverify()
        try:
            yield qualified
        finally:
            reverify()


def _independent_run_roots(tmp_path: Path) -> tuple[_RunRoots, _RunRoots]:
    runs: list[_RunRoots] = []
    for name in ("run-primary", "run-independent"):
        root = tmp_path / name
        roots = _RunRoots(
            database_root=root / "database",
            artifact_root=root / "artifacts",
            snapshot_root=root / "snapshots",
            checkpoint_root=root / "checkpoints",
            scratch_root=root / "scratch",
            evidence_root=root / "evidence",
        )
        for path in roots.all_roots():
            path.mkdir(parents=True, mode=0o700, exist_ok=False)
        runs.append(roots)
    return runs[0], runs[1]


def _file_source(name: str, path: Path) -> Any:
    from video_lab.trainers.wan_mps import WanSourceFile

    payload = path.read_bytes()
    info = path.stat(follow_symlinks=False)
    return WanSourceFile(
        name=name,
        path=path,
        digest="sha256:" + hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        device=info.st_dev,
        inode=info.st_ino,
    )


def _request_for(
    qualified: _QualifiedProductionCanary, receipt: Any, *, run_number: int
) -> Any:
    from video_lab.domain import canonical_json
    from video_lab.trainers.wan_production_models import (
        WanBranchRecipeV1,
        WanProductionCanaryRequestV2,
    )

    receipt_digest = "sha256:" + hashlib.sha256(
        canonical_json(receipt.model_dump(mode="json")).encode()
    ).hexdigest()
    batch = next(item for item in receipt.outputs if item.role == "batch")
    recipes = tuple(
        WanBranchRecipeV1(
            schema_version="wan-mps-branch-recipe.v1",
            branch=branch,
            rank=2,
            alpha=2,
            dropout=0.0,
            target_suffixes=("to_k", "to_out.0", "to_q", "to_v"),
            optimizer_algorithm="adamw",
            optimizer_parameters=(("learning_rate", 0.0001),),
            gradient_clip_norm=1.0,
            seed=7,
            tolerance=qualified.tolerance,
        )
        for branch in ("high_noise", "low_noise")
    )
    return WanProductionCanaryRequestV2(
        schema_version="wan-mps-production-canary-request.v2",
        recipes=recipes,
        inventories=qualified.inventories,
        compilation_receipt_digest=receipt_digest,
        batch_digest="sha256:" + batch.sha256,
        primary_run_id=f"wan-production-{run_number}-trained",
        independent_run_id=f"wan-production-{run_number}-fresh-restore",
        requested_tensor_envelope=qualified.tensor_envelope,
        memory_release_bound_bytes=qualified.memory_release_bound_bytes,
        required_artifact_bytes=qualified.required_artifact_bytes,
        dependency_versions=qualified.dependency_versions,
        hardware_identity=(
            ("device", "mps"),
            ("machine", platform.machine()),
            ("system", platform.system()),
        ),
    )


def _run_one_production_canary(
    qualified: _QualifiedProductionCanary, roots: _RunRoots, *, run_number: int
) -> Any:
    from video_lab.artifacts import ArtifactStore
    from video_lab.host_canaries.wan_curated import (
        WanCapacityExceeded,
        WanProductionCanaryExecutionV2,
        WanProductionCanaryUnavailableV2,
        run_wan_curated_host_canary,
    )
    from video_lab.trainers.wan_branch_loader import WanBranchLoadRequest
    from video_lab.trainers.wan_factory import WanPinnedLocalRoot, root_manifest_digest
    from video_lab.trainers.wan_mps import (
        WanSourceDescriptor,
        WanTrainingSourceResolver,
    )

    qualified.reverify()
    receipt = qualified.receipt
    if receipt.receipt_id != qualified.receipt_id:
        _fail("wan_mps_production_canary_receipt_input_stale")
    if receipt.operation_id != qualified.operation_id:
        _fail("wan_mps_production_canary_operation_input_stale")
    database = qualified.run_databases[run_number - 1]
    if database.path != roots.database_path:
        _fail("wan_mps_production_canary_database_authority_stale")
    artifacts = ArtifactStore(database, roots.artifact_root)
    manifest, batch = qualified.run_inputs[run_number - 1]
    model_authority = WanPinnedLocalRoot(
        qualified.model_root,
        qualified.model_revision,
        root_manifest_digest(qualified.model_root),
    )
    branch_request = WanBranchLoadRequest(
        model_root=model_authority,
        snapshot_parent=roots.snapshot_root,
        branch="high_noise",
        scheduler=qualified.adapter_paths["scheduler"],
        text_encoder=qualified.adapter_paths["text_encoder"],
        tokenizer=qualified.adapter_paths["tokenizer"],
        transformer_config=qualified.adapter_paths["transformer_config"],
        high_transformer=qualified.adapter_paths["high_transformer"],
        low_transformer=qualified.adapter_paths["low_transformer"],
        vae=qualified.adapter_paths["vae"],
    )
    base = (
        qualified.model_root
        / qualified.adapter_paths["wan_snapshot"]
        / "model_index.json"
    )
    source_files = (
        _file_source("base_model", base),
        _file_source("batch_manifest", manifest),
        _file_source("batch_tensors", batch),
    )
    descriptors = tuple(
        WanSourceDescriptor(
            name=source.name,
            path=lambda path=source.path: path,
            expected_digest=source.digest,
            expected_size=source.size_bytes,
            expected_device=source.device,
            expected_inode=source.inode,
        )
        for source in source_files
    )
    source_lease = WanTrainingSourceResolver(descriptors).acquire()
    request = _request_for(qualified, receipt, run_number=run_number)
    execution = WanProductionCanaryExecutionV2(
        request=request,
        branch_load_request=branch_request,
        training_sources=source_lease.sources,
        training_source_lease=source_lease,
        compilation_receipt_id=receipt.receipt_id,
        checkpoint_root=roots.checkpoint_root,
        output_root=roots.artifact_root,
    )
    qualified.reverify()
    try:
        try:
            result = run_wan_curated_host_canary(
                execution=execution,
                database=database,
                artifacts=artifacts,
            )
        except WanCapacityExceeded as error:
            qualified.accept_run_database_mutation(run_number)
            qualified.reverify()
            _atomic_write_json(
                roots.evidence_root / "capacity.json",
                error.result.model_dump(mode="json"),
            )
            qualified.reverify()
            _fail("wan_mps_production_canary_capacity_observed")
    finally:
        source_lease.close()
    qualified.accept_run_database_mutation(run_number)
    qualified.reverify()
    if isinstance(result, WanProductionCanaryUnavailableV2):
        manifest_payload = {
            "schema_version": "wan-mps-production-acquisition.v1",
            "branches": [
                {
                    "branch": result.branch,
                    "missing_roles": list(result.missing_roles),
                }
            ],
        }
        _atomic_write_json(roots.evidence_root / "acquisition.json", manifest_payload)
        qualified.reverify()
        _fail("wan_mps_production_canary_inventory_incomplete")
    _atomic_write_json(
        roots.evidence_root / "receipt.json",
        result.model_dump(mode="json"),
    )
    qualified.reverify()
    return result


def _run_production_canaries(
    tmp_path: Path, qualified: _QualifiedProductionCanary
) -> bytes:
    first_roots, second_roots = qualified.run_roots
    first = _run_one_production_canary(qualified, first_roots, run_number=1)
    second = _run_one_production_canary(qualified, second_roots, run_number=2)
    first_structure = (
        first.compilation_receipt_digest,
        first.batch_digest,
        first.inventory_family_digest,
        first.recipe_family_digest,
        tuple(
            (item.branch, item.recipe_digest, item.output_digest)
            for item in first.branch_results
        ),
    )
    second_structure = (
        second.compilation_receipt_digest,
        second.batch_digest,
        second.inventory_family_digest,
        second.recipe_family_digest,
        tuple(
            (item.branch, item.recipe_digest, item.output_digest)
            for item in second.branch_results
        ),
    )
    assert second_structure == first_structure
    for reference, candidate in zip(
        first.branch_results, second.branch_results, strict=True
    ):
        assert abs(candidate.max_absolute_error - reference.max_absolute_error) <= (
            qualified.tolerance.atol
            + qualified.tolerance.rtol * abs(reference.max_absolute_error)
        )
        assert abs(candidate.max_relative_error - reference.max_relative_error) <= (
            qualified.tolerance.atol
            + qualified.tolerance.rtol * abs(reference.max_relative_error)
        )
    assert len(first.branch_probes) == len(second.branch_probes) == 2
    for reference, candidate in zip(
        first.branch_probes, second.branch_probes, strict=True
    ):
        assert (
            candidate.branch,
            candidate.recipe_digest,
            candidate.tensor_name,
            candidate.shape,
            candidate.dtype,
        ) == (
            reference.branch,
            reference.recipe_digest,
            reference.tensor_name,
            reference.shape,
            reference.dtype,
        )
        for expected, observed in zip(
            reference.values, candidate.values, strict=True
        ):
            assert abs(observed - expected) <= (
                qualified.tolerance.atol
                + qualified.tolerance.rtol * abs(expected)
            )
    evidence = {
        "schema_version": "wan-mps-production-two-run-evidence.v1",
        "first": first.model_dump(mode="json"),
        "second": second.model_dump(mode="json"),
    }
    payload = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    qualified.reverify()
    return payload


def _run_enabled_production_canary(tmp_path: Path) -> None:
    with _production_preflight(tmp_path / "preflight") as qualified:
        payload = _run_production_canaries(tmp_path, qualified)
    _publish_combined_evidence(tmp_path, payload)


def _write_model(root: Path) -> None:
    for relative, payload in _MODEL_FILES.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)


def _create_durable_receipt(
    database_path: Path,
    artifact_root: Path,
    recipe_bindings: tuple[tuple[str, str], ...],
) -> Any:
    from video_lab.artifacts import ArtifactStore
    from video_lab.database import Database
    from video_lab.dataset_compiler import BackendOutput, DatasetCompilationService
    from video_lab.dataset_receipts import DatasetCompilationRequest, InputBinding

    class ReceiptBackend:
        compiler_id = "wan_curated_v1"

        def validate_recipe(self, recipe_bytes: bytes) -> None:
            if not recipe_bytes:
                raise ValueError

        def compile_batch(
            self, source_bytes: bytes, recipe_bytes: bytes
        ) -> tuple[BackendOutput, ...]:
            if not source_bytes or not recipe_bytes:
                raise ValueError
            return (
                BackendOutput("batch", b"compiled-batch", "application/vnd.safetensors"),
                BackendOutput("manifest", b"{}", "application/json"),
                BackendOutput("provenance", b'{"bound":true}', "application/json"),
            )

    database = Database(database_path)
    database.migrate()
    artifacts = ArtifactStore(database, artifact_root)
    source = artifacts.commit(
        artifacts.stage_bytes(b'{"source":1}', "application/json"), None
    )
    recipe = artifacts.commit(
        artifacts.stage_bytes(
            json.dumps(
                {
                    "schema_version": "wan-production-qualification-recipe.v1",
                    "authority_bindings": recipe_bindings,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            "application/json",
        ),
        None,
    )
    request = DatasetCompilationRequest(
        operation_key="wan_production_qualification",
        compiler_id="wan_curated_v1",
        source=InputBinding(
            role="source_manifest",
            artifact_id=source.id,
            sha256=source.sha256,
            byte_size=source.byte_size,
            media_type=source.media_type,
        ),
        recipe=InputBinding(
            role="transform_recipe",
            artifact_id=recipe.id,
            sha256=recipe.sha256,
            byte_size=recipe.byte_size,
            media_type=recipe.media_type,
        ),
        output_roles=("batch", "manifest", "provenance"),
    )
    backend = ReceiptBackend()
    return DatasetCompilationService(
        database, artifacts, {backend.compiler_id: backend}
    ).compile(request)


def _qualified_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    model_root = tmp_path / "model"
    cache_root = tmp_path / "cache"
    artifact_root = tmp_path / "input-artifacts"
    _write_model(model_root)
    cache_root.mkdir()
    (cache_root / "inventory.json").write_text("{}", encoding="utf-8")
    database = tmp_path / "input.sqlite3"
    from video_lab.trainers.wan_factory import root_manifest_digest

    model_revision = hashlib.sha256(root_manifest_digest(model_root).encode()).hexdigest()[:40]
    cache_revision = hashlib.sha256(root_manifest_digest(cache_root).encode()).hexdigest()[:40]
    scientific_values = {
        "VIDEO_LAB_WAN_MODEL_REVISION": model_revision,
        "VIDEO_LAB_WAN_CACHE_REVISION": cache_revision,
        "VIDEO_LAB_WAN_ADAPTER_PATHS_JSON": json.dumps(_ADAPTER_PATHS),
        "VIDEO_LAB_WAN_DEPENDENCY_VERSIONS_JSON": json.dumps(
            [[name, importlib.metadata.version(name)] for name in _REQUIRED_DEPENDENCIES]
        ),
        "VIDEO_LAB_WAN_TOLERANCE_JSON": json.dumps({"atol": 1e-5, "rtol": 1e-4}),
        "VIDEO_LAB_WAN_TENSOR_ENVELOPE_JSON": json.dumps(
            [
                {
                    "schema_version": "wan-mps-tensor-envelope.v2",
                    "name": "activation",
                    "role": "activation",
                    "shape": [1, 2, 2],
                    "dtype": "bfloat16",
                    "copies": 1,
                },
                {
                    "schema_version": "wan-mps-tensor-envelope.v2",
                    "name": "adapter",
                    "role": "parameter",
                    "shape": [2, 2],
                    "dtype": "bfloat16",
                    "copies": 1,
                },
                {
                    "schema_version": "wan-mps-tensor-envelope.v2",
                    "name": "optimizer",
                    "role": "optimizer",
                    "shape": [2, 2],
                    "dtype": "float32",
                    "copies": 2,
                },
            ]
        ),
        "VIDEO_LAB_WAN_REQUIRED_ARTIFACT_BYTES": "1",
        "VIDEO_LAB_WAN_MEMORY_RELEASE_BOUND_BYTES": "8388608",
    }
    recipe_names = {
        "VIDEO_LAB_WAN_ADAPTER_PATHS_JSON",
        "VIDEO_LAB_WAN_CACHE_REVISION",
        "VIDEO_LAB_WAN_DEPENDENCY_VERSIONS_JSON",
        "VIDEO_LAB_WAN_MEMORY_RELEASE_BOUND_BYTES",
        "VIDEO_LAB_WAN_MODEL_REVISION",
        "VIDEO_LAB_WAN_REQUIRED_ARTIFACT_BYTES",
        "VIDEO_LAB_WAN_TENSOR_ENVELOPE_JSON",
        "VIDEO_LAB_WAN_TOLERANCE_JSON",
    }
    receipt = _create_durable_receipt(
        database,
        artifact_root,
        tuple(
            sorted(
                (name, value)
                for name, value in scientific_values.items()
                if name in recipe_names
            )
        ),
    )
    values = {
        "VIDEO_LAB_RUN_WAN_MPS_PRODUCTION_CANARY": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        "VIDEO_LAB_WAN_ARTIFACT_DATABASE": str(database),
        "VIDEO_LAB_WAN_ARTIFACT_ROOT": str(artifact_root),
        "VIDEO_LAB_WAN_MODEL_ROOT": str(model_root),
        "VIDEO_LAB_WAN_CACHE_ROOT": str(cache_root),
        **scientific_values,
        "VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID": receipt.receipt_id,
        "VIDEO_LAB_WAN_COMPILATION_OPERATION_ID": receipt.operation_id,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


def _blocked_qualification_manifest(
    status: Literal["missing", "ambiguous"],
) -> WanQualificationManifestV1:
    roles = WAN_PRODUCTION_AUTHORITY_NAMES
    return WanQualificationManifestV1(
        schema_version="wan-production-qualification-manifest.v1",
        status=status,
        bindings=(),
        missing_roles=roles if status == "missing" else (),
        ambiguous_roles=roles if status == "ambiguous" else (),
        candidate_digests=(),
    )


@pytest.mark.parametrize("status", ("missing", "ambiguous"))
def test_enabled_canary_stops_before_micro_or_model_run(
    status: Literal["missing", "ambiguous"],
) -> None:
    blocked = _blocked_qualification_manifest(status)
    sentinels = _QualificationExecutionSentinels(events=[])

    with pytest.raises(
        pytest.fail.Exception,
        match=f"^wan_mps_production_canary_inventory_{status}$",
    ):
        run_qualified_production_canary(
            blocked,
            micro_canary=sentinels.run_micro,
            production_canary=sentinels.run_production,
        )

    assert sentinels.micro_calls == 0
    assert sentinels.production_calls == 0
    assert sentinels.events == []


def test_qualified_manifest_exports_only_its_bindings_for_ordered_canaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {name: os.environ.get(name) for name in WAN_PRODUCTION_AUTHORITY_NAMES}
    bindings = tuple(
        (name, f"qualified-authority-{index}")
        for index, name in enumerate(WAN_PRODUCTION_AUTHORITY_NAMES)
    )
    digest = "sha256:" + "1" * 64
    qualified = WanQualificationManifestV1(
        schema_version="wan-production-qualification-manifest.v1",
        status="qualified",
        bindings=bindings,
        missing_roles=(),
        ambiguous_roles=(),
        candidate_digests=tuple((name, digest) for name in WAN_PRODUCTION_AUTHORITY_NAMES),
    )
    monkeypatch.setenv(WAN_PRODUCTION_AUTHORITY_NAMES[0], "stale-authority")
    before[WAN_PRODUCTION_AUTHORITY_NAMES[0]] = "stale-authority"
    sentinels = _QualificationExecutionSentinels(events=[])

    run_qualified_production_canary(
        qualified,
        micro_canary=sentinels.run_micro,
        production_canary=sentinels.run_production,
    )

    assert sentinels.micro_calls == 1
    assert sentinels.production_calls == 1
    assert sentinels.events == [("micro", bindings), ("production", bindings)]
    assert {
        name: os.environ.get(name) for name in WAN_PRODUCTION_AUTHORITY_NAMES
    } == before


def test_marked_entry_blocks_before_micro_or_preflight_without_qualified_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIDEO_LAB_RUN_WAN_MPS_PRODUCTION_CANARY", "1")
    events: list[str] = []
    blocked = _blocked_qualification_manifest("missing")
    monkeypatch.setitem(
        globals(), "_load_canonical_qualification_manifest", lambda: blocked
    )
    monkeypatch.setitem(
        globals(), "_run_real_mps_micro_canary", lambda: events.append("micro")
    )

    @contextmanager
    def forbidden_preflight(_path: Path) -> Iterator[_QualifiedProductionCanary]:
        events.append("preflight")
        pytest.fail("production preflight reached without qualified manifest")
        yield  # pragma: no cover

    monkeypatch.setattr(sys.modules[__name__], "_production_preflight", forbidden_preflight)

    with pytest.raises(
        pytest.fail.Exception,
        match="^wan_mps_production_canary_inventory_missing$",
    ):
        test_wan_mps_production_adapter_canary(tmp_path)

    assert events == []


def test_manifest_authority_loader_revalidates_strict_qualified_manifest(
    tmp_path: Path,
) -> None:
    from video_lab.host_canaries.wan_qualification import (
        load_wan_qualification_manifest,
    )

    bindings = tuple(
        (name, f"qualified-authority-{index}")
        for index, name in enumerate(WAN_PRODUCTION_AUTHORITY_NAMES)
    )
    digest = "sha256:" + "3" * 64
    manifest = WanQualificationManifestV1(
        schema_version="wan-production-qualification-manifest.v1",
        status="qualified",
        bindings=bindings,
        missing_roles=(),
        ambiguous_roles=(),
        candidate_digests=tuple((name, digest) for name in WAN_PRODUCTION_AUTHORITY_NAMES),
    )
    path = tmp_path / "wan-production-qualification.json"
    path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")

    assert load_wan_qualification_manifest(path) == manifest


def test_manifest_authority_loader_rejects_unavailable_without_creation(
    tmp_path: Path,
) -> None:
    from video_lab.host_canaries.wan_qualification import (
        load_wan_qualification_manifest,
    )

    path = tmp_path / "missing" / "wan-production-qualification.json"
    with pytest.raises(
        ValueError,
        match="^wan_mps_production_canary_inventory_manifest_unavailable$",
    ):
        load_wan_qualification_manifest(path)

    assert not path.parent.exists()


def test_marked_entry_rejects_unavailable_canonical_manifest_before_micro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    monkeypatch.setenv("VIDEO_LAB_RUN_WAN_MPS_PRODUCTION_CANARY", "1")
    monkeypatch.setattr(
        sys.modules[__name__],
        "_QUALIFICATION_MANIFEST_PATH",
        tmp_path / "missing" / "wan-production-qualification.json",
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "_run_real_mps_micro_canary",
        lambda: events.append("micro"),
    )

    with pytest.raises(
        pytest.fail.Exception,
        match="^wan_mps_production_canary_inventory_manifest_unavailable$",
    ):
        test_wan_mps_production_adapter_canary(tmp_path)

    assert events == []
    assert not (tmp_path / "missing").exists()


def test_marked_entry_micro_failure_prevents_production_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIDEO_LAB_RUN_WAN_MPS_PRODUCTION_CANARY", "1")
    bindings = tuple(
        (name, f"qualified-authority-{index}")
        for index, name in enumerate(WAN_PRODUCTION_AUTHORITY_NAMES)
    )
    digest = "sha256:" + "4" * 64
    manifest = WanQualificationManifestV1(
        schema_version="wan-production-qualification-manifest.v1",
        status="qualified",
        bindings=bindings,
        missing_roles=(),
        ambiguous_roles=(),
        candidate_digests=tuple((name, digest) for name in WAN_PRODUCTION_AUTHORITY_NAMES),
    )
    events: list[str] = []
    monkeypatch.setattr(
        sys.modules[__name__],
        "_load_canonical_qualification_manifest",
        lambda: manifest,
    )

    def failing_micro() -> None:
        events.append("micro")
        pytest.fail("wan_mps_production_canary_micro_failed", pytrace=False)

    monkeypatch.setattr(
        sys.modules[__name__], "_run_real_mps_micro_canary", failing_micro
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "_run_enabled_production_canary",
        lambda _path: events.append("production"),
    )

    with pytest.raises(
        pytest.fail.Exception, match="^wan_mps_production_canary_micro_failed$"
    ):
        test_wan_mps_production_adapter_canary(tmp_path)

    assert events == ["micro"]


def test_marked_entry_runs_micro_then_preflight_and_publishes_after_preflight_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIDEO_LAB_RUN_WAN_MPS_PRODUCTION_CANARY", "1")
    bindings = tuple(
        (name, f"qualified-authority-{index}")
        for index, name in enumerate(WAN_PRODUCTION_AUTHORITY_NAMES)
    )
    digest = "sha256:" + "2" * 64
    manifest = WanQualificationManifestV1(
        schema_version="wan-production-qualification-manifest.v1",
        status="qualified",
        bindings=bindings,
        missing_roles=(),
        ambiguous_roles=(),
        candidate_digests=tuple((name, digest) for name in WAN_PRODUCTION_AUTHORITY_NAMES),
    )
    events: list[object] = []
    qualified = object()
    monkeypatch.setitem(
        globals(), "_load_canonical_qualification_manifest", lambda: manifest
    )
    monkeypatch.setitem(
        globals(),
        "_run_real_mps_micro_canary",
        lambda: events.append(
            (
                "micro",
                tuple((name, os.environ.get(name)) for name in WAN_PRODUCTION_AUTHORITY_NAMES),
            )
        ),
    )

    @contextmanager
    def preflight(_path: Path) -> Iterator[object]:
        events.append("preflight-enter")
        try:
            yield qualified
        finally:
            events.append("preflight-exit")

    monkeypatch.setattr(sys.modules[__name__], "_production_preflight", preflight)
    monkeypatch.setattr(
        sys.modules[__name__],
        "_run_production_canaries",
        lambda _path, observed: events.append(("production", observed)) or b"evidence",
    )
    monkeypatch.setitem(
        globals(),
        "_publish_combined_evidence",
        lambda _path, payload: events.append(("publish", payload)),
    )

    test_wan_mps_production_adapter_canary(tmp_path)

    assert events == [
        ("micro", bindings),
        "preflight-enter",
        ("production", qualified),
        "preflight-exit",
        ("publish", b"evidence"),
    ]


def test_discovered_qualification_is_the_exact_production_canary_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _qualified_environment(tmp_path, monkeypatch)
    json_names = {
        "VIDEO_LAB_WAN_ADAPTER_PATHS_JSON",
        "VIDEO_LAB_WAN_DEPENDENCY_VERSIONS_JSON",
        "VIDEO_LAB_WAN_TENSOR_ENVELOPE_JSON",
        "VIDEO_LAB_WAN_TOLERANCE_JSON",
    }
    bindings = tuple(
        sorted(
            (
                name,
                json.dumps(
                    json.loads(values[name]), sort_keys=True, separators=(",", ":")
                )
                if name in json_names
                else values[name],
            )
            for name in WAN_PRODUCTION_AUTHORITY_NAMES
        )
    )
    scientific_names = {
        "VIDEO_LAB_WAN_ADAPTER_PATHS_JSON",
        "VIDEO_LAB_WAN_COMPILATION_OPERATION_ID",
        "VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID",
        "VIDEO_LAB_WAN_DEPENDENCY_VERSIONS_JSON",
        "VIDEO_LAB_WAN_MEMORY_RELEASE_BOUND_BYTES",
        "VIDEO_LAB_WAN_MODEL_REVISION",
        "VIDEO_LAB_WAN_CACHE_REVISION",
        "VIDEO_LAB_WAN_REQUIRED_ARTIFACT_BYTES",
        "VIDEO_LAB_WAN_TENSOR_ENVELOPE_JSON",
        "VIDEO_LAB_WAN_TOLERANCE_JSON",
    }
    candidate = {
        "schema_version": "wan-production-qualification-candidate.v1",
        "bindings": bindings,
        "accepted_compilation_evidence": tuple(
            item for item in bindings if item[0] in scientific_names
        ),
    }
    (tmp_path / "wan-production-qualification-candidate.v1.json").write_text(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    manifest = discover_wan_production_authorities(
        WanQualificationSearchV1(
            schema_version="wan-production-qualification-search.v1",
            search_roots=(tmp_path,),
            maximum_directories=20_000,
            maximum_files=200_000,
        )
    )

    assert manifest.status == "qualified"
    assert manifest.bindings == bindings


def _fake_torch(*, available: bool = True, device: str = "mps") -> Any:
    class Mps:
        @staticmethod
        def is_built() -> bool:
            return available

        @staticmethod
        def is_available() -> bool:
            return available

        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def empty_cache() -> None:
            return None

    return SimpleNamespace(
        backends=SimpleNamespace(mps=Mps()),
        mps=Mps(),
        bfloat16="torch.bfloat16",
        empty=lambda *_args, **_kwargs: SimpleNamespace(
            device=device, dtype="torch.bfloat16"
        ),
    )


def _run_single_test_in_subprocess(
    test_name: str, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    probe = """
import json
import sys

import pytest

status = pytest.main([sys.argv[1] + "::" + sys.argv[2], "-q", "-p", "no:cacheprovider"])
print("WAN_PRODUCTION_IMPORT_PROBE=" + json.dumps({
    name: name in sys.modules
    for name in ("torch", "peft", "diffusers", "transformers", "safetensors.torch")
}, sort_keys=True))
raise SystemExit(status)
"""
    return subprocess.run(
        [sys.executable, "-c", probe, str(Path(__file__).resolve()), test_name],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_heavy_modules_absent(result: subprocess.CompletedProcess[str]) -> None:
    line = next(
        item
        for item in result.stdout.splitlines()
        if item.startswith("WAN_PRODUCTION_IMPORT_PROBE=")
    )
    imported = json.loads(line.removeprefix("WAN_PRODUCTION_IMPORT_PROBE="))
    assert imported == {name: False for name in _HEAVY_MODULES}


def test_disabled_production_canary_skips_before_heavy_imports() -> None:
    environment = os.environ.copy()
    environment.pop("VIDEO_LAB_RUN_WAN_MPS_PRODUCTION_CANARY", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = _run_single_test_in_subprocess(
        "test_wan_mps_production_adapter_canary", environment
    )
    assert result.returncode == 0, result.stdout + result.stderr
    _assert_heavy_modules_absent(result)


@pytest.mark.parametrize("fallback", ("", "1", "false", "true"))
def test_enabled_production_canary_rejects_fallback_before_heavy_imports(
    fallback: str,
) -> None:
    environment = os.environ.copy()
    environment["VIDEO_LAB_RUN_WAN_MPS_PRODUCTION_CANARY"] = "1"
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = fallback
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = _run_single_test_in_subprocess(
        "test_wan_mps_production_adapter_canary", environment
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "wan_mps_production_canary_fallback_forbidden" in result.stdout
    _assert_heavy_modules_absent(result)


@pytest.mark.parametrize(
    ("removed", "replacement", "system", "available", "blocker"),
    [
        (None, None, "Linux", True, "wan_mps_production_canary_requires_darwin"),
        (
            "VIDEO_LAB_WAN_DEPENDENCY_VERSIONS_JSON",
            None,
            "Darwin",
            True,
            "wan_mps_production_canary_dependency_input_missing",
        ),
        (
            "VIDEO_LAB_WAN_MODEL_ROOT",
            None,
            "Darwin",
            True,
            "wan_mps_production_canary_root_input_missing",
        ),
        (
            "VIDEO_LAB_WAN_MODEL_REVISION",
            None,
            "Darwin",
            True,
            "wan_mps_production_canary_revision_input_missing",
        ),
        (
            "VIDEO_LAB_WAN_MODEL_REVISION",
            "A" * 40,
            "Darwin",
            True,
            "wan_mps_production_canary_revision_input_invalid",
        ),
        (
            "VIDEO_LAB_WAN_REQUIRED_ARTIFACT_BYTES",
            None,
            "Darwin",
            True,
            "wan_mps_production_canary_capacity_input_missing",
        ),
        (
            "VIDEO_LAB_WAN_TOLERANCE_JSON",
            None,
            "Darwin",
            True,
            "wan_mps_production_canary_tolerance_input_missing",
        ),
        (None, None, "Darwin", False, "wan_mps_production_canary_mps_unavailable"),
    ],
)
def test_enabled_preflight_reports_one_exact_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    removed: str | None,
    replacement: str | None,
    system: str,
    available: bool,
    blocker: str,
) -> None:
    _qualified_environment(tmp_path, monkeypatch)
    if removed is not None:
        if replacement is None:
            monkeypatch.delenv(removed)
        else:
            monkeypatch.setenv(removed, replacement)
    imported: list[str] = []

    def importer(name: str) -> object:
        imported.append(name)
        return _fake_torch(available=available)

    with pytest.raises(pytest.fail.Exception, match=f"^{blocker}$"):
        with _production_preflight(
            tmp_path / "preflight",
            importer=importer,
            system=lambda: system,
        ):
            pass
    if blocker != "wan_mps_production_canary_mps_unavailable":
        assert imported == []
    else:
        assert imported == ["torch"]


def test_enabled_preflight_accepts_indexed_mps_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualified_environment(tmp_path, monkeypatch)

    with _production_preflight(
        tmp_path / "preflight",
        importer=lambda _name: _fake_torch(device="mps:0"),
        system=lambda: "Darwin",
    ) as qualified:
        assert qualified.model_root.is_dir()


def test_capacity_and_receipt_authorities_precede_torch_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualified_environment(tmp_path, monkeypatch)
    events: list[str] = []

    def disk_usage(_root: Path) -> object:
        events.append("capacity")
        return SimpleNamespace(free=10**12)

    def qualify(**_kwargs: object) -> _ReceiptAuthority:
        events.append("receipt")
        return original_qualifier(**_kwargs)

    def importer(_name: str) -> object:
        events.append("torch")
        return _fake_torch()

    monkeypatch.setattr(shutil, "disk_usage", disk_usage)
    original_qualifier = _prepare_receipt_authority
    monkeypatch.setattr(
        sys.modules[__name__], "_prepare_receipt_authority", qualify
    )
    with _production_preflight(
        tmp_path / "preflight",
        importer=importer,
        system=lambda: "Darwin",
    ) as qualified:
        from video_lab.trainers.wan_factory import root_manifest_digest

        assert qualified.receipt.receipt_id == os.environ[
            "VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID"
        ]
        assert qualified.model_revision == hashlib.sha256(
            root_manifest_digest(qualified.model_root).encode()
        ).hexdigest()[:40]
        for roots, (manifest, batch) in zip(
            qualified.run_roots, qualified.run_inputs, strict=True
        ):
            assert manifest.is_relative_to(roots.scratch_root) and manifest.is_file()
            assert batch.is_relative_to(roots.scratch_root) and batch.is_file()
    assert events == ["capacity", "capacity", "capacity", "capacity", "receipt", "torch"]


def test_capacity_failure_precedes_receipt_and_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualified_environment(tmp_path, monkeypatch)
    monkeypatch.setenv("VIDEO_LAB_WAN_REQUIRED_ARTIFACT_BYTES", str(10**30))
    imported: list[str] = []

    with pytest.raises(
        pytest.fail.Exception,
        match="^wan_mps_production_canary_artifact_capacity_insufficient$",
    ):
        with _production_preflight(
            tmp_path / "preflight",
            importer=lambda name: imported.append(name),
            system=lambda: "Darwin",
        ):
            pass
    assert imported == []
    assert not tuple((tmp_path / "production-runs").rglob("lab.sqlite3"))


def test_local_evidence_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "evidence.json"
    observed: list[tuple[bool, bool]] = []
    replace = os.replace

    def observe_replace(source: Path, target: Path) -> None:
        observed.append((source.is_file(), target.exists()))
        replace(source, target)

    monkeypatch.setattr(os, "replace", observe_replace)
    _atomic_write_json(destination, {"status": "retained"})
    assert observed == [(True, False)]
    assert destination.read_text(encoding="utf-8") == '{"status":"retained"}'
    assert tuple(tmp_path.iterdir()) == (destination,)


def test_inventory_only_is_read_only_and_avoids_heavy_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualified_environment(tmp_path, monkeypatch)
    model = Path(os.environ["VIDEO_LAB_WAN_MODEL_ROOT"])
    cache = Path(os.environ["VIDEO_LAB_WAN_CACHE_ROOT"])
    before = {
        path.relative_to(tmp_path).as_posix(): (
            path.stat(follow_symlinks=False).st_dev,
            path.stat(follow_symlinks=False).st_ino,
            path.stat(follow_symlinks=False).st_size,
            path.stat(follow_symlinks=False).st_mtime_ns,
        )
        for root in (model, cache)
        for path in (root, *root.rglob("*"))
    }
    imported_before = {name: name in sys.modules for name in _HEAVY_MODULES}
    result = inventory_wan_mps_production_inputs(tmp_path / "inventory")
    after = {
        path.relative_to(tmp_path).as_posix(): (
            path.stat(follow_symlinks=False).st_dev,
            path.stat(follow_symlinks=False).st_ino,
            path.stat(follow_symlinks=False).st_size,
            path.stat(follow_symlinks=False).st_mtime_ns,
        )
        for root in (model, cache)
        for path in (root, *root.rglob("*"))
    }
    assert result.available
    assert len(result.inventories) == 2
    assert dict(result.inventories[0].component_manifests)["transformer"] != dict(
        result.inventories[1].component_manifests
    )["transformer"]
    assert result.acquisition_manifest is None
    assert before == after
    assert {name: name in sys.modules for name in _HEAVY_MODULES} == imported_before


def test_inventory_only_returns_canonical_path_free_acquisition_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualified_environment(tmp_path, monkeypatch)
    missing = Path(
        os.environ["VIDEO_LAB_WAN_MODEL_ROOT"], "weights", "low.safetensors"
    )
    missing.parent.chmod(0o700)
    missing.unlink()
    result = inventory_wan_mps_production_inputs(tmp_path / "inventory")
    assert not result.available
    assert result.inventories == ()
    assert result.acquisition_manifest is not None
    assert result.acquisition_manifest.to_payload() == {
        "schema_version": "wan-mps-production-acquisition.v1",
        "branches": [
            {"branch": "low_noise", "missing_roles": ["low_transformer"]}
        ],
    }
    assert str(tmp_path) not in json.dumps(
        result.acquisition_manifest.to_payload(), sort_keys=True
    )


def test_inventory_missing_explicit_inputs_returns_frozen_path_free_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "VIDEO_LAB_WAN_MODEL_ROOT",
        "VIDEO_LAB_WAN_CACHE_ROOT",
        "VIDEO_LAB_WAN_MODEL_REVISION",
        "VIDEO_LAB_WAN_CACHE_REVISION",
        "VIDEO_LAB_WAN_ADAPTER_PATHS_JSON",
    ):
        monkeypatch.delenv(name, raising=False)
    imported_before = {name: name in sys.modules for name in _HEAVY_MODULES}

    result = inventory_wan_mps_production_inputs(tmp_path / "inventory")

    assert not result.available
    assert result.inventories == ()
    assert result.acquisition_manifest is not None
    assert result.acquisition_manifest.to_payload() == {
        "schema_version": "wan-mps-production-acquisition.v1",
        "branches": [
            {
                "branch": "high_noise",
                "missing_roles": [
                    "high_transformer",
                    "scheduler",
                    "text_encoder",
                    "tokenizer",
                    "transformer_config",
                    "vae",
                ],
            },
            {
                "branch": "low_noise",
                "missing_roles": [
                    "low_transformer",
                    "scheduler",
                    "text_encoder",
                    "tokenizer",
                    "transformer_config",
                    "vae",
                ],
            }
        ],
    }
    with pytest.raises(ValidationError, match="frozen"):
        result.available = True  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        result.acquisition_manifest.branches = ()  # type: ignore[misc,union-attr]
    payload = json.dumps(result.acquisition_manifest.to_payload(), sort_keys=True)
    assert str(tmp_path) not in payload
    assert not (tmp_path / "inventory").exists()
    assert {name: name in sys.modules for name in _HEAVY_MODULES} == imported_before


@pytest.mark.parametrize(
    "role",
    (
        "/private/model",
        "../model",
        "nested/model",
        "nested\\model",
        ".",
        "",
        "arbitrary_component",
    ),
)
def test_acquisition_entries_reject_nonlogical_or_path_bearing_roles(role: str) -> None:
    with pytest.raises(ValidationError, match="wan_mps_acquisition_role_invalid") as captured:
        _AcquisitionEntry(branch="high_noise", missing_roles=(role,))
    assert captured.value.errors(
        include_url=False, include_context=False, include_input=False
    ) == [
        {
            "type": "value_error",
            "loc": ("missing_roles",),
            "msg": "Value error, wan_mps_acquisition_role_invalid",
        }
    ]


def _component_inventory(suffix: str) -> WanComponentInventoryV1:
    return WanComponentInventoryV1(
        schema_version="wan-mps-component-inventory.v1",
        component_manifests=tuple(
            (name, "sha256:" + suffix * 64)
            for name in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae")
        ),
    )


def _valid_inventory_result() -> _InventoryOnlyResult:
    return _InventoryOnlyResult(
        available=True,
        inventories=(_component_inventory("1"), _component_inventory("2")),
        acquisition_manifest=None,
    )


@pytest.mark.parametrize(
    "payload",
    (
        {
            "available": 1,
            "inventories": (),
            "acquisition_manifest": object(),
        },
        {
            "available": True,
            "inventories": (object(), object()),
            "acquisition_manifest": None,
        },
        {
            "available": True,
            "inventories": [{}, {}],
            "acquisition_manifest": None,
        },
        {
            "available": True,
            "inventories": ({}, {}),
            "acquisition_manifest": None,
        },
        {
            "available": False,
            "inventories": (),
            "acquisition_manifest": object(),
        },
        {
            "available": False,
            "inventories": (),
            "acquisition_manifest": {"branches": ()},
        },
        {
            "available": False,
            "inventories": (),
            "acquisition_manifest": None,
            "extra": "forbidden",
        },
    ),
)
def test_inventory_result_rejects_coercion_wrong_types_and_extras(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _InventoryOnlyResult(**payload)  # type: ignore[arg-type]


def test_inventory_result_revalidates_copy_and_rejects_mutation() -> None:
    result = _valid_inventory_result()
    assert result.model_copy() == result
    assert result.model_copy(deep=True) == result
    with pytest.raises(ValidationError):
        result.model_copy(update={"available": False})
    with pytest.raises(ValidationError):
        result.model_copy(update={"acquisition_manifest": object()})
    with pytest.raises(ValidationError, match="frozen"):
        result.inventories = ()  # type: ignore[misc]


def test_inventory_result_rejects_subclass_and_constructed_invalid_authorities() -> None:
    class _InventorySubclass(WanComponentInventoryV1):
        pass

    subclass = _InventorySubclass.model_validate(
        _component_inventory("1").model_dump(round_trip=True), strict=True
    )
    malformed = WanComponentInventoryV1.model_construct(
        schema_version="wan-mps-component-inventory.v1",
        component_manifests=(),
    )
    for authority in (subclass, malformed):
        with pytest.raises(ValidationError, match="wan_mps_inventory_result_invalid"):
            _InventoryOnlyResult(
                available=True,
                inventories=(authority, _component_inventory("2")),
                acquisition_manifest=None,
            )


def test_inventory_result_rejects_reused_cross_branch_inventory() -> None:
    inventory = _component_inventory("1")
    with pytest.raises(ValidationError, match="wan_mps_inventory_result_invalid"):
        _InventoryOnlyResult(
            available=True,
            inventories=(inventory, inventory),
            acquisition_manifest=None,
        )


@pytest.mark.parametrize(
    "entry",
    (
        {"branch": "high_noise", "missing_roles": ()},
        {"branch": "high_noise", "missing_roles": ("scheduler", "scheduler")},
        {"branch": "high_noise", "missing_roles": ("vae", "scheduler")},
        {"branch": "high_noise", "missing_roles": ["scheduler"]},
        {"branch": "high_noise", "missing_roles": (True,)},
        {"branch": True, "missing_roles": ("scheduler",)},
        {"branch": "high_noise", "missing_roles": ("low_transformer",)},
        {"branch": "low_noise", "missing_roles": ("high_transformer",)},
        {"branch": "high_noise", "missing_roles": ("scheduler",), "extra": True},
    ),
)
def test_acquisition_entry_rejects_noncanonical_and_cross_branch_roles(
    entry: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _AcquisitionEntry(**entry)  # type: ignore[arg-type]


def test_acquisition_manifest_is_exact_ordered_frozen_and_revalidated() -> None:
    high = _AcquisitionEntry(
        branch="high_noise", missing_roles=("high_transformer",)
    )
    low = _AcquisitionEntry(branch="low_noise", missing_roles=("low_transformer",))
    manifest = _AcquisitionManifest(branches=(high, low))
    assert manifest.schema_version == "wan-mps-production-acquisition.v1"
    assert manifest.model_copy() == manifest
    assert manifest.model_copy(deep=True) == manifest
    with pytest.raises(ValidationError):
        _AcquisitionManifest(
            schema_version="wan-mps-production-acquisition.v2", branches=(high, low)
        )
    with pytest.raises(ValidationError):
        _AcquisitionManifest(branches=(low, high))
    with pytest.raises(ValidationError):
        _AcquisitionManifest(branches=[high, low])  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        _AcquisitionManifest(branches=({"branch": "high_noise"},))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        _AcquisitionManifest(branches=(high, low), extra=True)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        manifest.model_copy(update={"branches": (low, high)})
    with pytest.raises(ValidationError):
        high.model_copy(update={"missing_roles": ("arbitrary_component",)})
    with pytest.raises(ValidationError, match="frozen"):
        manifest.schema_version = "wan-mps-production-acquisition.v2"  # type: ignore[misc]


def _receipt_double(
    execution: Any,
    *,
    structural_suffix: str = "a",
    error: float = 0.0,
    probe_value: float = 1.0,
) -> Any:
    from video_lab.trainers.wan_production_models import (
        WanBranchCanaryResultV1,
        WanBranchNumericalProbeV2,
        WanProductionCanaryReceiptV2,
        wan_branch_recipe_digest,
        wan_inventory_family_digest,
        wan_recipe_family_digest,
    )

    request = execution.request
    branch_results = tuple(
        WanBranchCanaryResultV1(
            schema_version="wan-mps-branch-canary-result.v1",
            branch=recipe.branch,
            recipe_digest=wan_branch_recipe_digest(recipe),
            output_digest="sha256:" + structural_suffix * 64,
            max_absolute_error=float(error),
            max_relative_error=float(error),
        )
        for recipe in request.recipes
    )
    branch_probes = tuple(
        WanBranchNumericalProbeV2(
            branch=recipe.branch,
            recipe_digest=wan_branch_recipe_digest(recipe),
            tensor_name="probe.output",
            shape=(1,),
            dtype="float32",
            values=(probe_value,),
        )
        for recipe in request.recipes
    )
    return WanProductionCanaryReceiptV2(
        schema_version="wan-mps-production-canary-receipt.v2",
        branch_results=branch_results,
        branch_probes=branch_probes,
        compilation_receipt_digest=request.compilation_receipt_digest,
        batch_digest=request.batch_digest,
        inventory_family_digest=wan_inventory_family_digest(request.inventories),
        recipe_family_digest=wan_recipe_family_digest(request.recipes),
        primary_run_id=request.primary_run_id,
        independent_run_id=request.independent_run_id,
        dependency_versions=request.dependency_versions,
        hardware_identity=request.hardware_identity,
        memory_release_bytes=(
            ("high_noise.trained", 0),
            ("high_noise.restored", 0),
            ("low_noise.trained", 0),
            ("low_noise.restored", 0),
        ),
        cleanup_state="cleaned",
    )


def _replace_same_content(path: Path, detached: Path) -> None:
    payload = path.read_bytes()
    path.rename(detached)
    path.write_bytes(payload)


def _restore_replaced_file(path: Path, detached: Path) -> None:
    path.unlink()
    detached.rename(path)


def test_source_database_same_content_replacement_blocks_public_orchestrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualified_environment(tmp_path, monkeypatch)
    calls: list[object] = []
    import video_lab.host_canaries.wan_curated as canary_module

    monkeypatch.setattr(
        canary_module,
        "run_wan_curated_host_canary",
        lambda **kwargs: calls.append(kwargs) or _receipt_double(kwargs["execution"]),
    )
    with _production_preflight(
        tmp_path / "preflight", importer=lambda _name: _fake_torch(), system=lambda: "Darwin"
    ) as qualified:
        detached = tmp_path / "detached-source.sqlite3"
        _replace_same_content(qualified.database_path, detached)
        try:
            with pytest.raises(
                ValueError, match="^wan_mps_production_canary_database_changed$"
            ):
                _run_one_production_canary(qualified, qualified.run_roots[0], run_number=1)
            assert calls == []
            assert not (qualified.run_roots[0].evidence_root / "receipt.json").exists()
        finally:
            _restore_replaced_file(qualified.database_path, detached)


def test_run_database_same_content_replacement_blocks_evidence_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualified_environment(tmp_path, monkeypatch)
    calls: list[object] = []
    detached: Path | None = None
    import video_lab.host_canaries.wan_curated as canary_module

    def replace_during_public_orchestrator(**kwargs: Any) -> Any:
        nonlocal detached
        calls.append(kwargs)
        database_path = kwargs["database"].path
        detached = database_path.with_name("detached-run.sqlite3")
        _replace_same_content(database_path, detached)
        return _receipt_double(kwargs["execution"])

    monkeypatch.setattr(
        canary_module, "run_wan_curated_host_canary", replace_during_public_orchestrator
    )
    with _production_preflight(
        tmp_path / "preflight", importer=lambda _name: _fake_torch(), system=lambda: "Darwin"
    ) as qualified:
        roots = qualified.run_roots[0]
        try:
            with pytest.raises(
                ValueError, match="^wan_mps_production_canary_database_changed$"
            ):
                _run_one_production_canary(qualified, roots, run_number=1)
            assert len(calls) == 1
            assert not (roots.evidence_root / "receipt.json").exists()
        finally:
            assert detached is not None
            _restore_replaced_file(roots.database_path, detached)


def test_sqlite_read_never_reopens_caller_path_during_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legitimate = tmp_path / "legitimate.sqlite3"
    attacker = tmp_path / "attacker.sqlite3"
    with sqlite3.connect(legitimate) as connection:
        connection.execute("CREATE TABLE authority(value TEXT NOT NULL)")
        connection.execute("INSERT INTO authority VALUES ('legitimate')")
    with sqlite3.connect(attacker) as connection:
        connection.execute("CREATE TABLE authority(value TEXT NOT NULL)")
        connection.execute("INSERT INTO authority VALUES ('attacker')")

    detached = tmp_path / "detached-legitimate.sqlite3"
    real_connect = sqlite3.connect
    swapped = False

    def swap_only_while_open(database: object, *args: object, **kwargs: object) -> Any:
        nonlocal swapped
        if str(legitimate) not in str(database) or swapped:
            return real_connect(database, *args, **kwargs)
        swapped = True
        legitimate.rename(detached)
        attacker.rename(legitimate)
        try:
            return real_connect(database, *args, **kwargs)
        finally:
            legitimate.rename(attacker)
            detached.rename(legitimate)

    authority = RetainedSqliteAuthority.capture(legitimate)
    try:
        monkeypatch.setattr(sqlite3, "connect", swap_only_while_open)
        with authority.connect_readonly() as connection:
            assert connection.execute("SELECT value FROM authority").fetchone()[0] == (
                "legitimate"
            )
    finally:
        authority.close()

    assert not swapped


def test_sqlite_rejects_attacker_shared_pager_during_open(tmp_path: Path) -> None:
    legitimate = tmp_path / "legitimate.sqlite3"
    attacker = tmp_path / "attacker.sqlite3"
    setup = sqlite3.connect(legitimate)
    try:
        setup.execute("CREATE TABLE authority(value TEXT NOT NULL)")
        setup.execute("INSERT INTO authority VALUES ('legitimate')")
        setup.commit()
    finally:
        setup.close()
    with sqlite3.connect(attacker) as connection:
        connection.execute("CREATE TABLE authority(value TEXT NOT NULL)")
        connection.execute("INSERT INTO authority VALUES ('attacker')")

    authority = RetainedSqliteAuthority.capture(legitimate)
    detached = tmp_path / "detached-legitimate.sqlite3"
    legitimate.rename(detached)
    attacker.rename(legitimate)
    shared_uri = f"file:{legitimate}?mode=ro&cache=shared"
    existing = sqlite3.connect(shared_uri, uri=True)
    try:
        assert existing.execute("SELECT value FROM authority").fetchone()[0] == "attacker"
        with pytest.raises(ValueError, match="^wan_canary_database_authority_changed$"):
            with authority.connect_readonly() as connection:
                connection.execute("SELECT value FROM authority").fetchone()
    finally:
        existing.close()
        authority.close()
        legitimate.unlink()
        detached.rename(legitimate)


def test_two_run_glue_invokes_public_orchestrator_with_six_distinct_authority_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualified_environment(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []
    import video_lab.host_canaries.wan_curated as canary_module

    def public_orchestrator(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _receipt_double(kwargs["execution"])

    monkeypatch.setattr(canary_module, "run_wan_curated_host_canary", public_orchestrator)
    with _production_preflight(
        tmp_path / "preflight", importer=lambda _name: _fake_torch(), system=lambda: "Darwin"
    ) as qualified:
        payload = _run_production_canaries(tmp_path, qualified)
        assert len(calls) == 2
        first_roots, second_roots = qualified.run_roots
        assert set(first_roots.all_roots()).isdisjoint(second_roots.all_roots())
        for call, roots in zip(calls, qualified.run_roots, strict=True):
            execution = call["execution"]
            assert call["database"].path == roots.database_path
            assert call["artifacts"].root == roots.artifact_root.resolve()
            assert execution.branch_load_request.snapshot_parent == roots.snapshot_root
            assert execution.checkpoint_root == roots.checkpoint_root
            assert execution.output_root == roots.artifact_root
            assert all(
                source.path.is_relative_to(roots.scratch_root)
                for source in execution.training_sources.files
                if source.name != "base_model"
            )
            assert (roots.evidence_root / "receipt.json").is_file()
    _publish_combined_evidence(tmp_path, payload)
    assert (tmp_path / _COMBINED_EVIDENCE_NAME).is_file()


def test_combined_evidence_deduplicates_preexisting_verified_payload(
    tmp_path: Path,
) -> None:
    payload = b'{"schema_version":"wan-mps-production-two-run-evidence.v1"}'
    _publish_combined_evidence(tmp_path, payload)
    published = tmp_path / _COMBINED_EVIDENCE_NAME
    identity = (published.stat().st_dev, published.stat().st_ino)

    _publish_combined_evidence(tmp_path, payload)

    assert published.read_bytes() == payload
    assert (published.stat().st_dev, published.stat().st_ino) == identity
    assert tuple(tmp_path.glob(f".{_COMBINED_EVIDENCE_NAME}.*.tmp")) == ()


def test_combined_evidence_concurrent_publishers_never_replace_installed_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_staged = threading.Event()
    release_first = threading.Event()
    real_fsync = os.fsync
    successes: list[str] = []
    errors: list[tuple[str, BaseException]] = []

    def pause_first_after_staging(descriptor: int) -> None:
        real_fsync(descriptor)
        if (
            threading.current_thread().name == "combined-publisher-one"
            and not first_staged.is_set()
        ):
            first_staged.set()
            if not release_first.wait(timeout=10):
                raise TimeoutError("first combined publisher was not released")

    def publish(label: str, payload: bytes) -> None:
        try:
            _publish_combined_evidence(tmp_path, payload)
        except BaseException as error:
            errors.append((label, error))
        else:
            successes.append(label)

    monkeypatch.setattr(os, "fsync", pause_first_after_staging)
    first = threading.Thread(
        target=publish,
        args=("first", b"first-payload"),
        name="combined-publisher-one",
    )
    second = threading.Thread(
        target=publish,
        args=("second", b"second-payload"),
        name="combined-publisher-two",
    )
    try:
        first.start()
        assert first_staged.wait(timeout=10)
        second.start()
        second.join(timeout=10)
        assert not second.is_alive()
        release_first.set()
        first.join(timeout=10)
        assert not first.is_alive()
        assert successes == ["second"]
        assert len(errors) == 1
        assert errors[0][0] == "first"
        assert (tmp_path / _COMBINED_EVIDENCE_NAME).read_bytes() == b"second-payload"
        assert tuple(tmp_path.glob(f".{_COMBINED_EVIDENCE_NAME}.*.tmp")) == ()
    finally:
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)


def test_combined_evidence_root_swap_at_install_rolls_back_detached_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "combined"
    root.mkdir()
    detached = tmp_path / "detached-combined"
    real_link = os.link
    real_replace = os.replace
    swapped = False

    def swap_root() -> None:
        nonlocal swapped
        if not swapped:
            root.rename(detached)
            root.mkdir()
            swapped = True

    def swap_after_link(*args: Any, **kwargs: Any) -> None:
        real_link(*args, **kwargs)
        swap_root()

    def swap_after_replace(*args: Any, **kwargs: Any) -> None:
        real_replace(*args, **kwargs)
        swap_root()

    monkeypatch.setattr(os, "link", swap_after_link)
    monkeypatch.setattr(os, "replace", swap_after_replace)
    try:
        with pytest.raises(
            ValueError,
            match="^wan_mps_production_canary_evidence_authority_changed$",
        ):
            _publish_combined_evidence(root, b"combined-payload")
        assert swapped
        assert not (root / _COMBINED_EVIDENCE_NAME).exists()
        assert not (detached / _COMBINED_EVIDENCE_NAME).exists()
        assert tuple(root.glob(f".{_COMBINED_EVIDENCE_NAME}.*.tmp")) == ()
        assert tuple(detached.glob(f".{_COMBINED_EVIDENCE_NAME}.*.tmp")) == ()
    finally:
        if swapped:
            root.rmdir()
            detached.rename(root)


@pytest.mark.parametrize("mismatch", ("structural", "numerical"))
def test_two_run_glue_rejects_cross_run_mismatch_before_combined_evidence(
    mismatch: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualified_environment(tmp_path, monkeypatch)
    calls = 0
    import video_lab.host_canaries.wan_curated as canary_module

    def public_orchestrator(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return _receipt_double(
            kwargs["execution"],
            structural_suffix="b" if calls == 2 and mismatch == "structural" else "a",
            error=1.0 if calls == 2 and mismatch == "numerical" else 0.0,
        )

    monkeypatch.setattr(canary_module, "run_wan_curated_host_canary", public_orchestrator)
    with _production_preflight(
        tmp_path / "preflight", importer=lambda _name: _fake_torch(), system=lambda: "Darwin"
    ) as qualified:
        with pytest.raises(AssertionError):
            _run_production_canaries(tmp_path, qualified)
        assert calls == 2
        assert not (tmp_path / "wan-mps-production-two-run-evidence.json").exists()


def test_two_run_glue_rejects_probe_mismatch_when_self_restore_errors_are_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualified_environment(tmp_path, monkeypatch)
    calls = 0
    import video_lab.host_canaries.wan_curated as canary_module

    def public_orchestrator(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        receipt = _receipt_double(kwargs["execution"], error=0.0)
        value = 100.0 if calls == 2 else 1.0
        object.__setattr__(
            receipt,
            "branch_probes",
            tuple(
                SimpleNamespace(
                    branch=branch,
                    recipe_digest=result.recipe_digest,
                    tensor_name="probe.output",
                    shape=(1,),
                    dtype="bfloat16",
                    values=(value,),
                )
                for branch, result in zip(
                    ("high_noise", "low_noise"), receipt.branch_results, strict=True
                )
            ),
        )
        return receipt

    monkeypatch.setattr(canary_module, "run_wan_curated_host_canary", public_orchestrator)
    with _production_preflight(
        tmp_path / "preflight", importer=lambda _name: _fake_torch(), system=lambda: "Darwin"
    ) as qualified:
        with pytest.raises(AssertionError):
            _run_production_canaries(tmp_path, qualified)

    assert calls == 2
    assert not (tmp_path / "wan-mps-production-two-run-evidence.json").exists()


def test_two_run_glue_accepts_distinct_probe_values_within_direct_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualified_environment(tmp_path, monkeypatch)
    calls = 0
    import video_lab.host_canaries.wan_curated as canary_module

    def public_orchestrator(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return _receipt_double(
            kwargs["execution"],
            error=0.0,
            probe_value=1.0001 if calls == 2 else 1.0,
        )

    monkeypatch.setattr(canary_module, "run_wan_curated_host_canary", public_orchestrator)
    with _production_preflight(
        tmp_path / "preflight", importer=lambda _name: _fake_torch(), system=lambda: "Darwin"
    ) as qualified:
        first_roots, second_roots = qualified.run_roots
        assert set(first_roots.all_roots()).isdisjoint(second_roots.all_roots())
        payload = _run_production_canaries(tmp_path, qualified)

    _publish_combined_evidence(tmp_path, payload)
    assert calls == 2
    assert (tmp_path / _COMBINED_EVIDENCE_NAME).is_file()


def test_final_reverify_detects_retained_root_inode_swap_without_combined_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualified_environment(tmp_path, monkeypatch)
    calls = 0
    import video_lab.host_canaries.wan_curated as canary_module

    def public_orchestrator(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return _receipt_double(kwargs["execution"])

    monkeypatch.setattr(canary_module, "run_wan_curated_host_canary", public_orchestrator)
    preflight = _production_preflight(
        tmp_path / "preflight", importer=lambda _name: _fake_torch(), system=lambda: "Darwin"
    )
    qualified = preflight.__enter__()
    retained_root = qualified.model_root
    retained_identity = retained_root.stat(follow_symlinks=False)
    detached_root = tmp_path / "detached-model-root"
    real_dumps = json.dumps
    swapped = False
    cleanup_error: ValueError | None = None

    def swap_after_combined_serialization(value: object, *args: Any, **kwargs: Any) -> str:
        nonlocal swapped
        payload = real_dumps(value, *args, **kwargs)
        if (
            not swapped
            and isinstance(value, dict)
            and value.get("schema_version")
            == "wan-mps-production-two-run-evidence.v1"
        ):
            retained_root.rename(detached_root)
            retained_root.mkdir(mode=0o700)
            swapped = True
        return payload

    monkeypatch.setattr(json, "dumps", swap_after_combined_serialization)
    try:
        with pytest.raises(ValueError, match="^wan_factory_snapshot_invalid$"):
            _run_production_canaries(tmp_path, qualified)

        assert calls == 2
        assert swapped
        assert retained_root.is_dir()
        assert detached_root.is_dir()
        assert not (tmp_path / _COMBINED_EVIDENCE_NAME).exists()
        assert not (retained_root / _COMBINED_EVIDENCE_NAME).exists()
        assert not (detached_root / _COMBINED_EVIDENCE_NAME).exists()
    finally:
        if swapped:
            retained_root.rmdir()
            detached_root.rename(retained_root)
        try:
            preflight.__exit__(None, None, None)
        except ValueError as error:
            cleanup_error = error

    restored_identity = retained_root.stat(follow_symlinks=False)
    assert (restored_identity.st_dev, restored_identity.st_ino) == (
        retained_identity.st_dev,
        retained_identity.st_ino,
    )
    assert not detached_root.exists()
    assert cleanup_error is not None
    assert str(cleanup_error) == "wan_factory_snapshot_invalid"


def test_two_run_glue_records_capacity_but_never_counts_it_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualified_environment(tmp_path, monkeypatch)
    calls = 0
    import video_lab.host_canaries.wan_curated as canary_module
    from video_lab.trainers.wan_production_models import (
        WanCapacityResultV2,
        wan_branch_recipe_digest,
        wan_inventory_digest,
        wan_required_tensor_bytes,
    )

    def capacity_only(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        request = kwargs["execution"].request
        raise canary_module.WanCapacityExceeded(
            WanCapacityResultV2(
                schema_version="wan-mps-capacity-result.v2",
                branch="high_noise",
                phase="step",
                device="mps",
                dtype="bfloat16",
                requested_tensor_envelope=request.requested_tensor_envelope,
                driver_allocated_before_bytes=1,
                driver_allocated_after_failure_bytes=2,
                driver_allocated_after_cleanup_bytes=0,
                recommended_maximum_bytes=1,
                required_memory_bytes=wan_required_tensor_bytes(
                    request.requested_tensor_envelope
                ),
                recipe_digest=wan_branch_recipe_digest(request.recipes[0]),
                inventory_digest=wan_inventory_digest(request.inventories[0]),
                fits=False,
                classification="recognized_mps_allocation_failure",
            )
        )

    monkeypatch.setattr(canary_module, "run_wan_curated_host_canary", capacity_only)
    with _production_preflight(
        tmp_path / "preflight", importer=lambda _name: _fake_torch(), system=lambda: "Darwin"
    ) as qualified:
        with pytest.raises(
            pytest.fail.Exception, match="^wan_mps_production_canary_capacity_observed$"
        ):
            _run_production_canaries(tmp_path, qualified)
        assert calls == 1
        assert (qualified.run_roots[0].evidence_root / "capacity.json").is_file()
        assert not (tmp_path / "wan-mps-production-two-run-evidence.json").exists()


def test_independent_run_roots_are_pairwise_distinct(tmp_path: Path) -> None:
    first, second = _independent_run_roots(tmp_path)
    first_values = set(first.all_roots())
    second_values = set(second.all_roots())
    assert len(first_values) == 6
    assert len(second_values) == 6
    assert first_values.isdisjoint(second_values)
    assert all(path.is_dir() for path in first_values | second_values)


@pytest.mark.wan_mps_production_host
def test_wan_mps_production_adapter_canary(tmp_path: Path) -> None:
    if os.environ.get("VIDEO_LAB_RUN_WAN_MPS_PRODUCTION_CANARY") != "1":
        pytest.skip("set VIDEO_LAB_RUN_WAN_MPS_PRODUCTION_CANARY=1 to enable")
    _validate_production_opt_in_environment()
    manifest = _load_canonical_qualification_manifest()
    run_qualified_production_canary(
        manifest,
        micro_canary=_run_real_mps_micro_canary,
        production_canary=lambda: _run_enabled_production_canary(tmp_path),
    )

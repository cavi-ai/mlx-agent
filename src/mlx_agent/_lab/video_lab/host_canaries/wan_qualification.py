"""Bounded, read-only discovery of explicit local Wan production authorities."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import sqlite3
import stat
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePath, PureWindowsPath
from typing import Annotated, Any, Literal, NoReturn, Self, cast

from pydantic import ConfigDict, Field, StrictInt, field_validator, model_validator

from video_lab.dataset_receipts import (
    CompilationOutput,
    DatasetCompilationReceipt,
    DatasetCompilationRequest,
    InputBinding,
)
from video_lab.domain import FrozenModel, canonical_json
from video_lab.local_authority import RetainedDirectoryAuthority, RetainedSqliteAuthority
from video_lab.trainers.wan_production_models import (
    WanNumericalToleranceV1,
    WanTensorEnvelopeV2,
)
from video_lab.training_models import Sha256Digest

WanProductionAuthorityName = Literal[
    "VIDEO_LAB_WAN_ADAPTER_PATHS_JSON",
    "VIDEO_LAB_WAN_ARTIFACT_DATABASE",
    "VIDEO_LAB_WAN_ARTIFACT_ROOT",
    "VIDEO_LAB_WAN_CACHE_REVISION",
    "VIDEO_LAB_WAN_CACHE_ROOT",
    "VIDEO_LAB_WAN_COMPILATION_OPERATION_ID",
    "VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID",
    "VIDEO_LAB_WAN_DEPENDENCY_VERSIONS_JSON",
    "VIDEO_LAB_WAN_MEMORY_RELEASE_BOUND_BYTES",
    "VIDEO_LAB_WAN_MODEL_REVISION",
    "VIDEO_LAB_WAN_MODEL_ROOT",
    "VIDEO_LAB_WAN_REQUIRED_ARTIFACT_BYTES",
    "VIDEO_LAB_WAN_TENSOR_ENVELOPE_JSON",
    "VIDEO_LAB_WAN_TOLERANCE_JSON",
]

WAN_PRODUCTION_AUTHORITY_NAMES: tuple[WanProductionAuthorityName, ...] = (
    "VIDEO_LAB_WAN_ADAPTER_PATHS_JSON",
    "VIDEO_LAB_WAN_ARTIFACT_DATABASE",
    "VIDEO_LAB_WAN_ARTIFACT_ROOT",
    "VIDEO_LAB_WAN_CACHE_REVISION",
    "VIDEO_LAB_WAN_CACHE_ROOT",
    "VIDEO_LAB_WAN_COMPILATION_OPERATION_ID",
    "VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID",
    "VIDEO_LAB_WAN_DEPENDENCY_VERSIONS_JSON",
    "VIDEO_LAB_WAN_MEMORY_RELEASE_BOUND_BYTES",
    "VIDEO_LAB_WAN_MODEL_REVISION",
    "VIDEO_LAB_WAN_MODEL_ROOT",
    "VIDEO_LAB_WAN_REQUIRED_ARTIFACT_BYTES",
    "VIDEO_LAB_WAN_TENSOR_ENVELOPE_JSON",
    "VIDEO_LAB_WAN_TOLERANCE_JSON",
)

_AUTHORITY_NAMES = frozenset(WAN_PRODUCTION_AUTHORITY_NAMES)
_RECIPE_EVIDENCE_NAMES: tuple[WanProductionAuthorityName, ...] = (
    "VIDEO_LAB_WAN_ADAPTER_PATHS_JSON",
    "VIDEO_LAB_WAN_CACHE_REVISION",
    "VIDEO_LAB_WAN_DEPENDENCY_VERSIONS_JSON",
    "VIDEO_LAB_WAN_MEMORY_RELEASE_BOUND_BYTES",
    "VIDEO_LAB_WAN_MODEL_REVISION",
    "VIDEO_LAB_WAN_REQUIRED_ARTIFACT_BYTES",
    "VIDEO_LAB_WAN_TENSOR_ENVELOPE_JSON",
    "VIDEO_LAB_WAN_TOLERANCE_JSON",
)
_SCIENTIFIC_EVIDENCE_NAMES: tuple[WanProductionAuthorityName, ...] = tuple(
    sorted(
        (
            *_RECIPE_EVIDENCE_NAMES,
            "VIDEO_LAB_WAN_COMPILATION_OPERATION_ID",
            "VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID",
        )
    )
)
_RECIPE_EVIDENCE_SET = frozenset(_RECIPE_EVIDENCE_NAMES)
_SCIENTIFIC_EVIDENCE_SET = frozenset(_SCIENTIFIC_EVIDENCE_NAMES)
_CANDIDATE_FILENAME = "wan-production-qualification-candidate.v1.json"
_MAXIMUM_CANDIDATE_BYTES = 16 * 1024 * 1024
_MAXIMUM_AUTHORITY_BYTES = 64 * 1024 * 1024 * 1024
_MAXIMUM_AUTHORITY_DEPTH = 32
_MAXIMUM_SEARCH_DIRECTORIES = 250_000
_MAXIMUM_SEARCH_FILES = 2_000_000
_MAXIMUM_ROOT_INVENTORY_DIRECTORIES = 25_000
_MAXIMUM_ROOT_INVENTORY_FILES = 100_000
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_DEPENDENCIES = ("diffusers", "peft", "safetensors", "torch", "transformers")
_ADAPTER_ROLES = (
    "high_transformer",
    "low_transformer",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer_config",
    "vae",
    "wan_snapshot",
)


class _StrictQualificationModel(FrozenModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
        hide_input_in_errors=True,
    )

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        values = {name: getattr(self, name) for name in type(self).model_fields}
        if deep:
            values = deepcopy(values)
        if update is not None:
            values.update(update)
        return type(self).model_validate(values, strict=True)


def _exact_pairs(value: object, *, label: str) -> object:
    if type(value) is not tuple:
        raise ValueError(f"{label} must be an exact tuple")
    for entry in cast(tuple[object, ...], value):
        if type(entry) is not tuple or len(cast(tuple[object, ...], entry)) != 2:
            raise ValueError(f"{label} entries must be exact pairs")
    return value


def _validate_authority_pairs(
    value: tuple[tuple[WanProductionAuthorityName, str], ...],
    *,
    allowed: frozenset[str] = _AUTHORITY_NAMES,
) -> tuple[tuple[WanProductionAuthorityName, str], ...]:
    names = tuple(name for name, _binding in value)
    if names != tuple(sorted(set(names))) or any(name not in allowed for name in names):
        raise ValueError("authority bindings must be canonical, unique, and closed")
    if any(type(binding) is not str or not binding or "\x00" in binding for _, binding in value):
        raise ValueError("authority bindings must be non-empty exact strings")
    return value


class WanQualificationSearchV1(_StrictQualificationModel):
    schema_version: Literal["wan-production-qualification-search.v1"]
    search_roots: tuple[Path, ...]
    maximum_directories: Annotated[
        StrictInt, Field(gt=0, le=_MAXIMUM_SEARCH_DIRECTORIES)
    ]
    maximum_files: Annotated[StrictInt, Field(gt=0, le=_MAXIMUM_SEARCH_FILES)]

    @field_validator("search_roots", mode="before")
    @classmethod
    def validate_exact_roots(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("search roots must be an exact tuple")
        return value

    @field_validator("search_roots")
    @classmethod
    def validate_roots(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        if not value or any(not isinstance(root, Path) or not root.is_absolute() for root in value):
            raise ValueError("search roots must be explicit absolute paths")
        texts = tuple(os.fspath(root) for root in value)
        if texts != tuple(sorted(set(texts))) or any(
            text.startswith("//") or os.path.normpath(text) != text for text in texts
        ):
            raise ValueError("search roots must be sorted and unique")
        return value

    @field_validator("maximum_directories", "maximum_files", mode="before")
    @classmethod
    def validate_exact_limits(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("search limits must be exact built-in integers")
        return value


class WanQualificationCandidateV1(_StrictQualificationModel):
    schema_version: Literal["wan-production-qualification-candidate.v1"]
    bindings: tuple[tuple[WanProductionAuthorityName, str], ...]
    accepted_compilation_evidence: tuple[
        tuple[WanProductionAuthorityName, str], ...
    ]

    @field_validator("bindings", "accepted_compilation_evidence", mode="before")
    @classmethod
    def validate_exact_containers(cls, value: object) -> object:
        return _exact_pairs(value, label="candidate authority")

    @field_validator("bindings")
    @classmethod
    def validate_bindings(
        cls, value: tuple[tuple[WanProductionAuthorityName, str], ...]
    ) -> tuple[tuple[WanProductionAuthorityName, str], ...]:
        return _validate_authority_pairs(value)

    @field_validator("accepted_compilation_evidence")
    @classmethod
    def validate_evidence(
        cls, value: tuple[tuple[WanProductionAuthorityName, str], ...]
    ) -> tuple[tuple[WanProductionAuthorityName, str], ...]:
        return _validate_authority_pairs(value, allowed=_SCIENTIFIC_EVIDENCE_SET)

    @model_validator(mode="after")
    def validate_evidence_bindings(self) -> WanQualificationCandidateV1:
        bindings = dict(self.bindings)
        if any(bindings.get(name) != value for name, value in self.accepted_compilation_evidence):
            raise ValueError("accepted compilation evidence must bind candidate values exactly")
        return self


class _WanQualificationRecipeV1(_StrictQualificationModel):
    schema_version: Literal["wan-production-qualification-recipe.v1"]
    authority_bindings: tuple[tuple[WanProductionAuthorityName, str], ...]

    @field_validator("authority_bindings", mode="before")
    @classmethod
    def validate_exact_container(cls, value: object) -> object:
        return _exact_pairs(value, label="qualification recipe authority")

    @field_validator("authority_bindings")
    @classmethod
    def validate_bindings(
        cls, value: tuple[tuple[WanProductionAuthorityName, str], ...]
    ) -> tuple[tuple[WanProductionAuthorityName, str], ...]:
        return _validate_authority_pairs(value, allowed=_RECIPE_EVIDENCE_SET)


class WanQualificationRejectionV1(_StrictQualificationModel):
    candidate_digest: Sha256Digest
    reason: Literal["malformed", "unsupported", "incomplete"]


class WanQualificationManifestV1(_StrictQualificationModel):
    schema_version: Literal["wan-production-qualification-manifest.v1"]
    status: Literal["qualified", "missing", "ambiguous"]
    bindings: tuple[tuple[WanProductionAuthorityName, str], ...]
    missing_roles: tuple[WanProductionAuthorityName, ...]
    ambiguous_roles: tuple[WanProductionAuthorityName, ...]
    candidate_digests: tuple[tuple[WanProductionAuthorityName, Sha256Digest], ...]
    rejections: tuple[WanQualificationRejectionV1, ...] = ()

    @field_validator("bindings", "candidate_digests", mode="before")
    @classmethod
    def validate_exact_pair_containers(cls, value: object) -> object:
        return _exact_pairs(value, label="manifest authority")

    @field_validator("missing_roles", "ambiguous_roles", mode="before")
    @classmethod
    def validate_exact_role_containers(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("manifest roles must be exact tuples")
        return value

    @field_validator("rejections", mode="before")
    @classmethod
    def validate_exact_rejection_container(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("manifest rejections must be an exact tuple")
        return value

    @field_validator("rejections")
    @classmethod
    def validate_rejections(
        cls, value: tuple[WanQualificationRejectionV1, ...]
    ) -> tuple[WanQualificationRejectionV1, ...]:
        keys = tuple((item.candidate_digest, item.reason) for item in value)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("manifest rejections must be canonical and unique")
        return value

    @field_validator("bindings")
    @classmethod
    def validate_bindings(
        cls, value: tuple[tuple[WanProductionAuthorityName, str], ...]
    ) -> tuple[tuple[WanProductionAuthorityName, str], ...]:
        return _validate_authority_pairs(value)

    @field_validator("candidate_digests")
    @classmethod
    def validate_candidate_digests(
        cls, value: tuple[tuple[WanProductionAuthorityName, Sha256Digest], ...]
    ) -> tuple[tuple[WanProductionAuthorityName, Sha256Digest], ...]:
        names = tuple(name for name, _digest in value)
        if names != tuple(sorted(set(names))) or any(
            name not in _AUTHORITY_NAMES for name in names
        ):
            raise ValueError("candidate digests must be canonical and closed")
        return value

    @field_validator("missing_roles", "ambiguous_roles")
    @classmethod
    def validate_roles(
        cls, value: tuple[WanProductionAuthorityName, ...]
    ) -> tuple[WanProductionAuthorityName, ...]:
        if value != tuple(sorted(set(value))) or any(
            name not in _AUTHORITY_NAMES for name in value
        ):
            raise ValueError("manifest roles must be canonical, unique, and closed")
        return value

    @model_validator(mode="after")
    def validate_status(self) -> WanQualificationManifestV1:
        binding_names = tuple(name for name, _ in self.bindings)
        digest_names = tuple(name for name, _ in self.candidate_digests)
        if self.status == "qualified":
            if (
                binding_names != WAN_PRODUCTION_AUTHORITY_NAMES
                or digest_names != WAN_PRODUCTION_AUTHORITY_NAMES
                or self.missing_roles
                or self.ambiguous_roles
            ):
                raise ValueError("qualified manifest must bind one complete authority set")
        elif self.status == "missing":
            if (
                self.bindings
                or self.candidate_digests
                or self.ambiguous_roles
                or (not self.missing_roles and not self.rejections)
            ):
                raise ValueError("missing manifest fields are mutually exclusive")
        elif (
            self.bindings
            or self.candidate_digests
            or self.missing_roles
            or not self.ambiguous_roles
        ):
            raise ValueError("ambiguous manifest fields are mutually exclusive")
        return self


def load_wan_qualification_manifest(path: Path) -> WanQualificationManifestV1:
    """Load one strict manifest through a retained, read-only local authority."""

    unavailable = "wan_mps_production_canary_inventory_manifest_unavailable"
    invalid = "wan_mps_production_canary_inventory_manifest_invalid"
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path.name in {"", ".", ".."}
        or "\x00" in path.name
    ):
        raise ValueError(invalid)
    try:
        authority = _capture_existing_directory_authority(
            path.parent, error_code=unavailable
        )
    except ValueError:
        raise ValueError(unavailable) from None
    try:
        try:
            expected = os.stat(
                path.name, dir_fd=authority.descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            raise ValueError(unavailable) from None
        payload = _read_regular_file_at(authority.descriptor, path.name, expected)
        authority.reverify()
        if payload is None:
            raise ValueError(invalid)
        decoded = _load_strict_json(payload)
        return WanQualificationManifestV1.model_validate(
            _json_arrays_to_tuples(decoded), strict=True
        )
    except ValueError as error:
        if str(error) == unavailable:
            raise
        raise ValueError(invalid) from None
    except (OSError, TypeError):
        raise ValueError(invalid) from None
    finally:
        authority.close()


def discover_wan_production_authorities(
    request: WanQualificationSearchV1,
) -> WanQualificationManifestV1:
    """Resolve a unique complete set from explicit local roots without mutation."""

    request = WanQualificationSearchV1.model_validate(request, strict=True)
    authorities = _capture_search_roots(request.search_roots)
    candidates: list[tuple[dict[WanProductionAuthorityName, str], str]] = []
    rejected: set[tuple[str, Literal["malformed", "unsupported", "incomplete"]]] = set()
    try:
        for payload, digest in _walk_candidate_payloads(request, authorities):
            try:
                decoded = _load_strict_json(payload)
            except (TypeError, ValueError):
                rejected.add((digest, "malformed"))
                continue
            try:
                candidate = WanQualificationCandidateV1.model_validate(
                    _json_arrays_to_tuples(decoded), strict=True
                )
            except (TypeError, ValueError):
                rejected.add((digest, "unsupported"))
                continue
            accepted = _qualify_candidate(candidate, authorities, request)
            candidates.append((accepted, digest))
            if tuple(sorted(accepted)) != WAN_PRODUCTION_AUTHORITY_NAMES:
                rejected.add((digest, "incomplete"))
    finally:
        for authority in reversed(authorities):
            authority.close()

    complete: dict[
        tuple[tuple[WanProductionAuthorityName, str], ...],
        tuple[str, ...],
    ] = {}
    for accepted, digest in candidates:
        if tuple(sorted(accepted)) != WAN_PRODUCTION_AUTHORITY_NAMES:
            continue
        bindings = tuple(sorted(accepted.items()))
        complete[bindings] = tuple(sorted(set((*complete.get(bindings, ()), digest))))
    if len(complete) == 1:
        bindings, digests = next(iter(complete.items()))
        selected_digest = digests[0]
        return WanQualificationManifestV1(
            schema_version="wan-production-qualification-manifest.v1",
            status="qualified",
            bindings=bindings,
            missing_roles=(),
            ambiguous_roles=(),
            candidate_digests=tuple((name, selected_digest) for name, _ in bindings),
            rejections=_qualification_rejections(rejected),
        )
    if len(complete) > 1:
        all_bindings = tuple(complete)
        ambiguous = tuple(
            name
            for name in WAN_PRODUCTION_AUTHORITY_NAMES
            if len({dict(bindings)[name] for bindings in all_bindings}) > 1
        )
        return _unqualified_manifest(
            status="ambiguous", roles=ambiguous, rejected=rejected
        )

    role_values: dict[WanProductionAuthorityName, set[str]] = {
        name: set() for name in WAN_PRODUCTION_AUTHORITY_NAMES
    }
    for accepted, _digest in candidates:
        for name, value in accepted.items():
            role_values[name].add(value)
    ambiguous = tuple(name for name, values in role_values.items() if len(values) > 1)
    if ambiguous:
        return _unqualified_manifest(
            status="ambiguous", roles=ambiguous, rejected=rejected
        )
    missing = tuple(name for name, values in role_values.items() if not values)
    return _unqualified_manifest(
        status="missing",
        roles=missing if candidates else WAN_PRODUCTION_AUTHORITY_NAMES,
        rejected=rejected,
    )


def _qualification_rejections(
    rejected: set[tuple[str, Literal["malformed", "unsupported", "incomplete"]]],
) -> tuple[WanQualificationRejectionV1, ...]:
    return tuple(
        WanQualificationRejectionV1(candidate_digest=digest, reason=reason)
        for digest, reason in sorted(rejected)
    )


def _unqualified_manifest(
    *,
    status: Literal["missing", "ambiguous"],
    roles: tuple[WanProductionAuthorityName, ...],
    rejected: set[
        tuple[str, Literal["malformed", "unsupported", "incomplete"]]
    ] | None = None,
) -> WanQualificationManifestV1:
    return WanQualificationManifestV1(
        schema_version="wan-production-qualification-manifest.v1",
        status=status,
        bindings=(),
        missing_roles=roles if status == "missing" else (),
        ambiguous_roles=roles if status == "ambiguous" else (),
        candidate_digests=(),
        rejections=_qualification_rejections(rejected or set()),
    )


def _capture_search_roots(
    roots: tuple[Path, ...],
) -> tuple[RetainedDirectoryAuthority, ...]:
    for index, root in enumerate(roots):
        if any(
            root != other and root.is_relative_to(other)
            for other in roots[:index]
        ):
            raise ValueError("wan_qualification_search_root_invalid")
    authorities: list[RetainedDirectoryAuthority] = []
    identities: set[tuple[int, int]] = set()
    try:
        for root in roots:
            authority = _capture_existing_directory_authority(
                root, error_code="wan_qualification_search_root_invalid"
            )
            identity = (
                os.fstat(authority.descriptor).st_dev,
                os.fstat(authority.descriptor).st_ino,
            )
            if identity in identities:
                authority.close()
                raise ValueError
            identities.add(identity)
            authorities.append(authority)
        return tuple(authorities)
    except (OSError, ValueError):
        for authority in reversed(authorities):
            authority.close()
        raise ValueError("wan_qualification_search_root_invalid") from None


def _capture_existing_directory_authority(
    path: Path, *, error_code: str
) -> RetainedDirectoryAuthority:
    descriptors: list[int] = []
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        if not isinstance(path, Path) or not path.is_absolute() or not error_code:
            raise ValueError
        descriptors.append(os.open(path.anchor, flags))
        for component in path.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError
            expected = os.stat(
                component, dir_fd=descriptors[-1], follow_symlinks=False
            )
            if not stat.S_ISDIR(expected.st_mode):
                raise ValueError
            child = os.open(component, flags, dir_fd=descriptors[-1])
            descriptors.append(child)
            if not os.path.samestat(expected, os.fstat(child)):
                raise ValueError
        identities = tuple(
            (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
            for descriptor in descriptors
        )
        authority = RetainedDirectoryAuthority(
            path=path,
            descriptor=descriptors[-1],
            descriptors=tuple(descriptors),
            identities=identities,
            error_code=error_code,
        )
        if not _retained_search_root_is_current(authority):
            raise ValueError
        return authority
    except (OSError, ValueError):
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise ValueError(error_code) from None


def _retained_search_root_is_current(
    authority: RetainedDirectoryAuthority,
) -> bool:
    try:
        if not _retained_directory_is_stable(authority):
            return False
        for parent, component, child in zip(
            authority.descriptors[:-1],
            authority.path.parts[1:],
            authority.descriptors[1:],
            strict=True,
        ):
            linked = os.stat(component, dir_fd=parent, follow_symlinks=False)
            observed = os.fstat(child)
            if not stat.S_ISDIR(linked.st_mode) or not os.path.samestat(linked, observed):
                return False
        return True
    except (OSError, ValueError):
        return False


def _walk_candidate_payloads(
    request: WanQualificationSearchV1,
    authorities: tuple[RetainedDirectoryAuthority, ...],
) -> Iterator[tuple[bytes, Sha256Digest]]:
    directory_count = 0
    file_count = 0

    def walk(descriptor: int) -> Iterator[tuple[bytes, Sha256Digest]]:
        nonlocal directory_count, file_count
        directory_count += 1
        if directory_count > request.maximum_directories:
            raise ValueError("wan_qualification_search_limit_exceeded")
        with os.scandir(descriptor) as entries:
            for entry in entries:
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISDIR(info.st_mode):
                    child: int | None = None
                    try:
                        child = os.open(
                            entry.name,
                            os.O_RDONLY
                            | os.O_DIRECTORY
                            | getattr(os, "O_NOFOLLOW", 0)
                            | getattr(os, "O_CLOEXEC", 0),
                            dir_fd=descriptor,
                        )
                        if not os.path.samestat(info, os.fstat(child)):
                            continue
                        yield from walk(child)
                    except OSError:
                        continue
                    finally:
                        if child is not None:
                            os.close(child)
                    continue
                file_count += 1
                if file_count > request.maximum_files:
                    raise ValueError("wan_qualification_search_limit_exceeded")
                if not stat.S_ISREG(info.st_mode) or entry.name != _CANDIDATE_FILENAME:
                    continue
                payload = _read_regular_file_at(descriptor, entry.name, info)
                if payload is not None:
                    yield payload, "sha256:" + hashlib.sha256(payload).hexdigest()

    for authority in authorities:
        root_descriptor = os.dup(authority.descriptor)
        try:
            yield from walk(root_descriptor)
        finally:
            os.close(root_descriptor)


def _read_regular_file_at(
    parent_descriptor: int, name: str, expected: os.stat_result
) -> bytes | None:
    descriptor: int | None = None
    try:
        expected_identity = _stable_regular_identity(expected)
        if expected.st_size > _MAXIMUM_CANDIDATE_BYTES:
            return None
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        before = os.fstat(descriptor)
        identity = _stable_regular_identity(before)
        if identity != expected_identity or before.st_size > _MAXIMUM_CANDIDATE_BYTES:
            return None
        payload = bytearray()
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor, min(1024 * 1024, before.st_size - offset), offset
            )
            if not chunk:
                return None
            payload.extend(chunk)
            digest.update(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, before.st_size):
            return None
        if _stable_regular_identity(os.fstat(descriptor)) != identity:
            return None
        verification = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor, min(1024 * 1024, before.st_size - offset), offset
            )
            if not chunk:
                return None
            verification.update(chunk)
            offset += len(chunk)
        if (
            os.pread(descriptor, 1, before.st_size)
            or verification.digest() != digest.digest()
            or _stable_regular_identity(os.fstat(descriptor)) != identity
        ):
            return None
        return bytes(payload)
    except (OSError, ValueError):
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _json_arrays_to_tuples(value: object) -> object:
    if type(value) is list:
        return tuple(_json_arrays_to_tuples(item) for item in cast(list[object], value))
    if type(value) is dict:
        return {
            key: _json_arrays_to_tuples(item)
            for key, item in cast(dict[object, object], value).items()
        }
    return value


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> NoReturn:
    raise ValueError("non-finite JSON number")


def _load_strict_json(value: str | bytes) -> object:
    return json.loads(
        value,
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )


def _qualify_candidate(
    candidate: WanQualificationCandidateV1,
    authorities: tuple[RetainedDirectoryAuthority, ...],
    request: WanQualificationSearchV1,
) -> dict[WanProductionAuthorityName, str]:
    supplied = dict(candidate.bindings)
    evidence = dict(candidate.accepted_compilation_evidence)
    accepted: dict[WanProductionAuthorityName, str] = {}
    if not all(_retained_search_root_is_current(item) for item in authorities):
        return accepted

    artifact_root = _safe_directory_binding(
        supplied.get("VIDEO_LAB_WAN_ARTIFACT_ROOT"), authorities
    )
    database = _safe_file_binding(
        supplied.get("VIDEO_LAB_WAN_ARTIFACT_DATABASE"), authorities
    )
    receipt_id = supplied.get("VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID")
    operation_id = supplied.get("VIDEO_LAB_WAN_COMPILATION_OPERATION_ID")
    provenance = (
        _load_compilation_authorities(
            database, artifact_root, receipt_id=receipt_id, operation_id=operation_id
        )
        if database is not None and artifact_root is not None
        else None
    )
    if provenance is not None:
        assert database is not None and artifact_root is not None
        accepted["VIDEO_LAB_WAN_ARTIFACT_DATABASE"] = os.fspath(database)
        accepted["VIDEO_LAB_WAN_ARTIFACT_ROOT"] = os.fspath(artifact_root)
        for name, authoritative_value in provenance.items():
            supplied_value = supplied.get(name)
            if supplied_value is None or evidence.get(name) != supplied_value:
                continue
            normalized = _normalize_scientific_binding(name, supplied_value)
            if normalized == authoritative_value:
                accepted[name] = normalized

    model_root = _safe_directory_binding(
        supplied.get("VIDEO_LAB_WAN_MODEL_ROOT"), authorities
    )
    model_revision = accepted.get("VIDEO_LAB_WAN_MODEL_REVISION")
    adapter_value = accepted.get("VIDEO_LAB_WAN_ADAPTER_PATHS_JSON")
    if (
        model_root is not None
        and model_revision is not None
        and _REVISION.fullmatch(model_revision)
        and adapter_value is not None
        and _validate_model_inventory(model_root, model_revision, adapter_value, request)
    ):
        accepted["VIDEO_LAB_WAN_MODEL_ROOT"] = os.fspath(model_root)
        accepted["VIDEO_LAB_WAN_MODEL_REVISION"] = model_revision

    cache_root = _safe_directory_binding(
        supplied.get("VIDEO_LAB_WAN_CACHE_ROOT"), authorities
    )
    cache_revision = accepted.get("VIDEO_LAB_WAN_CACHE_REVISION")
    if (
        cache_root is not None
        and cache_revision is not None
        and _REVISION.fullmatch(cache_revision)
        and _validate_cache_revision(cache_root, cache_revision, request)
    ):
        accepted["VIDEO_LAB_WAN_CACHE_ROOT"] = os.fspath(cache_root)
        accepted["VIDEO_LAB_WAN_CACHE_REVISION"] = cache_revision

    if not all(_retained_search_root_is_current(item) for item in authorities):
        return {}
    return accepted


def _normalize_scientific_binding(
    name: WanProductionAuthorityName, value: str
) -> str | None:
    try:
        if name == "VIDEO_LAB_WAN_ADAPTER_PATHS_JSON":
            parsed = _load_strict_json(value)
            if type(parsed) is not dict or tuple(sorted(parsed)) != _ADAPTER_ROLES:
                return None
            for role in _ADAPTER_ROLES:
                _safe_relative_path(parsed[role], allow_dot=role == "wan_snapshot")
            return canonical_json(parsed)
        if name == "VIDEO_LAB_WAN_DEPENDENCY_VERSIONS_JSON":
            parsed = _load_strict_json(value)
            if type(parsed) is not list:
                return None
            pairs = tuple(tuple(item) if type(item) is list else item for item in parsed)
            if any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not str
                for item in pairs
            ):
                return None
            expected = tuple((name, importlib.metadata.version(name)) for name in _DEPENDENCIES)
            if pairs != expected:
                return None
            return _canonical_json_value(pairs)
        if name in {
            "VIDEO_LAB_WAN_REQUIRED_ARTIFACT_BYTES",
            "VIDEO_LAB_WAN_MEMORY_RELEASE_BOUND_BYTES",
        }:
            if not value.isascii() or not value.isdigit() or (len(value) > 1 and value[0] == "0"):
                return None
            amount = int(value)
            if amount <= 0:
                return None
            return str(amount)
        if name == "VIDEO_LAB_WAN_TOLERANCE_JSON":
            parsed = WanNumericalToleranceV1.model_validate(
                _load_strict_json(value), strict=True
            )
            return canonical_json(parsed.model_dump(mode="json", exclude={"schema_version"}))
        if name == "VIDEO_LAB_WAN_TENSOR_ENVELOPE_JSON":
            raw = _load_strict_json(value)
            if type(raw) is not list:
                return None
            envelope = tuple(
                WanTensorEnvelopeV2.model_validate(_json_arrays_to_tuples(item), strict=True)
                for item in raw
            )
            names = tuple(item.name for item in envelope)
            if (
                not names
                or names != tuple(sorted(set(names)))
                or {item.role for item in envelope} != {"activation", "optimizer", "parameter"}
            ):
                return None
            return _canonical_json_value(
                tuple(item.model_dump(mode="json") for item in envelope)
            )
        if name in {
            "VIDEO_LAB_WAN_COMPILATION_OPERATION_ID",
            "VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID",
        }:
            if not value.isascii() or not re.fullmatch(r"[a-z][a-z0-9_]{2,127}", value):
                return None
            return value
        if name in {
            "VIDEO_LAB_WAN_CACHE_REVISION",
            "VIDEO_LAB_WAN_MODEL_REVISION",
        }:
            if _REVISION.fullmatch(value) is None:
                return None
            return value
    except (ImportError, LookupError, TypeError, ValueError):
        return None
    return None


def _safe_relative_path(value: object, *, allow_dot: bool = False) -> Path:
    if type(value) is not str:
        raise ValueError
    path = PurePath(value)
    windows = PureWindowsPath(value)
    if allow_dot and value == ".":
        return Path(value)
    if (
        not value
        or value == "."
        or path.is_absolute()
        or windows.anchor
        or windows.drive
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(PureWindowsPath(part).is_reserved() for part in path.parts)
    ):
        raise ValueError
    return Path(value)


def _canonical_json_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _safe_directory_binding(
    value: str | None, authorities: tuple[RetainedDirectoryAuthority, ...]
) -> Path | None:
    return _safe_path_binding(value, authorities, directory=True)


def _safe_file_binding(
    value: str | None, authorities: tuple[RetainedDirectoryAuthority, ...]
) -> Path | None:
    return _safe_path_binding(value, authorities, directory=False)


def _safe_path_binding(
    value: str | None,
    authorities: tuple[RetainedDirectoryAuthority, ...],
    *,
    directory: bool,
) -> Path | None:
    if type(value) is not str or not value or "\x00" in value:
        return None
    if "\\" in value or value.startswith("//") or os.path.normpath(value) != value:
        return None
    path = Path(value)
    containing_roots = tuple(
        authority
        for authority in authorities
        if _lexically_contained(path, authority.path)
    )
    if not path.is_absolute() or not containing_roots:
        return None
    if not any(
        _validate_contained_path(path, authority, directory=directory)
        for authority in containing_roots
    ):
        return None
    return path


def _lexically_contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_contained_path(
    path: Path, authority: RetainedDirectoryAuthority, *, directory: bool
) -> bool:
    descriptors: list[int] = []
    identities: list[tuple[int, int, int, int, int, int, int]] = []
    try:
        relative = path.relative_to(authority.path)
        if not _retained_search_root_is_current(authority):
            return False
        if not relative.parts:
            return directory and _retained_search_root_is_current(authority)
        parent = authority.descriptor
        for index, component in enumerate(relative.parts):
            last = index == len(relative.parts) - 1
            info = os.stat(component, dir_fd=parent, follow_symlinks=False)
            expected_directory = directory if last else True
            if (expected_directory and not stat.S_ISDIR(info.st_mode)) or (
                not expected_directory and not stat.S_ISREG(info.st_mode)
            ):
                return False
            flags = (
                os.O_RDONLY
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            if expected_directory:
                flags |= os.O_DIRECTORY
            child = os.open(component, flags, dir_fd=parent)
            descriptors.append(child)
            expected_identity = (
                _directory_inventory_identity(info)
                if expected_directory
                else _stable_regular_identity(info)
            )
            observed_identity = (
                _directory_inventory_identity(os.fstat(child))
                if expected_directory
                else _stable_regular_identity(os.fstat(child))
            )
            if expected_identity != observed_identity:
                return False
            identities.append(expected_identity)
            parent = child
        if any(
            identity
            != (
                _directory_inventory_identity(os.fstat(descriptor))
                if stat.S_ISDIR(os.fstat(descriptor).st_mode)
                else _stable_regular_identity(os.fstat(descriptor))
            )
            for descriptor, identity in zip(descriptors, identities, strict=True)
        ):
            return False
        return _retained_search_root_is_current(authority)
    except (OSError, ValueError):
        return False
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _validate_model_inventory(
    root: Path,
    revision: str,
    adapter_value: str,
    request: WanQualificationSearchV1,
) -> bool:
    try:
        adapter = _load_strict_json(adapter_value)
        if type(adapter) is not dict:
            return False
        paths = {
            name: _safe_relative_path(value, allow_dot=name == "wan_snapshot")
            for name, value in cast(dict[str, object], adapter).items()
        }
        if paths["wan_snapshot"] != Path("."):
            return False
        inventory = _bounded_root_inventory(root, request)
        if not _matches_local_revision(root, revision, inventory.manifest_digest):
            return False
        for role in ("scheduler", "text_encoder", "tokenizer", "vae"):
            location = paths[role]
            if not any(
                item == location or item.is_relative_to(location)
                for item in inventory.files
            ):
                return False
        for role in ("transformer_config", "high_transformer", "low_transformer"):
            if paths[role] not in inventory.files:
                return False
        return True
    except (OSError, TypeError, ValueError):
        return False


def _matches_local_revision(root: Path, revision: str, manifest_digest: str) -> bool:
    if root.name == revision and root.parent.name == "snapshots":
        return True
    component_revision = hashlib.sha256(manifest_digest.encode("ascii")).hexdigest()[:40]
    return revision == component_revision


def _validate_cache_revision(
    root: Path, revision: str, request: WanQualificationSearchV1
) -> bool:
    authority: RetainedDirectoryAuthority | None = None
    try:
        authority, inventory = _capture_bounded_root_inventory(root, request)
        if Path("snapshots", revision) in inventory.directories:
            return True
        for relative in sorted(
            item
            for item in inventory.files
            if item.parent == Path("refs")
        ):
            payload = _read_bounded_relative_file(
                authority.descriptor, relative, maximum_bytes=4096
            )
            if payload is not None and payload.decode("utf-8").strip() == revision:
                return True
        return _matches_local_revision(root, revision, inventory.manifest_digest)
    except (OSError, UnicodeError, ValueError):
        return False
    finally:
        if authority is not None:
            authority.close()


@dataclass(frozen=True, slots=True)
class _BoundedRootInventory:
    manifest_digest: str
    files: frozenset[Path]
    directories: frozenset[Path]


def _bounded_root_inventory(
    root: Path, request: WanQualificationSearchV1
) -> _BoundedRootInventory:
    authority: RetainedDirectoryAuthority | None = None
    try:
        authority, inventory = _capture_bounded_root_inventory(root, request)
        return inventory
    finally:
        if authority is not None:
            authority.close()


def _capture_bounded_root_inventory(
    root: Path, request: WanQualificationSearchV1
) -> tuple[RetainedDirectoryAuthority, _BoundedRootInventory]:
    authority = _capture_existing_directory_authority(
        root, error_code="wan_qualification_inventory_root_invalid"
    )
    try:
        records: list[bytes] = []
        files: set[Path] = set()
        directories: set[Path] = set()
        observed_bytes = 0
        observed_directories = 0
        maximum_files = min(request.maximum_files, _MAXIMUM_ROOT_INVENTORY_FILES)
        maximum_directories = min(
            request.maximum_directories, _MAXIMUM_ROOT_INVENTORY_DIRECTORIES
        )

        def walk(descriptor: int, relative: Path) -> None:
            nonlocal observed_bytes, observed_directories
            before = _directory_inventory_identity(os.fstat(descriptor))
            with os.scandir(descriptor) as scan:
                for entry in scan:
                    child_relative = relative / entry.name
                    if len(child_relative.parts) > _MAXIMUM_AUTHORITY_DEPTH:
                        raise ValueError("wan_qualification_inventory_limit_exceeded")
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        observed_directories += 1
                        if observed_directories > maximum_directories:
                            raise ValueError("wan_qualification_inventory_limit_exceeded")
                        child = os.open(
                            entry.name,
                            os.O_RDONLY
                            | os.O_DIRECTORY
                            | getattr(os, "O_NOFOLLOW", 0)
                            | getattr(os, "O_CLOEXEC", 0),
                            dir_fd=descriptor,
                        )
                        try:
                            child_identity = _directory_inventory_identity(info)
                            if child_identity != _directory_inventory_identity(
                                os.fstat(child)
                            ):
                                raise ValueError("wan_qualification_inventory_changed")
                            directories.add(child_relative)
                            walk(child, child_relative)
                            if child_identity != _directory_inventory_identity(
                                os.fstat(child)
                            ):
                                raise ValueError("wan_qualification_inventory_changed")
                        finally:
                            os.close(child)
                        continue
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError("wan_qualification_inventory_special_file")
                    if len(files) >= maximum_files:
                        raise ValueError("wan_qualification_inventory_limit_exceeded")
                    observed_bytes += info.st_size
                    if observed_bytes > _MAXIMUM_AUTHORITY_BYTES:
                        raise ValueError("wan_qualification_inventory_limit_exceeded")
                    record = _hash_regular_file_at(
                        descriptor, entry.name, info, child_relative
                    )
                    files.add(child_relative)
                    records.append(record)
            if before != _directory_inventory_identity(os.fstat(descriptor)):
                raise ValueError("wan_qualification_inventory_changed")

        walk(authority.descriptor, Path())
        if not _retained_directory_is_stable(authority):
            raise ValueError("wan_qualification_inventory_changed")
        inventory = _BoundedRootInventory(
            manifest_digest="sha256:"
            + hashlib.sha256(b"\n".join(sorted(records))).hexdigest(),
            files=frozenset(files),
            directories=frozenset(directories),
        )
        return authority, inventory
    except BaseException:
        authority.close()
        raise


def _directory_inventory_identity(
    observed: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    if not stat.S_ISDIR(observed.st_mode):
        raise ValueError("wan_qualification_inventory_changed")
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_nlink,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _hash_regular_file_at(
    directory_fd: int,
    name: str,
    expected: os.stat_result,
    relative: Path,
) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY
        | os.O_NONBLOCK
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if _stable_regular_identity(expected) != _stable_regular_identity(before):
            raise ValueError("wan_qualification_inventory_changed")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(descriptor, min(1 << 20, before.st_size - offset), offset)
            if not chunk:
                raise ValueError("wan_qualification_inventory_changed")
            digest.update(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, offset) or _stable_regular_identity(
            before
        ) != _stable_regular_identity(os.fstat(descriptor)):
            raise ValueError("wan_qualification_inventory_changed")
        return (
            relative.as_posix().encode()
            + b"\0"
            + str(offset).encode()
            + b"\0"
            + digest.digest()
        )
    finally:
        os.close(descriptor)


def _read_bounded_relative_file(
    root_descriptor: int, relative: Path, *, maximum_bytes: int
) -> bytes | None:
    opened: list[int] = []
    parent = root_descriptor
    try:
        for component in relative.parts[:-1]:
            parent = os.open(
                component,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent,
            )
            opened.append(parent)
        info = os.stat(relative.name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum_bytes:
            return None
        return _read_regular_file_at(parent, relative.name, info)
    except (OSError, ValueError):
        return None
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)
    return False


def _retained_directory_is_stable(authority: RetainedDirectoryAuthority) -> bool:
    try:
        return tuple(
            (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
            for descriptor in authority.descriptors
        ) == authority.identities
    except OSError:
        return False


def _stable_regular_identity(
    observed: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise ValueError("qualification file authority invalid")
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_nlink,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _read_retained_artifact(
    authority: RetainedDirectoryAuthority,
    relative_value: str,
    *,
    expected_size: int,
    expected_digest: str,
    capture_payload: bool,
) -> bytes | None:
    descriptors: list[int] = []
    try:
        relative = _safe_relative_path(relative_value)
        parent = authority.descriptor
        for component in relative.parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent,
            )
            descriptors.append(child)
            parent = child
        descriptor = os.open(
            relative.name,
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
        descriptors.append(descriptor)
        identity = _stable_regular_identity(os.fstat(descriptor))
        if (
            expected_size < 0
            or expected_size > _MAXIMUM_AUTHORITY_BYTES
            or identity[4] != expected_size
        ):
            return None
        digest = hashlib.sha256()
        payload = bytearray()
        offset = 0
        while offset < expected_size:
            chunk = os.pread(descriptor, min(1024 * 1024, expected_size - offset), offset)
            if not chunk:
                return None
            digest.update(chunk)
            if capture_payload:
                payload.extend(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, expected_size):
            return None
        if _stable_regular_identity(os.fstat(descriptor)) != identity:
            return None
        if digest.hexdigest() != expected_digest or not _retained_directory_is_stable(authority):
            return None
        return bytes(payload)
    except (OSError, TypeError, ValueError):
        return None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _input_from_row(row: sqlite3.Row) -> InputBinding:
    return InputBinding(
        role=cast(Literal["source_manifest", "transform_recipe"], str(row["role"])),
        artifact_id=str(row["artifact_id"]),
        sha256=str(row["sha256"]),
        byte_size=int(row["byte_size"]),
        media_type=str(row["media_type"]),
    )


def _output_from_row(row: sqlite3.Row) -> CompilationOutput:
    return CompilationOutput(
        role=str(row["role"]),
        artifact_id=str(row["artifact_id"]),
        sha256=str(row["sha256"]),
        byte_size=int(row["byte_size"]),
        media_type=str(row["media_type"]),
    )


def _load_compilation_authorities(
    database: Path,
    artifact_root: Path,
    *,
    receipt_id: str | None,
    operation_id: str | None,
) -> dict[WanProductionAuthorityName, str] | None:
    if receipt_id is None or operation_id is None:
        return None
    database_authority: RetainedSqliteAuthority | None = None
    artifact_authority: RetainedDirectoryAuthority | None = None
    try:
        database_authority = RetainedSqliteAuthority.capture(database)
        if os.fstat(database_authority.descriptor).st_nlink != 1:
            return None
        artifact_authority = _capture_existing_directory_authority(
            artifact_root, error_code="wan_qualification_artifact_root_invalid"
        )
        with database_authority.connect_readonly() as connection:
            operation = connection.execute(
                "SELECT id,operation_key,compiler_id,request_digest,request_json,status,attempt,"
                "receipt_id,receipt_digest,receipt_json FROM dataset_compilation_operations "
                "WHERE id=? AND receipt_id=?",
                (operation_id, receipt_id),
            ).fetchone()
            if operation is None:
                return None
            request = DatasetCompilationRequest.model_validate(
                _json_arrays_to_tuples(_load_strict_json(str(operation["request_json"]))),
                strict=True,
            )
            receipt = DatasetCompilationReceipt.model_validate(
                _json_arrays_to_tuples(_load_strict_json(str(operation["receipt_json"]))),
                strict=True,
            )
            canonical_request = canonical_json(request.model_dump(mode="json"))
            canonical_receipt = canonical_json(receipt.model_dump(mode="json"))
            receipt_digest = hashlib.sha256(canonical_receipt.encode()).hexdigest()
            if (
                str(operation["id"]) != operation_id
                or str(operation["status"]) != "completed"
                or str(operation["operation_key"]) != request.operation_key
                or request.operation_key != receipt.operation_key
                or request.operation_key != "wan_production_qualification"
                or str(operation["compiler_id"]) != request.compiler_id
                or request.compiler_id != receipt.compiler_id
                or request.compiler_id != "wan_curated_v1"
                or str(operation["request_json"]) != canonical_request
                or str(operation["request_digest"]) != request.digest
                or str(operation["request_digest"]) != receipt.request_digest
                or str(operation["receipt_id"]) != receipt_id
                or str(operation["receipt_json"]) != canonical_receipt
                or str(operation["receipt_digest"]) != receipt_digest
                or receipt.receipt_id != receipt_id
                or receipt.operation_id != operation_id
                or receipt.attempt != int(operation["attempt"])
                or receipt.source != request.source
                or receipt.recipe != request.recipe
                or receipt.reproduces_receipt_id != request.reproduces_receipt_id
                or request.output_roles != ("batch", "manifest", "provenance")
                or tuple(output.role for output in receipt.outputs) != request.output_roles
            ):
                return None
            input_rows = connection.execute(
                "SELECT operation_id,role,artifact_id,sha256,byte_size,media_type "
                "FROM dataset_compilation_inputs WHERE operation_id=? ORDER BY role",
                (operation_id,),
            ).fetchall()
            output_rows = connection.execute(
                "SELECT receipt_id,operation_id,role,artifact_id,sha256,byte_size,"
                "media_type,attempt "
                "FROM dataset_compilation_outputs WHERE operation_id=? ORDER BY role",
                (operation_id,),
            ).fetchall()
            if (
                any(str(row["operation_id"]) != operation_id for row in input_rows)
                or any(
                    str(row["receipt_id"]) != receipt.receipt_id
                    or str(row["receipt_id"]) != str(operation["receipt_id"])
                    or str(row["operation_id"]) != receipt.operation_id
                    or int(row["attempt"]) != receipt.attempt
                    or int(row["attempt"]) != int(operation["attempt"])
                    for row in output_rows
                )
                or tuple(_input_from_row(row) for row in input_rows)
                != (request.source, request.recipe)
                or tuple(_output_from_row(row) for row in output_rows) != receipt.outputs
            ):
                return None
            completed = connection.execute(
                "SELECT operation_id,attempt,evidence_digest "
                "FROM dataset_compilation_events "
                "WHERE operation_id=? AND event_kind='completed' ORDER BY sequence",
                (operation_id,),
            ).fetchall()
            if len(completed) != 1 or tuple(completed[0]) != (
                operation_id,
                receipt.attempt,
                receipt_digest,
            ):
                return None
            bindings: tuple[InputBinding | CompilationOutput, ...] = (
                request.source,
                request.recipe,
                *receipt.outputs,
            )
            recipe_payload: bytes | None = None
            for binding in bindings:
                artifact = connection.execute(
                    "SELECT id,sha256,byte_size,media_type,relative_path FROM artifacts WHERE id=?",
                    (binding.artifact_id,),
                ).fetchone()
                if artifact is None or tuple(artifact)[:4] != (
                    binding.artifact_id,
                    binding.sha256,
                    binding.byte_size,
                    binding.media_type,
                ):
                    return None
                payload = _read_retained_artifact(
                    artifact_authority,
                    str(artifact["relative_path"]),
                    expected_size=binding.byte_size,
                    expected_digest=binding.sha256,
                    capture_payload=binding.role == "transform_recipe",
                )
                if payload is None:
                    return None
                if binding.role == "transform_recipe":
                    recipe_payload = payload
        database_authority.reverify()
        if recipe_payload is None or not _retained_directory_is_stable(artifact_authority):
            return None
        recipe = _WanQualificationRecipeV1.model_validate(
            _json_arrays_to_tuples(_load_strict_json(recipe_payload)), strict=True
        )
        authorities: dict[WanProductionAuthorityName, str] = {
            "VIDEO_LAB_WAN_COMPILATION_OPERATION_ID": operation_id,
            "VIDEO_LAB_WAN_COMPILATION_RECEIPT_ID": receipt_id,
        }
        for name, value in recipe.authority_bindings:
            normalized = _normalize_scientific_binding(name, value)
            if normalized is not None:
                authorities[name] = normalized
        return authorities
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return None
    finally:
        if artifact_authority is not None:
            artifact_authority.close()
        if database_authority is not None:
            database_authority.close()


__all__ = (
    "WAN_PRODUCTION_AUTHORITY_NAMES",
    "WanProductionAuthorityName",
    "WanQualificationCandidateV1",
    "WanQualificationManifestV1",
    "WanQualificationRejectionV1",
    "WanQualificationSearchV1",
    "discover_wan_production_authorities",
)

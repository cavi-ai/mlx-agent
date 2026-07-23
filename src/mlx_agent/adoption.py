"""Provider-neutral, resumable model adoption state machine."""

from __future__ import annotations

import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .discovery import DiscoveryRequest, DiscoveryService
from .transactions import (
    ConcurrentTransactionError,
    _atomic_in_directory,
    _read_target,
    _target_locks,
)
from .verification import (
    EvidenceStrength,
    TOOL_USE_PROBE_ID,
    VerificationEvidence,
    VerificationStatus,
    Verifier,
)


ADOPTION_SCHEMA_VERSION = "1.2"
LEGACY_ADOPTION_SCHEMA_VERSION = "1.1"
PHASES = ("inspect", "discover", "shortlist", "verify", "compare", "recommend", "complete")
UTILITY_ROLES = {"general", "embedding"}
ALLOWED_ROLES = {
    "general", "coding", "reasoning", "vision", "embedding", "tool-use"
}
EVIDENCE_SCORES = {
    EvidenceStrength.RUNTIME_TESTED.value: 400,
    EvidenceStrength.RUNTIME_INVENTORY.value: 300,
    EvidenceStrength.METADATA_ONLY.value: 200,
    EvidenceStrength.HEURISTIC_ONLY.value: 100,
}


class AdoptionStateConflictError(ValueError):
    """A persisted handoff changed or is locked by another workflow."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AdoptionRequest:
    """Bounded request fields that are safe to persist in a handoff."""

    roles: Tuple[str, ...] = ("general",)
    state_path: object = None
    shortlist_limit: int = 4
    allow_network: bool = True
    offline: bool = False
    refresh: bool = False
    memory_gb: object = None
    quantization: object = None
    licenses: object = None
    include_gated: bool = True
    publishers: object = None
    runtime: object = None
    fast: bool = False

    def __post_init__(self):
        if isinstance(self.roles, str):
            supplied_roles = (self.roles,)
        elif isinstance(self.roles, (list, tuple)):
            supplied_roles = tuple(self.roles)
        else:
            raise TypeError("roles must be a string or an array of strings")
        if any(not isinstance(role, str) or not role for role in supplied_roles):
            raise TypeError("roles must contain non-empty strings")
        roles = tuple(supplied_roles)
        if not roles or any(role not in ALLOWED_ROLES for role in roles) or len(roles) > 6:
            raise ValueError("roles must contain one to six supported roles")
        if len(set(roles)) != len(roles):
            raise ValueError("roles must not contain duplicates")
        if not isinstance(self.shortlist_limit, int) or isinstance(self.shortlist_limit, bool):
            raise TypeError("shortlist_limit must be an integer")
        if self.shortlist_limit < 1 or self.shortlist_limit > 20:
            raise ValueError("shortlist_limit must be between 1 and 20")
        if not isinstance(self.allow_network, bool):
            raise TypeError("allow_network must be a boolean")
        for name in ("offline", "refresh", "include_gated", "fast"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError("{0} must be a boolean".format(name))
        if self.memory_gb is not None and (
            not isinstance(self.memory_gb, (int, float)) or isinstance(self.memory_gb, bool)
        ):
            raise TypeError("memory_gb must be a number or null")
        for name in ("quantization", "runtime"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError("{0} must be a string or null".format(name))
        object.__setattr__(self, "roles", roles)

    def to_dict(self):
        return {
            "roles": list(self.roles),
            "shortlist_limit": self.shortlist_limit,
            "allow_network": self.allow_network,
            "offline": bool(self.offline),
            "refresh": bool(self.refresh),
            "memory_gb": self.memory_gb,
            "quantization": self.quantization,
            "licenses": _string_list(self.licenses),
            "include_gated": bool(self.include_gated),
            "publishers": _string_list(self.publishers),
            "runtime": self.runtime,
            "fast": bool(self.fast),
        }

    @classmethod
    def from_dict(cls, value, state_path=None):
        if not isinstance(value, dict):
            raise TypeError("adoption request must be an object")
        allowed = {
            "roles", "state_path", "state", "shortlist_limit", "limit",
            "allow_network", "offline", "refresh", "memory_gb", "quantization",
            "licenses", "include_gated", "publishers", "runtime", "fast",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError("adoption request has unexpected keys: {0}".format(unknown))
        return cls(
            roles=value.get("roles", ("general",)),
            state_path=state_path or value.get("state_path") or value.get("state"),
            shortlist_limit=value.get("shortlist_limit", value.get("limit", 4)),
            allow_network=value.get("allow_network", True),
            offline=value.get("offline", False),
            refresh=value.get("refresh", False),
            memory_gb=value.get("memory_gb"),
            quantization=value.get("quantization"),
            licenses=value.get("licenses"),
            include_gated=value.get("include_gated", True),
            publishers=value.get("publishers"),
            runtime=value.get("runtime"),
            fast=value.get("fast", False),
        )


@dataclass
class AdoptionState:
    """Schema-versioned, serializable adoption handoff."""

    workflow_id: str
    phase: str
    status: str
    request: Dict[str, Any]
    state_path: Path = field(repr=False)
    revision: int = 0
    completed_phases: List[str] = field(default_factory=list)
    host: Dict[str, Any] = field(default_factory=dict)
    discovery: Dict[str, Any] = field(default_factory=dict)
    shortlist: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    comparisons: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, str]] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    schema_version: str = ADOPTION_SCHEMA_VERSION
    _source_hash: object = field(default=None, repr=False, compare=False)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
            "revision": self.revision,
            "phase": self.phase,
            "status": self.status,
            "request": self.request,
            "completed_phases": list(self.completed_phases),
            "host": self.host,
            "discovery": self.discovery,
            "shortlist": self.shortlist,
            "evidence": self.evidence,
            "comparisons": self.comparisons,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "errors": self.errors,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value, state_path):
        value = _migrate_state(value)
        _validate_state(value)
        return cls(
            workflow_id=value["workflow_id"],
            revision=value["revision"],
            phase=value["phase"],
            status=value["status"],
            request=dict(value["request"]),
            state_path=Path(state_path),
            completed_phases=list(value["completed_phases"]),
            host=dict(value["host"]),
            discovery=dict(value["discovery"]),
            shortlist=list(value["shortlist"]),
            evidence=list(value["evidence"]),
            comparisons=list(value["comparisons"]),
            recommendations=list(value["recommendations"]),
            warnings=list(value["warnings"]),
            errors=list(value["errors"]),
            created_at=value["created_at"],
            updated_at=value["updated_at"],
            schema_version=value["schema_version"],
        )


class AdoptionWorkflow:
    """Advance adoption one durable phase at a time."""

    def __init__(self, discovery_service=None, verifier=None, state_path=None):
        self.discovery_service = discovery_service or DiscoveryService()
        metadata_client = getattr(self.discovery_service, "_huggingface", None)
        self.verifier = verifier or Verifier(metadata_client=metadata_client)
        self.state_path = Path(state_path) if state_path is not None else None

    def start(self, request):
        if isinstance(request, dict):
            request = AdoptionRequest.from_dict(request)
        if not isinstance(request, AdoptionRequest):
            raise TypeError("request must be an AdoptionRequest or object")
        path_value = request.state_path or self.state_path
        if path_value is None:
            raise ValueError("a state path is required")
        state = AdoptionState(
            workflow_id=str(uuid.uuid4()),
            phase="inspect",
            status="running",
            request=request.to_dict(),
            state_path=Path(path_value),
        )
        self._persist(state)
        return state

    def advance(self, state):
        if not isinstance(state, AdoptionState):
            raise TypeError("state must be an AdoptionState")
        state.phase = _first_incomplete(state.completed_phases)
        if state.phase == "complete":
            state.status = "complete"
            return state

        phase = state.phase
        getattr(self, "_phase_{0}".format(phase))(state)
        if phase not in state.completed_phases:
            state.completed_phases.append(phase)
        state.phase = _first_incomplete(state.completed_phases)
        state.status = "complete" if state.phase == "complete" else "running"
        self._persist(state)
        return state

    def resume(self, path):
        state_path = Path(path)
        content, exists, _mode = _read_target(state_path)
        if not exists:
            raise ValueError("adoption state does not exist")
        value = json.loads(content.decode("utf-8"))
        state = AdoptionState.from_dict(value, state_path)
        state._source_hash = hashlib.sha256(content).hexdigest()
        state.phase = _first_incomplete(state.completed_phases)
        state.status = "complete" if state.phase == "complete" else "running"
        return state

    def status(self, path):
        return self.resume(path)

    def _phase_inspect(self, state):
        host = self.discovery_service.host
        if hasattr(host, "to_dict"):
            host = host.to_dict()
        if not isinstance(host, dict):
            raise TypeError("host inventory must be an object")
        state.host = dict(host)

    def _phase_discover(self, state):
        request = state.request
        roles = request["roles"]
        result = self.discovery_service.discover(DiscoveryRequest(
            role=roles[0] if len(roles) == 1 else None,
            memory_gb=request.get("memory_gb"),
            quantization=request.get("quantization"),
            licenses=request.get("licenses"),
            include_gated=request.get("include_gated", True),
            publishers=request.get("publishers"),
            runtime=request.get("runtime"),
            refresh=request.get("refresh", False),
            offline=request.get("offline", False),
            limit=request["shortlist_limit"],
            fast=request.get("fast", False),
        ))
        if result.status != "ok":
            error = result.to_dict()["error"]
            state.errors.append({key: str(error[key]) for key in ("code", "message", "remediation")})
            self._persist(state)
            raise RuntimeError("discovery failed: {0}".format(error["message"]))
        state.discovery = dict(result.data)
        state.warnings.extend(result.warnings)

    def _phase_shortlist(self, state):
        requested = set(state.request["roles"])
        limit = state.request["shortlist_limit"]
        shortlisted = []
        roles = state.discovery.get("roles", {})
        if not isinstance(roles, dict):
            raise TypeError("discovery roles must be an object")
        for role in state.request["roles"]:
            if role not in requested:
                continue
            for candidate in (roles.get(role) or [])[:limit]:
                record = dict(candidate)
                record["role"] = role
                shortlisted.append(record)
        state.shortlist = shortlisted

    def _phase_verify(self, state):
        clear_inventory_cache = getattr(self.verifier, "clear_inventory_cache", None)
        if callable(clear_inventory_cache):
            clear_inventory_cache()
        workers = verification_concurrency(state.host.get("ram_gb"))

        def verify_one(candidate):
            try:
                evidence = self.verifier.verify(
                    candidate,
                    state.host,
                    allow_network=state.request.get("allow_network", True),
                )
                if not isinstance(evidence, VerificationEvidence):
                    raise TypeError("verifier returned an invalid evidence record")
                return evidence.to_dict()
            except Exception as error:
                repo = str(
                    candidate.get("repo")
                    or candidate.get("repository")
                    or "unknown"
                )
                role = str(candidate.get("role") or "general")
                if role == "tool-use":
                    return VerificationEvidence(
                        repo=repo,
                        role=role,
                        strength=EvidenceStrength.HEURISTIC_ONLY,
                        status=VerificationStatus.FAILED,
                        available_locally=False,
                        loads=None,
                        reasoning_confirmed=None,
                        runtime=None,
                        note=_bounded("Candidate tool-use verification failed."),
                        details={
                            "probe_id": TOOL_USE_PROBE_ID,
                            "reason": "verification_exception",
                        },
                    ).to_dict()
                return VerificationEvidence(
                    repo=repo,
                    role=role,
                    strength=EvidenceStrength.HEURISTIC_ONLY,
                    status=VerificationStatus.FAILED,
                    available_locally=False,
                    loads=None,
                    reasoning_confirmed=None,
                    runtime=None,
                    note=_bounded("Candidate verification failed: {0}".format(error)),
                    details={"error": _bounded(str(error), 160)},
                ).to_dict()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            state.evidence = list(executor.map(verify_one, state.shortlist))

    def _phase_compare(self, state):
        evidence_by_candidate = {
            (item["repo"], item["role"]): item for item in state.evidence
        }
        comparisons = []
        for candidate in state.shortlist:
            repo = candidate.get("repo") or candidate.get("repository")
            role = candidate.get("role")
            evidence = evidence_by_candidate.get((repo, role), {})
            reasons = []
            if candidate.get("fits") is False:
                reasons.append("memory_budget")
            if evidence.get("loads") is False:
                reasons.append("runtime_generation_failed")
            utility_role = role in UTILITY_ROLES or state.request.get("fast", False)
            if utility_role and evidence.get("reasoning_confirmed") is True:
                reasons.append("confirmed_reasoner_for_utility_role")
            evidence_status = evidence.get(
                "status", VerificationStatus.FAILED.value
            )
            if (
                role == "tool-use"
                and evidence_status != VerificationStatus.VERIFIED.value
            ):
                reasons.append("tool_use_not_verified")
            score = EVIDENCE_SCORES.get(evidence.get("strength"), 0)
            score += int(candidate.get("rank_score") or 0)
            score += 20 if candidate.get("trusted") else 0
            comparisons.append({
                "repo": repo,
                "role": role,
                "eligible": not reasons,
                "score": score,
                "evidence_strength": evidence.get("strength", EvidenceStrength.HEURISTIC_ONLY.value),
                "evidence_status": evidence_status,
                "verification_status": evidence_status,
                "rejection_reasons": reasons,
            })
        state.comparisons = comparisons

    def _phase_recommend(self, state):
        candidates_by_candidate = {
            (
                item.get("repo") or item.get("repository"),
                item.get("role"),
            ): item
            for item in state.shortlist
        }
        recommendations = []
        for role in state.request["roles"]:
            eligible = [
                item for item in state.comparisons
                if item["role"] == role and item["eligible"]
                and (
                    role != "tool-use"
                    or item.get("evidence_status")
                    == VerificationStatus.VERIFIED.value
                )
            ]
            eligible.sort(key=lambda item: (-item["score"], item["repo"]))
            if not eligible:
                continue
            chosen = eligible[0]
            candidate = candidates_by_candidate[(chosen["repo"], chosen["role"])]
            recommendations.append({
                "role": role,
                "repo": chosen["repo"],
                "evidence_strength": chosen["evidence_strength"],
                "evidence_status": chosen.get(
                    "evidence_status",
                    chosen.get("verification_status", VerificationStatus.FAILED.value),
                ),
                "estimated_ram_gb": candidate.get("est_ram_gb"),
                "wiring": candidate.get("wiring"),
                "alternatives": [item["repo"] for item in eligible[1:4]],
            })
        state.recommendations = recommendations

    def _persist(self, state):
        destination = state.state_path
        try:
            with _target_locks([destination], create_parents=True):
                current, exists, _mode = _read_target(destination)
                if state._source_hash is None:
                    if state.revision != 0 or exists:
                        raise AdoptionStateConflictError(
                            "adoption state already exists or lacks a CAS baseline"
                        )
                else:
                    if not exists or hashlib.sha256(current).hexdigest() != state._source_hash:
                        raise AdoptionStateConflictError(
                            "adoption state changed after it was resumed"
                        )
                    try:
                        current_value = json.loads(current.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise AdoptionStateConflictError(
                            "adoption state changed after it was resumed"
                        ) from error
                    if current_value.get("revision") != state.revision:
                        raise AdoptionStateConflictError(
                            "adoption state revision changed after it was resumed"
                        )

                next_revision = state.revision + 1
                next_updated_at = _utc_now()
                value = state.to_dict()
                value["revision"] = next_revision
                value["updated_at"] = next_updated_at
                _validate_state(value)
                content = (
                    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode("utf-8")

                def validate_staged(text):
                    _validate_state(json.loads(text))

                _atomic_in_directory(
                    destination.parent,
                    destination.name,
                    content,
                    0o600,
                    validator=validate_staged,
                )
                state.revision = next_revision
                state.updated_at = next_updated_at
                state._source_hash = hashlib.sha256(content).hexdigest()
        except ConcurrentTransactionError as error:
            raise AdoptionStateConflictError(
                "another adoption workflow is updating this handoff"
            ) from error


def verification_concurrency(ram_gb):
    try:
        memory = int(float(ram_gb))
    except (TypeError, ValueError, OverflowError):
        memory = 0
    return int(min(4, max(1, memory // 16)))


def _first_incomplete(completed):
    completed_set = set(completed)
    for phase in PHASES[:-1]:
        if phase not in completed_set:
            return phase
    return "complete"


def _migrate_state(value):
    if not isinstance(value, dict):
        raise TypeError("adoption state must be an object")
    version = value.get("schema_version")
    if version == ADOPTION_SCHEMA_VERSION:
        return value
    if version != LEGACY_ADOPTION_SCHEMA_VERSION:
        raise ValueError("unsupported adoption state schema version")

    migrated = deepcopy(value)
    request = migrated.get("request")
    legacy_roles = (
        list(request.get("roles", []))
        if isinstance(request, dict) and isinstance(request.get("roles", []), list)
        else []
    )
    for collection in ("shortlist", "evidence", "comparisons", "recommendations"):
        records = migrated.get(collection)
        if isinstance(records, list):
            legacy_roles.extend(
                item.get("role") for item in records if isinstance(item, dict)
            )
    if "tool-use" in legacy_roles:
        raise ValueError("legacy adoption state cannot contain tool-use")

    status_by_strength = {
        EvidenceStrength.RUNTIME_TESTED.value: VerificationStatus.VERIFIED.value,
        EvidenceStrength.RUNTIME_INVENTORY.value: VerificationStatus.FAILED.value,
        EvidenceStrength.METADATA_ONLY.value: VerificationStatus.METADATA_ONLY.value,
        EvidenceStrength.HEURISTIC_ONLY.value: VerificationStatus.METADATA_ONLY.value,
    }
    for item in migrated.get("evidence", []):
        if not isinstance(item, dict):
            continue
        strength = item.get("strength")
        if strength in status_by_strength:
            item["status"] = status_by_strength[strength]
    migrated["schema_version"] = ADOPTION_SCHEMA_VERSION
    return migrated


def _validate_state(value):
    required = {
        "schema_version", "workflow_id", "revision", "phase", "status", "request",
        "completed_phases", "host", "discovery", "shortlist", "evidence",
        "comparisons", "recommendations", "warnings", "errors", "created_at", "updated_at",
    }
    if not isinstance(value, dict):
        raise TypeError("adoption state must be an object")
    missing = sorted(required - set(value))
    unexpected = sorted(set(value) - required)
    if missing:
        raise ValueError("adoption state is missing keys: {0}".format(missing))
    if unexpected:
        raise ValueError("adoption state has unexpected keys: {0}".format(unexpected))
    if value["schema_version"] != ADOPTION_SCHEMA_VERSION:
        raise ValueError("unsupported adoption state schema version")
    if not isinstance(value["workflow_id"], str) or not value["workflow_id"]:
        raise ValueError("workflow_id must be a non-empty string")
    if (
        not isinstance(value["revision"], int)
        or isinstance(value["revision"], bool)
        or value["revision"] < 1
    ):
        raise ValueError("revision must be a positive integer")
    if value["phase"] not in PHASES:
        raise ValueError("invalid adoption phase")
    if value["status"] not in ("running", "complete"):
        raise ValueError("invalid adoption status")
    _validate_completed_phases(value)
    if not isinstance(value["request"], dict):
        raise TypeError("adoption request must be an object")
    _validate_request_shape(value["request"])
    AdoptionRequest.from_dict(value["request"])
    for key in ("host", "discovery"):
        if not isinstance(value[key], dict):
            raise TypeError("{0} must be an object".format(key))
    bounds = {
        "shortlist": 100,
        "evidence": 100,
        "comparisons": 100,
        "recommendations": 6,
        "warnings": 50,
        "errors": 50,
    }
    for key, maximum in bounds.items():
        if not isinstance(value[key], list):
            raise TypeError("{0} must be an array".format(key))
        if len(value[key]) > maximum:
            raise ValueError("{0} exceeds its maximum size".format(key))
    _validate_timestamp(value["created_at"], "created_at")
    _validate_timestamp(value["updated_at"], "updated_at")
    _validate_evidence(value["evidence"])
    _validate_phase_artifacts(value)


def _validate_completed_phases(value):
    completed = value["completed_phases"]
    if not isinstance(completed, list):
        raise TypeError("completed_phases must be an array")
    if len(completed) > len(PHASES) - 1:
        raise ValueError("completed_phases contains an invalid phase")
    expected = list(PHASES[:len(completed)])
    if completed != expected:
        raise ValueError("completed_phases must be a unique contiguous phase prefix")
    expected_phase = PHASES[len(completed)]
    if value["phase"] != expected_phase:
        raise ValueError("phase does not match completed_phases")
    expected_status = "complete" if expected_phase == "complete" else "running"
    if value["status"] != expected_status:
        raise ValueError("status does not match phase")


def _validate_request_shape(request):
    required = {
        "roles", "shortlist_limit", "allow_network", "offline", "refresh",
        "memory_gb", "quantization", "licenses", "include_gated", "publishers",
        "runtime", "fast",
    }
    if set(request) != required:
        raise ValueError("adoption request does not match the persisted schema")
    if not isinstance(request["roles"], list) or not request["roles"]:
        raise ValueError("request.roles must be a non-empty array")
    if len(request["roles"]) > 6 or len(set(request["roles"])) != len(request["roles"]):
        raise ValueError("request.roles must be bounded and unique")
    if any(not isinstance(role, str) or not role for role in request["roles"]):
        raise TypeError("request.roles must contain non-empty strings")
    if not isinstance(request["shortlist_limit"], int) or isinstance(request["shortlist_limit"], bool):
        raise TypeError("request.shortlist_limit must be an integer")
    if not 1 <= request["shortlist_limit"] <= 20:
        raise ValueError("request.shortlist_limit is out of range")
    for key in ("allow_network", "offline", "refresh", "include_gated", "fast"):
        if not isinstance(request[key], bool):
            raise TypeError("request.{0} must be a boolean".format(key))
    if request["memory_gb"] is not None and (
        not isinstance(request["memory_gb"], (int, float)) or isinstance(request["memory_gb"], bool)
    ):
        raise TypeError("request.memory_gb must be a number or null")
    for key in ("quantization", "runtime"):
        if request[key] is not None and not isinstance(request[key], str):
            raise TypeError("request.{0} must be a string or null".format(key))
    for key in ("licenses", "publishers"):
        if not isinstance(request[key], list) or len(request[key]) > 20:
            raise ValueError("request.{0} must be a bounded array".format(key))
        if any(not isinstance(item, str) for item in request[key]):
            raise TypeError("request.{0} must contain strings".format(key))


def _validate_timestamp(timestamp, field_name):
    if not isinstance(timestamp, str):
        raise TypeError("{0} must be an ISO-8601 timestamp".format(field_name))
    normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise ValueError("{0} must be an ISO-8601 timestamp".format(field_name))
    if parsed.tzinfo is None:
        raise ValueError("{0} must include a timezone".format(field_name))


def _validate_evidence(evidence):
    required = {
        "repo", "role", "strength", "status", "available_locally", "loads",
        "reasoning_confirmed", "runtime", "note", "details",
    }
    strengths = set(EVIDENCE_SCORES)
    statuses = {status.value for status in VerificationStatus}
    for index, item in enumerate(evidence):
        prefix = "evidence[{0}]".format(index)
        if not isinstance(item, dict):
            raise TypeError("{0} must be an object".format(prefix))
        if set(item) != required:
            raise ValueError("{0} does not match the evidence schema".format(prefix))
        for key in ("repo", "role", "note"):
            if not isinstance(item[key], str) or not item[key]:
                raise ValueError("{0}.{1} must be a non-empty string".format(prefix, key))
        if len(item["note"]) > 300:
            raise ValueError("{0}.note exceeds its maximum length".format(prefix))
        if item["strength"] not in strengths:
            raise ValueError("{0}.strength is invalid".format(prefix))
        if item["status"] not in statuses:
            raise ValueError("{0}.status is invalid".format(prefix))
        if not isinstance(item["available_locally"], bool):
            raise TypeError("{0}.available_locally must be a boolean".format(prefix))
        for key in ("loads", "reasoning_confirmed"):
            if item[key] is not None and not isinstance(item[key], bool):
                raise TypeError("{0}.{1} must be a boolean or null".format(prefix, key))
        if item["runtime"] is not None and not isinstance(item["runtime"], str):
            raise TypeError("{0}.runtime must be a string or null".format(prefix))
        if not isinstance(item["details"], dict):
            raise TypeError("{0}.details must be an object".format(prefix))


def _validate_phase_artifacts(value):
    completed = value["completed_phases"]
    verify_complete = "verify" in completed
    compare_complete = "compare" in completed
    recommend_complete = "recommend" in completed
    shortlist = value["shortlist"]
    evidence = value["evidence"]
    comparisons = value["comparisons"]
    recommendations = value["recommendations"]
    if not verify_complete and evidence:
        raise ValueError("evidence is not valid before verification completes")
    if verify_complete:
        _validate_records_match(shortlist, evidence, "evidence")
    if not compare_complete and comparisons:
        raise ValueError("comparisons are not valid before comparison completes")
    if compare_complete:
        _validate_records_match(shortlist, comparisons, "comparisons")
    if not recommend_complete and recommendations:
        raise ValueError("recommendations are not valid before recommendation completes")


def _validate_records_match(shortlist, records, field_name):
    if len(shortlist) != len(records):
        raise ValueError("{0} does not cover the current shortlist".format(field_name))
    for index, (candidate, record) in enumerate(zip(shortlist, records)):
        if not isinstance(candidate, dict) or not isinstance(record, dict):
            raise TypeError("{0}[{1}] must be an object".format(field_name, index))
        candidate_repo = candidate.get("repo") or candidate.get("repository")
        if not isinstance(candidate_repo, str) or not candidate_repo:
            raise ValueError("shortlist[{0}] has no repository".format(index))
        if not isinstance(candidate.get("role"), str) or not candidate["role"]:
            raise ValueError("shortlist[{0}] has no role".format(index))
        if record.get("repo") != candidate_repo or record.get("role") != candidate["role"]:
            raise ValueError("{0}[{1}] does not match the current shortlist".format(field_name, index))


def _string_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _bounded(value, limit=300):
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."

"""Provider-neutral, resumable model adoption state machine."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .discovery import DiscoveryRequest, DiscoveryService
from .verification import EvidenceStrength, VerificationEvidence, Verifier


ADOPTION_SCHEMA_VERSION = "1.0"
PHASES = ("inspect", "discover", "shortlist", "verify", "compare", "recommend", "complete")
UTILITY_ROLES = {"general", "embedding"}
ALLOWED_ROLES = {"general", "coding", "reasoning", "vision", "embedding"}
EVIDENCE_SCORES = {
    EvidenceStrength.RUNTIME_TESTED.value: 400,
    EvidenceStrength.RUNTIME_INVENTORY.value: 300,
    EvidenceStrength.METADATA_ONLY.value: 200,
    EvidenceStrength.HEURISTIC_ONLY.value: 100,
}


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
        supplied_roles = (self.roles,) if isinstance(self.roles, str) else self.roles
        roles = tuple(dict.fromkeys(str(role) for role in supplied_roles))
        if not roles or any(role not in ALLOWED_ROLES for role in roles) or len(roles) > 5:
            raise ValueError("roles must contain one to five supported roles")
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
            roles=tuple(value.get("roles") or ("general",)),
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

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
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
        _validate_state(value)
        return cls(
            workflow_id=value["workflow_id"],
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
        value = json.loads(state_path.read_text(encoding="utf-8"))
        state = AdoptionState.from_dict(value, state_path)
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
                return VerificationEvidence(
                    repo=str(candidate.get("repo") or candidate.get("repository") or "unknown"),
                    role=str(candidate.get("role") or "general"),
                    strength=EvidenceStrength.HEURISTIC_ONLY,
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
        evidence_by_repo = {item["repo"]: item for item in state.evidence}
        comparisons = []
        for candidate in state.shortlist:
            repo = candidate.get("repo") or candidate.get("repository")
            evidence = evidence_by_repo.get(repo, {})
            reasons = []
            if candidate.get("fits") is False:
                reasons.append("memory_budget")
            if evidence.get("loads") is False:
                reasons.append("runtime_generation_failed")
            utility_role = candidate.get("role") in UTILITY_ROLES or state.request.get("fast", False)
            if utility_role and evidence.get("reasoning_confirmed") is True:
                reasons.append("confirmed_reasoner_for_utility_role")
            score = EVIDENCE_SCORES.get(evidence.get("strength"), 0)
            score += int(candidate.get("rank_score") or 0)
            score += 20 if candidate.get("trusted") else 0
            comparisons.append({
                "repo": repo,
                "role": candidate.get("role"),
                "eligible": not reasons,
                "score": score,
                "evidence_strength": evidence.get("strength", EvidenceStrength.HEURISTIC_ONLY.value),
                "rejection_reasons": reasons,
            })
        state.comparisons = comparisons

    def _phase_recommend(self, state):
        candidates_by_repo = {
            item.get("repo") or item.get("repository"): item for item in state.shortlist
        }
        recommendations = []
        for role in state.request["roles"]:
            eligible = [
                item for item in state.comparisons
                if item["role"] == role and item["eligible"]
            ]
            eligible.sort(key=lambda item: (-item["score"], item["repo"]))
            if not eligible:
                continue
            chosen = eligible[0]
            candidate = candidates_by_repo[chosen["repo"]]
            recommendations.append({
                "role": role,
                "repo": chosen["repo"],
                "evidence_strength": chosen["evidence_strength"],
                "estimated_ram_gb": candidate.get("est_ram_gb"),
                "wiring": candidate.get("wiring"),
                "alternatives": [item["repo"] for item in eligible[1:4]],
            })
        state.recommendations = recommendations

    def _persist(self, state):
        state.updated_at = _utc_now()
        value = state.to_dict()
        _validate_state(value)
        destination = state.state_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=str(destination.parent), delete=False
            ) as temporary:
                json.dump(value, temporary, sort_keys=True)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.replace(temporary_name, destination)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)


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


def _validate_state(value):
    required = {
        "schema_version", "workflow_id", "phase", "status", "request",
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
    if value["phase"] not in PHASES:
        raise ValueError("invalid adoption phase")
    if value["status"] not in ("running", "complete"):
        raise ValueError("invalid adoption status")
    if not isinstance(value["completed_phases"], list) or any(
        phase not in PHASES[:-1] for phase in value["completed_phases"]
    ):
        raise ValueError("completed_phases contains an invalid phase")
    if not isinstance(value["request"], dict):
        raise TypeError("adoption request must be an object")
    AdoptionRequest.from_dict(value["request"])
    for key in ("host", "discovery"):
        if not isinstance(value[key], dict):
            raise TypeError("{0} must be an object".format(key))
    for key in ("shortlist", "evidence", "comparisons", "recommendations", "warnings", "errors"):
        if not isinstance(value[key], list):
            raise TypeError("{0} must be an array".format(key))


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

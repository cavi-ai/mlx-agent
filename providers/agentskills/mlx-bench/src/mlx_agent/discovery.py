"""Model discovery, filtering, ranking, and cached structured Scout results."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .contracts import ResultEnvelope
from .host import HostInventory
from .huggingface import HuggingFaceClient
from .models import (
    DISCOVERY_ROLES,
    REPUTABLE,
    base_name,
    classify,
    classify_roles,
    infer_quantization,
    quant_rank,
    resolve_ram,
    resolve_reasoning,
    wiring,
)


CACHE_SCHEMA_VERSION = "2.0"
CACHE_TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class DiscoveryRequest:
    """Bounded, explicit discovery policy.  Legacy ``new`` and ``fast`` remain supported."""

    role: object = None
    memory_gb: object = None
    quantization: object = None
    licenses: object = None
    include_gated: bool = True
    publishers: object = None
    runtime: object = None
    refresh: bool = False
    offline: bool = False
    limit: int = 6
    new: bool = False
    fast: bool = False

    def cache_request(self):
        """The reproducible request shape, excluding transport-only cache switches."""
        return {
            "role": self.role,
            "memory_gb": self.memory_gb,
            "quantization": _canonical_quantization(self.quantization),
            "licenses": _normalise_values(self.licenses),
            "include_gated": self.include_gated,
            "publishers": _normalise_values(self.publishers),
            "runtime": self.runtime,
            "limit": self.limit,
            "new": self.new,
            "fast": self.fast,
        }


def _normalise_values(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value.lower()]
    return sorted(str(item).lower() for item in value)


def _canonical_quantization(value):
    if value is None:
        return None
    normalized = infer_quantization(str(value))
    return normalized or str(value).strip().lower()


def _utc_now():
    return datetime.now(timezone.utc)


def _parse_timestamp(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return None


class DiscoveryService:
    def __init__(self, host=None, huggingface=None, state_dir=None, cache_enabled=None, cache_ttl_seconds=CACHE_TTL_SECONDS):
        self._host = host
        self._huggingface = huggingface or HuggingFaceClient()
        injected_state_dir = state_dir or os.environ.get("MLX_AGENT_STATE_DIR")
        self._state_dir = Path(injected_state_dir) if injected_state_dir else None
        self._cache_enabled = self._state_dir is not None if cache_enabled is None else cache_enabled
        self._cache_ttl_seconds = cache_ttl_seconds

    @property
    def host(self):
        if self._host is None:
            self._host = HostInventory.detect(self._huggingface.http_get)
        return self._host

    def discover(self, request):
        cache = self._read_cache(request)
        if request.offline:
            if cache is None:
                return ResultEnvelope.fail(
                    "discover",
                    "offline_cache_missing",
                    "No cached discovery response matches this request.",
                    "Run discovery while online first, or remove --offline.",
                )
            data, stale = cache
            warnings = [_warning("stale_cache")] if stale else []
            return ResultEnvelope.ok("discover", self._response_data(request, data, "stale" if stale else "fresh"), warnings=warnings)
        if cache is not None and not cache[1] and not request.refresh:
            return ResultEnvelope.ok("discover", self._response_data(request, cache[0], "fresh"))

        host = self.host
        list_sort = "lastModified" if request.new else "trendingScore"
        list_url_builder = getattr(self._huggingface, "list_models_url", None)
        list_url = (
            list_url_builder(sort=list_sort, limit_fetch=300)
            if callable(list_url_builder) else None
        )
        try:
            raw = self._huggingface.list_models(sort=list_sort)
        except Exception as error:
            return ResultEnvelope.fail(
                "discover",
                "network_unavailable",
                "HuggingFace query failed: {0}".format(error),
                "Check network access to huggingface.co and retry the discovery command.",
                retryable=True,
            )

        data = self._build_response(request, raw, host.to_dict(), list_url=list_url)
        self._write_cache(request, data)
        return ResultEnvelope.ok("discover", self._response_data(request, data, "refreshed" if request.refresh else "miss"))

    def _build_response(self, request, raw, host_data, list_url=None):
        buckets = {role: [] for role in DISCOVERY_ROLES}
        rejected = {}
        seen_repo, seen_base = set(), {}
        rejection_limit = max(1, int(request.limit or 1)) * 3
        for model in raw:
            repo = model.get("id") or model.get("modelId")
            if not repo or repo in seen_repo:
                continue
            seen_repo.add(repo)
            if (
                request.role
                and request.role != "tool-use"
                and classify(repo) != request.role
            ):
                continue
            enrichment = {} if request.fast else self._huggingface.inspect_model(repo)
            roles, tool_use = classify_roles(repo, enrichment)
            selected_roles = (
                (request.role,) if request.role in roles else ()
            ) if request.role else roles
            if not selected_roles:
                continue
            accepted = False
            rejected_candidate = None
            for role in selected_roles:
                candidate = self._candidate(
                    repo,
                    model,
                    role,
                    roles,
                    tool_use,
                    host_data,
                    enrichment,
                    request,
                    list_url,
                )
                reasons = self._rejection_reasons(candidate, request)
                candidate["rejection_reasons"] = reasons
                if reasons:
                    rejected_candidate = rejected_candidate or candidate
                    continue
                accepted = True
                key = (role, candidate["base"])
                current = seen_base.get(key)
                if current is not None:
                    if self._selection_key(candidate) > self._selection_key(current):
                        buckets[role].remove(current)
                        buckets[role].append(candidate)
                        seen_base[key] = candidate
                    continue
                seen_base[key] = candidate
                buckets[role].append(candidate)
            if not accepted and rejected_candidate is not None:
                if len(rejected) < rejection_limit:
                    rejected[repo] = rejected_candidate

        for role in buckets:
            buckets[role].sort(key=self._selection_key, reverse=True)
            buckets[role] = buckets[role][: max(0, int(request.limit))]

        return {
            "host": host_data,
            "fast": request.fast,
            "roles": {role: items for role, items in buckets.items() if items},
            "rejected": rejected,
            "request": request.cache_request(),
        }

    def _candidate(
        self,
        repo,
        model,
        role,
        roles,
        tool_use,
        host_data,
        enrichment,
        request,
        list_url=None,
    ):
        ram, ram_source = resolve_ram(repo, enrichment)
        reasoning, reason_source = resolve_reasoning(repo, enrichment)
        publisher = repo.split("/", 1)[0].lower()
        quantization = infer_quantization(repo)
        downloads = model.get("downloads", 0)
        likes = model.get("likes", 0)
        trusted = publisher in REPUTABLE
        budget = request.memory_gb if request.memory_gb is not None else host_data.get("ram_gb")
        fits = ram is None or budget is None or ram < float(budget) * 0.8
        rank_score = (1000000 if trusted else 0) + (quant_rank(repo) * 10000) + min(downloads, 9999) + min(likes, 999)
        metadata_available = enrichment.get("metadata_available") is True
        gated_status = "unknown"
        if metadata_available:
            gated_status = "gated" if enrichment.get("gated") else "public"
        facts = {
            "repository": repo,
            "downloads": downloads,
            "likes": likes,
            "gated": gated_status,
            "license": enrichment.get("license") if metadata_available else None,
        }
        if enrichment.get("tree_available") is True and enrichment.get("weight_bytes"):
            facts["weight_bytes"] = enrichment["weight_bytes"]
        return {
            # Legacy report keys.
            "repo": repo,
            "role": role,
            "roles": list(roles),
            "downloads": downloads,
            "likes": likes,
            "trusted": trusted,
            "base": base_name(repo),
            "qrank": quant_rank(repo),
            "est_ram_gb": ram,
            "ram_src": ram_source,
            "fits": fits,
            "reasoning": reasoning,
            "reason_src": reason_source,
            "gated": enrichment.get("gated") if metadata_available else None,
            "license": facts["license"],
            "wiring": wiring(repo, role, host_data),
            # Explainable structured fields. Memory remains an estimate, even when
            # the weights came from a measured file-size listing.
            "facts": facts,
            "estimates": {"ram_gb": ram, "memory_budget_gb": budget, "headroom_fraction": 0.2},
            "tool_use": tool_use,
            "heuristics": {
                "role": role,
                "primary_role": roles[0],
                "roles": list(roles),
                "quantization": quantization,
                "trusted_publisher": trusted,
            },
            "provenance": self._provenance(
                repo,
                role,
                tool_use,
                enrichment,
                request.fast,
                ram_source,
                reason_source,
                list_url,
            ),
            "rank_score": rank_score,
            "selection_reasons": self._selection_reasons(role, quantization, trusted, fits),
            "rejection_reasons": [],
        }

    @staticmethod
    def _provenance(
        repo,
        role,
        tool_use,
        enrichment,
        fast,
        ram_source,
        reason_source,
        list_url=None,
    ):
        encoded = repo.replace("/", "%2F")
        records = [{
            "source": "huggingface_model_list",
            "url": list_url or "https://huggingface.co/api/models",
            "fields": ["repository", "downloads", "likes"],
        }]
        if not fast and enrichment.get("metadata_available") is True:
            metadata_fields = ["gated", "license"]
            if reason_source in ("chat_template", "tags", "checked"):
                metadata_fields.append("reasoning")
            if tool_use["source"] in ("chat_template", "tags", "checked"):
                metadata_fields.append("tool_use")
            records.append({
                "source": "huggingface_model_metadata",
                "url": enrichment.get("metadata_url") or "https://huggingface.co/api/models/{0}".format(encoded),
                "fields": metadata_fields,
            })
        if not fast and enrichment.get("tree_available") is True and enrichment.get("weight_bytes"):
            records.append({
                "source": "huggingface_repository_tree",
                "url": enrichment.get("tree_url") or "https://huggingface.co/api/models/{0}/tree/main?recursive=true".format(encoded),
                "fields": ["weight_bytes"],
            })
        if tool_use["source"] in ("chat_template", "tags", "checked"):
            role_basis = (
                "primary role from repository name; tool-use signal from "
                "Hugging Face metadata ({0})"
            ).format(tool_use["source"])
        elif tool_use["source"] == "name":
            role_basis = "primary role and tool-use signal from repository name"
        else:
            role_basis = "primary role from repository name; no tool-use signal"
        records.append({
            "source": "local_role_derivation",
            "fields": ["role", "roles", "primary_role"],
            "basis": role_basis,
        })
        name_fields = ["quantization", "trusted_publisher"]
        if tool_use["source"] == "name":
            name_fields.append("tool_use")
        if reason_source == "name":
            name_fields.append("reasoning")
        records.append({
            "source": "local_name_derivation",
            "fields": name_fields,
        })
        records.append({
            "source": "local_memory_estimate",
            "fields": ["ram_gb", "fits"],
            "basis": ram_source or "unavailable",
        })
        records.append({
            "source": "local_ranking_derivation",
            "fields": ["rank_score", "selection_reasons", "rejection_reasons"],
        })
        return records

    @staticmethod
    def _selection_reasons(role, quantization, trusted, fits):
        reasons = ["role membership: {0}".format(role)]
        if quantization:
            reasons.append("quantization inferred from repository name: {0}".format(quantization))
        if trusted:
            reasons.append("publisher is in the configured reputable set")
        if fits:
            reasons.append("estimated weights fit within the memory budget with headroom")
        return reasons

    @staticmethod
    def _selection_key(candidate):
        # The final repository ID makes ranking stable across equal evidence.
        return (candidate["trusted"], candidate["qrank"], candidate["downloads"], candidate["likes"], candidate["repo"])

    @staticmethod
    def _rejection_reasons(candidate, request):
        reasons = []
        licenses = set(_normalise_values(request.licenses))
        publishers = set(_normalise_values(request.publishers))
        wanted_quantization = _canonical_quantization(request.quantization)
        if not request.include_gated and candidate["facts"]["gated"] == "gated":
            reasons.append("gated")
        if not request.include_gated and candidate["facts"]["gated"] == "unknown":
            reasons.append("gated_status_unknown")
        if licenses and (candidate["license"] or "").lower() not in licenses:
            reasons.append("license")
        if publishers and candidate["repo"].split("/", 1)[0].lower() not in publishers:
            reasons.append("publisher")
        if wanted_quantization and candidate["heuristics"]["quantization"] != wanted_quantization:
            reasons.append("quantization")
        if request.memory_gb is not None and not candidate["fits"]:
            reasons.append("memory_budget")
        if request.runtime and not HostInventory.runtime_supports(request.runtime, candidate["heuristics"]["role"]):
            reasons.append("runtime")
        return reasons

    def _cache_path(self, request):
        encoded = json.dumps(request.cache_request(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self._state_dir / "discovery-{0}.json".format(hashlib.sha256(encoded).hexdigest()[:20])

    def _read_cache(self, request):
        if not self._cache_enabled:
            return None
        try:
            payload = json.loads(self._cache_path(request).read_text())
            if payload.get("schema_version") != CACHE_SCHEMA_VERSION or payload.get("request") != request.cache_request():
                return None
            fetched_at = _parse_timestamp(payload.get("fetched_at"))
            if fetched_at is None or not isinstance(payload.get("response"), dict):
                return None
            stale = (_utc_now() - fetched_at).total_seconds() > self._cache_ttl_seconds
            return payload["response"], stale
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _write_cache(self, request, response):
        if not self._cache_enabled:
            return
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "fetched_at": _utc_now().isoformat(),
                "request": request.cache_request(),
                "response": response,
            }
            destination = self._cache_path(request)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(destination.parent), delete=False) as temporary:
                json.dump(payload, temporary, sort_keys=True)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.replace(temporary_name, destination)
        except OSError:
            # Discovery evidence is still useful if the local cache cannot be written.
            return

    def _cached_data(self, response, status):
        data = dict(response)
        data["cache"] = {"status": status, "ttl_seconds": self._cache_ttl_seconds}
        return data

    def _response_data(self, request, response, status):
        if not self._cache_enabled and self._legacy_compatible(request):
            return {key: response[key] for key in ("host", "fast", "roles")}
        return self._cached_data(response, status)

    @staticmethod
    def _legacy_compatible(request):
        return (
            request.memory_gb is None
            and request.quantization is None
            and request.licenses is None
            and request.include_gated
            and request.publishers is None
            and request.runtime is None
            and not request.refresh
            and not request.offline
        )


def _warning(code):
    return {"code": code, "message": "Cached discovery response is older than the 24 hour TTL."}

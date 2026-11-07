"""Bounded, read-only performance measurement of a locally served model.

Bench never installs, downloads, or starts anything. It measures a model that
is already served by an already-running loopback runtime, using the same
safety envelope as verification: validated loopback URLs, absolute monotonic
deadlines, bounded responses, and redacted errors.
"""

from __future__ import annotations

import json
import os
import secrets
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .verification import (
    EvidenceStrength,
    VerificationEvidence,
    VerificationStatus,
    _bounded,
    _is_installed,
    _runtime_name,
    _safe_error,
    installed_model_ids,
)


BENCH_PROBE_ID = "bench-v1"
BENCH_PROMPT = (
    "You are a careful technical writer. Write a detailed, factual explanation "
    "of how unified memory architecture on Apple Silicon affects local "
    "inference of large language models. Cover memory bandwidth, quantization "
    "tradeoffs, KV-cache growth with context length, and the difference "
    "between dense and mixture-of-experts models. Organize the answer into "
    "clear sections with concrete numbers where possible."
)
BENCH_PROMPT_TOKEN_ESTIMATE = 96
RUNS_MIN, RUNS_MAX, RUNS_DEFAULT = 1, 10, 3
GEN_TOKENS_MIN, GEN_TOKENS_MAX, GEN_TOKENS_DEFAULT = 16, 2048, 128
TIMEOUT_DEFAULT = 120.0
WORKLOADS = ("chat", "prefix", "prefill", "concurrent", "cold")
PROMPT_TOKEN_TARGETS = {"prefix": 16384, "prefill": 32768}
CONCURRENCY_MIN, CONCURRENCY_MAX, CONCURRENCY_DEFAULT = 2, 8, 4
# About 20 tokens a line with common tokenizers; results report the server's own prompt count.
_FILLER_LINE = "Record {0:05d}: unified memory bandwidth, quantization, KV cache growth and decode speed.\n"
_FILLER_LINE_TOKENS = 20


class BenchError(RuntimeError):
    """Classified bench failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


@dataclass(frozen=True)
class BenchMeasurement:
    """Aggregate measured performance for one model on one runtime."""

    repo: str
    runtime: str
    runs: int
    prompt_tokens: int
    gen_tokens: int
    ttft_ms: Optional[float]
    decode_toks: float
    prefill_toks: Optional[float]
    spread_pct: float
    chip: Optional[str]
    measured_at: str
    samples: List[Dict[str, Any]] = field(default_factory=list)
    workload: str = "chat"
    first_ttft_ms: Optional[float] = None
    cached_tokens: Optional[int] = None
    concurrency: Optional[int] = None
    aggregate_decode_toks: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": BENCH_PROBE_ID,
            "outcome": "measured",
            "workload": self.workload,
            "repo": self.repo,
            "runtime": self.runtime,
            "runs": self.runs,
            "prompt_tokens": self.prompt_tokens,
            "gen_tokens": self.gen_tokens,
            "ttft_ms": self.ttft_ms,
            "decode_toks": self.decode_toks,
            "prefill_toks": self.prefill_toks,
            "peak_mem_gb": None,
            "spread_pct": self.spread_pct,
            "chip": self.chip,
            "measured_at": self.measured_at,
            "first_ttft_ms": self.first_ttft_ms,
            "cached_tokens": self.cached_tokens,
            "concurrency": self.concurrency,
            "aggregate_decode_toks": self.aggregate_decode_toks,
            "samples": list(self.samples),
        }

    def to_evidence(self, role="general") -> VerificationEvidence:
        return VerificationEvidence(
            repo=self.repo,
            role=role,
            strength=EvidenceStrength.RUNTIME_MEASURED,
            status=VerificationStatus.VERIFIED,
            available_locally=True,
            loads=True,
            reasoning_confirmed=None,
            runtime=self.runtime,
            note=_bounded(
                "Measured {0} run(s) on this host; median decode "
                "{1} tok/s.".format(self.runs, self.decode_toks)
            ),
            details=self.to_dict(),
        )


def validate_bench_bounds(runs, gen_tokens, timeout):
    if not isinstance(runs, int) or isinstance(runs, bool):
        raise TypeError("runs must be an integer")
    if not RUNS_MIN <= runs <= RUNS_MAX:
        raise ValueError("runs must be between {0} and {1}".format(RUNS_MIN, RUNS_MAX))
    if not isinstance(gen_tokens, int) or isinstance(gen_tokens, bool):
        raise TypeError("gen_tokens must be an integer")
    if not GEN_TOKENS_MIN <= gen_tokens <= GEN_TOKENS_MAX:
        raise ValueError(
            "gen_tokens must be between {0} and {1}".format(GEN_TOKENS_MIN, GEN_TOKENS_MAX)
        )
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
        raise TypeError("timeout must be a number")
    if not 1.0 <= float(timeout) <= 600.0:
        raise ValueError("timeout must be between 1 and 600 seconds")


def validate_workload(workload, runs, concurrency):
    if workload not in WORKLOADS:
        raise ValueError("workload must be one of: {0}".format(", ".join(WORKLOADS)))
    if (not isinstance(concurrency, int) or isinstance(concurrency, bool)
            or not CONCURRENCY_MIN <= concurrency <= CONCURRENCY_MAX):
        raise ValueError("concurrency must be between {0} and {1}".format(CONCURRENCY_MIN, CONCURRENCY_MAX))
    if workload == "cold" and runs != 1:
        raise ValueError("the cold workload measures exactly one run; pass --runs 1")


def workload_prompt(workload, index, session):
    """One run's prompt: prefix runs share all but their final question, prefill runs share nothing.

    The session text starts every long prompt, so a later invocation never reuses this one's cache.
    """
    if workload == "prefix":
        return "Session {0}.\n{1}\nQuestion {2}: what does record {2:05d} describe? Answer in one sentence.".format(
            session, _filler(PROMPT_TOKEN_TARGETS["prefix"]), index + 1)
    if workload == "prefill":
        return "Session {0}, document {1}.\n{2}\nSummarize the document in one sentence.".format(
            session, index + 1, _filler(PROMPT_TOKEN_TARGETS["prefill"]))
    return BENCH_PROMPT


def _filler(tokens):
    return "".join(_FILLER_LINE.format(line) for line in range(1, tokens // _FILLER_LINE_TOKENS + 1))


def measure_runtime(repo, runtime, runs=RUNS_DEFAULT, gen_tokens=GEN_TOKENS_DEFAULT,
                    timeout=TIMEOUT_DEFAULT, chip=None, clock=time.monotonic,
                    now=None, workload="chat", concurrency=CONCURRENCY_DEFAULT):
    """Measure one already-served model; never starts or downloads anything.

    The cold workload skips the warm-up, so its single run includes a JIT load.
    """
    validate_bench_bounds(runs, gen_tokens, timeout)
    validate_workload(workload, runs, concurrency)
    if not isinstance(repo, str) or not repo.strip():
        raise BenchError(
            "invalid_repo",
            "bench requires a non-empty model repository identifier.",
            "Pass --repo as publisher/model exactly as the runtime serves it.",
        )
    stream_generate = getattr(runtime, "stream_generate", None)
    if not callable(stream_generate):
        raise BenchError(
            "unsupported_runtime",
            "This runtime does not support the bounded streaming measurement.",
            "Serve the model with a runtime that supports streaming generation.",
        )
    runtime_name = _runtime_name(runtime)
    try:
        installed = installed_model_ids(runtime.list_models())
    except Exception as error:
        raise BenchError(
            "runtime_unreachable",
            "The runtime inventory could not be read: {0}".format(_safe_error(error)),
            "Confirm the local runtime is running and reachable on its loopback port.",
        )
    if not _is_installed(repo, installed):
        raise BenchError(
            "model_not_serving",
            "The model is not served by the {0} runtime.".format(runtime_name),
            "Load the model in the runtime first; bench never downloads models.",
        )

    per_run_timeout = float(timeout) / runs
    session = secrets.token_hex(4)
    estimate = PROMPT_TOKEN_TARGETS.get(workload, BENCH_PROMPT_TOKEN_ESTIMATE)

    def run_once(index):
        return _timed_run(stream_generate, repo, gen_tokens, per_run_timeout, clock,
                          workload_prompt(workload, index, session), estimate)

    def run_round(index):
        if workload != "concurrent":
            return run_once(index)
        started = clock()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            streams = list(pool.map(lambda _: run_once(index), range(concurrency)))
        elapsed = max(clock() - started, 1e-9)
        tokens = sum(stream.get("completion_tokens") or stream["content_events"] for stream in streams)
        return {"streams": streams, "elapsed_ms": round(elapsed * 1000.0, 1),
                "aggregate_decode_toks": round(tokens / elapsed, 1)}

    try:
        first = None if workload == "cold" else run_round(0)
        rounds = [run_round(index) for index in range(1, runs + 1)]
    except BenchError:
        raise
    except Exception as error:
        raise BenchError(
            "measurement_failed",
            "The measurement run failed: {0}".format(_safe_error(error)),
            "Confirm the model responds to a normal generation request, then retry.",
        )

    samples = [stream for round_ in rounds for stream in round_["streams"]] if workload == "concurrent" else rounds
    decode_values = [sample["decode_toks"] for sample in samples]
    median_decode = statistics.median(decode_values)
    spread = 0.0
    if median_decode > 0 and len(decode_values) > 1:
        spread = (max(decode_values) - min(decode_values)) / median_decode * 100.0
    ttft_values = [sample["ttft_ms"] for sample in samples if sample["ttft_ms"] is not None]
    prefill_values = [
        sample["prefill_toks"] for sample in samples if sample["prefill_toks"] is not None
    ]
    # Prefer the server-reported prompt count over the fixed estimate.
    prompt_token_values = [
        sample["prompt_tokens"] for sample in samples if sample.get("prompt_tokens")
    ]
    prompt_tokens = (
        int(round(statistics.median(prompt_token_values)))
        if prompt_token_values else estimate
    )
    cached_values = [sample["cached_tokens"] for sample in samples if sample.get("cached_tokens") is not None]
    measured_at = now() if callable(now) else _utc_now()
    return BenchMeasurement(
        repo=repo,
        runtime=runtime_name,
        runs=runs,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        ttft_ms=round(statistics.median(ttft_values), 1) if ttft_values else None,
        decode_toks=round(median_decode, 1),
        prefill_toks=round(statistics.median(prefill_values), 1) if prefill_values else None,
        spread_pct=round(spread, 1),
        chip=chip,
        measured_at=measured_at,
        samples=rounds,
        workload=workload,
        # The prefix warm-up is the only run that reads the shared prefix uncached.
        first_ttft_ms=first["ttft_ms"] if workload == "prefix" else None,
        cached_tokens=int(round(statistics.median(cached_values))) if cached_values else None,
        concurrency=concurrency if workload == "concurrent" else None,
        aggregate_decode_toks=(round(statistics.median(round_["aggregate_decode_toks"] for round_ in rounds), 1)
                               if workload == "concurrent" else None),
    )


def _timed_run(stream_generate, repo, gen_tokens, timeout, clock, prompt=BENCH_PROMPT,
               prompt_estimate=BENCH_PROMPT_TOKEN_ESTIMATE):
    started = clock()
    first_content_at = None
    last_event_at = None
    content_events = 0
    usage = None
    for event in stream_generate(repo, prompt, gen_tokens, timeout=timeout):
        if not isinstance(event, dict):
            continue
        last_event_at = clock()
        text = _stream_event_text(event)
        if text:
            content_events += 1
            if first_content_at is None:
                first_content_at = last_event_at
        event_usage = _stream_event_usage(event)
        if event_usage:
            usage = event_usage
    if content_events == 0 and usage is None:
        raise BenchError(
            "empty_stream",
            "The runtime streamed no tokens for the measurement prompt.",
            "Confirm the model generates normally, then retry.",
        )
    finished = clock()
    sample = {
        "content_events": content_events,
        "elapsed_ms": round((finished - started) * 1000.0, 1),
    }
    if first_content_at is not None:
        sample["ttft_ms"] = round((first_content_at - started) * 1000.0, 1)
    else:
        sample["ttft_ms"] = None

    if usage and usage.get("eval_count") and usage.get("eval_duration"):
        sample["decode_toks"] = round(
            usage["eval_count"] / (usage["eval_duration"] / 1e9), 1
        )
        if usage.get("prompt_eval_count") and usage.get("prompt_eval_duration"):
            sample["prefill_toks"] = round(
                usage["prompt_eval_count"] / (usage["prompt_eval_duration"] / 1e9), 1
            )
            sample["prompt_tokens"] = usage["prompt_eval_count"]
    elif first_content_at is not None and content_events > 1 and last_event_at > first_content_at:
        sample["decode_toks"] = round(
            (content_events - 1) / (last_event_at - first_content_at), 1
        )
        ttft_seconds = first_content_at - started
        if ttft_seconds > 0:
            # A server-reported prompt count beats the fixed estimate.
            prompt_tokens = (usage or {}).get("prompt_tokens") or prompt_estimate
            sample["prefill_toks"] = round(prompt_tokens / ttft_seconds, 1)
    elif usage and usage.get("completion_tokens") and finished > started:
        sample["decode_toks"] = round(usage["completion_tokens"] / (finished - started), 1)
    else:
        sample["decode_toks"] = round(content_events / max(finished - started, 1e-9), 1)
    # A server-reported prompt count always beats the estimate, whatever
    # branch produced the decode number.
    if usage and usage.get("prompt_tokens"):
        sample["prompt_tokens"] = usage["prompt_tokens"]
    if usage and usage.get("completion_tokens"):
        sample["completion_tokens"] = usage["completion_tokens"]
    if usage and "cached_tokens" in usage:
        sample["cached_tokens"] = usage["cached_tokens"]
    sample.setdefault("prefill_toks", None)
    return sample


def _stream_event_text(event):
    message = event.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    choices = event.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            return delta["content"]
    return ""


def _stream_event_usage(event):
    usage = event.get("usage")
    normalized = {}
    if isinstance(usage, dict):
        if isinstance(usage.get("completion_tokens"), int):
            normalized["completion_tokens"] = usage["completion_tokens"]
        if isinstance(usage.get("prompt_tokens"), int):
            normalized["prompt_tokens"] = usage["prompt_tokens"]
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        if isinstance(cached, int) and not isinstance(cached, bool) and cached >= 0:
            normalized["cached_tokens"] = cached
    for key in ("eval_count", "eval_duration", "prompt_eval_count", "prompt_eval_duration"):
        value = event.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            normalized[key] = value
    return normalized or None


def _utc_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


BENCH_EXPORT_SCHEMA_VERSION = "1.0"
BENCH_AGGREGATE_SCHEMA_VERSION = "1.0"
MAX_EXPORT_LINE_BYTES = 4096
MAX_EXPORT_LINES_PER_FILE = 5000
MAX_AGGREGATE_ENTRIES = 5000
_COMMUNITY_BENCH_PATH = Path(__file__).resolve().parent / "resources" / "bench-community.json"
_community_bench_cache = None


def export_record(measurement, tool_version):
    """Build one bounded, anonymized export line from a measurement."""
    record = {
        "schema_version": BENCH_EXPORT_SCHEMA_VERSION,
        "repo": measurement.repo,
        "chip": measurement.chip,
        "runtime": measurement.runtime,
        "ttft_ms": measurement.ttft_ms,
        "decode_toks": measurement.decode_toks,
        "prefill_toks": measurement.prefill_toks,
        "measured_at": measurement.measured_at,
        "tool_version": tool_version,
    }
    validate_export_record(record)
    return record


def validate_export_record(record):
    if not isinstance(record, dict):
        raise ValueError("bench export record must be an object")
    if record.get("schema_version") != BENCH_EXPORT_SCHEMA_VERSION:
        raise ValueError("bench export record has an unsupported schema version")
    for key in ("repo", "chip", "runtime", "measured_at", "tool_version"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError("bench export record {0} must be a non-empty string".format(key))
    if len(record["repo"]) > 300 or len(record["chip"]) > 120:
        raise ValueError("bench export record identifiers exceed their bounds")
    for key in ("ttft_ms", "decode_toks", "prefill_toks"):
        value = record.get(key)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("bench export record {0} must be numeric or null".format(key))
        if not 0 <= value <= 1e9:
            raise ValueError("bench export record {0} is outside bench bounds".format(key))
    if record.get("decode_toks") is None:
        raise ValueError("bench export record requires decode_toks")
    return record


def append_export(path, record):
    """Append one validated export line; never reads or rewrites the file."""
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    if len(line.encode("utf-8")) > MAX_EXPORT_LINE_BYTES:
        raise BenchError(
            "invalid_arguments",
            "The export record exceeds its size bound.",
            "Export only records produced by bench run.",
        )
    destination = Path(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(destination), flags, 0o600)
    try:
        os.write(descriptor, line.encode("utf-8"))
    finally:
        os.close(descriptor)
    return destination


def aggregate_exports(paths):
    """Deduplicate and median-aggregate validated export records."""
    newest = {}
    for path in paths:
        location = Path(path)
        try:
            handle = location.open("r", encoding="utf-8")
        except OSError:
            continue
        with handle:
            for index, line in enumerate(handle):
                if index >= MAX_EXPORT_LINES_PER_FILE:
                    break
                stripped = line.strip()
                if not stripped or len(stripped.encode("utf-8")) > MAX_EXPORT_LINE_BYTES:
                    continue
                try:
                    record = validate_export_record(json.loads(stripped))
                except (ValueError, json.JSONDecodeError):
                    continue
                key = (record["repo"], record["chip"], record["runtime"])
                if key not in newest or record["measured_at"] > newest[key]["measured_at"]:
                    newest[key] = record
    groups = {}
    for record in newest.values():
        groups.setdefault((record["repo"], record["chip"]), []).append(record)
    entries = []
    for (repo, chip), records in sorted(groups.items()):
        if len(entries) >= MAX_AGGREGATE_ENTRIES:
            break
        def _median(key):
            values = [item[key] for item in records if item.get(key) is not None]
            return round(statistics.median(values), 1) if values else None

        entries.append({
            "repo": repo,
            "chip": chip,
            "decode_toks": _median("decode_toks"),
            "prefill_toks": _median("prefill_toks"),
            "ttft_ms": _median("ttft_ms"),
            "samples": len(records),
            "updated_at": max(item["measured_at"] for item in records),
        })
    return {"schema_version": BENCH_AGGREGATE_SCHEMA_VERSION, "entries": entries}


def load_community_bench(path=None):
    """Load the bundled community aggregate once; the empty stub yields {}."""
    global _community_bench_cache
    if _community_bench_cache is not None and path is None:
        return _community_bench_cache
    location = Path(path) if path is not None else _COMMUNITY_BENCH_PATH
    entries = {}
    try:
        value = json.loads(location.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("entries"), list):
            for entry in value["entries"]:
                if not isinstance(entry, dict):
                    continue
                repo = entry.get("repo")
                chip = entry.get("chip")
                if isinstance(repo, str) and isinstance(chip, str):
                    entries[(repo, chip)] = {
                        key: entry.get(key)
                        for key in ("decode_toks", "prefill_toks", "ttft_ms", "samples")
                    }
    except (OSError, ValueError, json.JSONDecodeError):
        entries = {}
    if path is None:
        _community_bench_cache = entries
    return entries

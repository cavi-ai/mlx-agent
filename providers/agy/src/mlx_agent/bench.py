"""Bounded, read-only performance measurement of a locally served model.

Bench never installs, downloads, or starts anything. It measures a model that
is already served by an already-running loopback runtime, using the same
safety envelope as verification: validated loopback URLs, absolute monotonic
deadlines, bounded responses, and redacted errors.
"""

from __future__ import annotations

import json
import math
import os
import re
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


BENCH_PROBE_ID = "bench-v2"
BENCH_PROMPT = (
    "You are a careful technical writer. Write a detailed, factual explanation "
    "of how unified memory architecture on Apple Silicon affects local "
    "inference of large language models. Cover memory bandwidth, quantization "
    "tradeoffs, KV-cache growth with context length, and the difference "
    "between dense and mixture-of-experts models. Organize the answer into "
    "clear sections with concrete numbers where possible."
)
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
    prompt_tokens: Optional[int]
    gen_tokens: int
    ttft_ms: Optional[float]
    decode_toks: Optional[float]
    prefill_toks: Optional[float]
    spread_pct: Optional[float]
    chip: Optional[str]
    measured_at: str
    samples: List[Dict[str, Any]] = field(default_factory=list)
    end_to_end_toks: Optional[float] = None
    elapsed_ms: Optional[float] = None
    measurement_method: str = "unknown"
    workload: str = "chat"
    first_ttft_ms: Optional[float] = None
    cached_tokens: Optional[int] = None
    concurrency: Optional[int] = None
    aggregate_decode_toks: Optional[float] = None
    aggregate_end_to_end_toks: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": BENCH_PROBE_ID,
            "outcome": "measured",
            "workload": self.workload,
            "first_ttft_ms": self.first_ttft_ms,
            "cached_tokens": self.cached_tokens,
            "concurrency": self.concurrency,
            "aggregate_decode_toks": self.aggregate_decode_toks,
            "aggregate_end_to_end_toks": self.aggregate_end_to_end_toks,
            "repo": self.repo,
            "runtime": self.runtime,
            "runs": self.runs,
            "prompt_tokens": self.prompt_tokens,
            "gen_tokens": self.gen_tokens,
            "ttft_ms": self.ttft_ms,
            "decode_toks": self.decode_toks,
            "prefill_toks": self.prefill_toks,
            "end_to_end_toks": self.end_to_end_toks,
            "elapsed_ms": self.elapsed_ms,
            "measurement_method": self.measurement_method,
            "peak_mem_gb": None,
            "spread_pct": self.spread_pct,
            "chip": self.chip,
            "measured_at": self.measured_at,
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
                "Measured {0} run(s) on this host; {1}.".format(
                    self.runs,
                    "median server decode {0} tok/s".format(self.decode_toks)
                    if self.decode_toks is not None else "decode stage timing unavailable",
                )
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
    """Measure one already-served model; never starts or downloads anything."""
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

    deadline = clock() + float(timeout)
    session = secrets.token_hex(4)
    round_count = runs + (workload != "cold")

    def run_round(index, budget):
        round_deadline = clock() + budget

        def run_once(_=None):
            remaining = round_deadline - clock()
            if remaining <= 0:
                raise _measurement_timeout()
            return _timed_run(stream_generate, repo, gen_tokens, remaining, clock,
                              workload_prompt(workload, index, session))

        if workload != "concurrent":
            return run_once()
        started = clock()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            streams = list(pool.map(run_once, range(concurrency)))
        elapsed = clock() - started
        counts = [stream["completion_tokens"] for stream in streams]
        return {"streams": streams, "elapsed_ms": round(elapsed * 1000.0, 1),
                "aggregate_decode_toks": None,
                "aggregate_end_to_end_toks": round(sum(counts) / elapsed, 1)
                if all(value is not None for value in counts) and elapsed > 0 else None}

    try:
        rounds = []
        first = None
        for index in range(round_count):
            remaining = deadline - clock()
            if remaining <= 0:
                raise _measurement_timeout()
            result = run_round(index, remaining / (round_count - index))
            if clock() > deadline:
                raise _measurement_timeout()
            if index == 0 and workload != "cold":
                first = result
            else:
                rounds.append(result)
    except BenchError:
        raise
    except Exception as error:
        raise BenchError(
            "measurement_failed",
            "The measurement run failed: {0}".format(_safe_error(error)),
            "Confirm the model responds to a normal generation request, then retry.",
        )

    samples = [stream for item in rounds for stream in item["streams"]] if workload == "concurrent" else rounds

    def median(key):
        values = [sample[key] for sample in samples]
        # A partial set must not look like every requested run measured a metric.
        return round(statistics.median(values), 1) if all(v is not None for v in values) else None

    median_decode = median("decode_toks")
    decode_values = [sample["decode_toks"] for sample in samples]
    spread = 0.0 if median_decode is not None else None
    if median_decode is not None and median_decode > 0 and len(decode_values) > 1:
        spread = (max(decode_values) - min(decode_values)) / median_decode * 100.0
    prompt_tokens = median("prompt_tokens")
    methods = {sample["measurement_method"] for sample in samples}
    method = next(iter(methods)) if len(methods) == 1 else "mixed"
    if method == "server_timings" and median_decode is None and median("prefill_toks") is None:
        method = "mixed"
    measured_at = now() if callable(now) else _utc_now()
    return BenchMeasurement(
        repo=repo,
        runtime=runtime_name,
        runs=runs,
        prompt_tokens=int(round(prompt_tokens)) if prompt_tokens is not None else None,
        gen_tokens=gen_tokens,
        ttft_ms=median("ttft_ms"),
        decode_toks=median_decode,
        prefill_toks=median("prefill_toks"),
        spread_pct=round(spread, 1) if spread is not None else None,
        chip=chip,
        measured_at=measured_at,
        samples=rounds,
        workload=workload,
        first_ttft_ms=first["ttft_ms"] if workload == "prefix" else None,
        cached_tokens=int(round(median("cached_tokens"))) if median("cached_tokens") is not None else None,
        concurrency=concurrency if workload == "concurrent" else None,
        aggregate_end_to_end_toks=(round(statistics.median(item["aggregate_end_to_end_toks"] for item in rounds), 1)
                                  if workload == "concurrent" and all(item["aggregate_end_to_end_toks"] is not None for item in rounds) else None),
        end_to_end_toks=median("end_to_end_toks"),
        elapsed_ms=median("elapsed_ms"),
        measurement_method=method,
    )


def _measurement_timeout():
    return BenchError("measurement_timeout", "The benchmark measurement deadline was exceeded.",
                      "Use fewer runs or a larger timeout for this local model.")


def _timed_run(stream_generate, repo, gen_tokens, timeout, clock, prompt=BENCH_PROMPT):
    started = clock()
    deadline = started + timeout
    first_content_at = None
    content_events = 0
    usage = {}
    for event in stream_generate(repo, prompt, gen_tokens, timeout=timeout):
        observed = clock()
        if observed > deadline:
            raise _measurement_timeout()
        if not isinstance(event, dict):
            continue
        text = _stream_event_text(event)
        if text:
            content_events += 1
            if first_content_at is None:
                first_content_at = observed
        usage.update(_stream_event_usage(event))
    if content_events == 0 and "completion_tokens" not in usage and "eval_count" not in usage:
        raise BenchError(
            "empty_stream",
            "The runtime streamed no tokens for the measurement prompt.",
            "Confirm the model generates normally, then retry.",
        )
    finished = clock()
    if finished > deadline:
        raise _measurement_timeout()
    elapsed = finished - started
    completion_tokens = usage.get("eval_count", usage.get("completion_tokens"))
    prompt_tokens = usage.get("prompt_eval_count", usage.get("prompt_tokens"))
    cached_tokens = usage.get("prompt_eval_cached_count", usage.get("cached_tokens"))
    if cached_tokens is not None and prompt_tokens is not None and cached_tokens > prompt_tokens:
        cached_tokens = None
    sample = {
        "content_events": content_events,
        "elapsed_ms": round(elapsed * 1000.0, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "decode_toks": None,
        "prefill_toks": None,
        "end_to_end_toks": round(completion_tokens / elapsed, 1)
        if completion_tokens is not None and elapsed > 0 else None,
        "usage": dict(usage),
    }
    if first_content_at is not None:
        sample["ttft_ms"] = round((first_content_at - started) * 1000.0, 1)
    else:
        sample["ttft_ms"] = None

    if "eval_count" in usage and usage.get("eval_duration"):
        sample["decode_toks"] = round(
            usage["eval_count"] / (usage["eval_duration"] / 1e9), 1
        )
    if "prompt_eval_count" in usage and usage.get("prompt_eval_duration"):
        uncached = usage["prompt_eval_count"] - cached_tokens if cached_tokens is not None else None
        if "prompt_eval_cached_count" not in usage:
            uncached = usage["prompt_eval_count"]
        if uncached is not None and uncached >= 0:
            sample["prefill_toks"] = round(uncached / (usage["prompt_eval_duration"] / 1e9), 1)
    sample["measurement_method"] = (
        "server_timings" if sample["decode_toks"] is not None or sample["prefill_toks"] is not None
        else "client_wall_clock" if sample["end_to_end_toks"] is not None else "latency_only"
    )
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
        for key in ("completion_tokens", "prompt_tokens"):
            value = usage.get(key)
            if type(value) is int and 0 <= value <= 1e9:
                normalized[key] = value
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            value = details.get("cached_tokens")
            if type(value) is int and 0 <= value <= 1e9:
                normalized["cached_tokens"] = value
    for key in ("eval_count", "prompt_eval_count", "prompt_eval_cached_count"):
        value = event.get(key)
        if type(value) is int and 0 <= value <= 1e9:
            normalized[key] = value
        elif key == "prompt_eval_cached_count" and key in event:
            normalized[key] = None
    for key in ("eval_duration", "prompt_eval_duration"):
        value = event.get(key)
        if type(value) in (int, float) and 1 <= value <= 2 ** 63 - 1 and math.isfinite(value):
            normalized[key] = value
    return normalized


def _utc_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


BENCH_EXPORT_SCHEMA_VERSION = "2.0"
BENCH_AGGREGATE_SCHEMA_VERSION = "2.0"
MEASUREMENT_METHODS = ("server_timings", "client_wall_clock", "latency_only", "mixed")
MAX_EXPORT_LINE_BYTES = 4096
MAX_EXPORT_LINES_PER_FILE = 5000
MAX_AGGREGATE_ENTRIES = 5000
_COMMUNITY_BENCH_PATH = Path(__file__).resolve().parent / "resources" / "bench-community.json"
_community_bench_cache = None


def export_record(measurement, tool_version):
    """Build one bounded, anonymized export line from a measurement."""
    if measurement.workload != "chat":
        raise ValueError("bench exports require the chat workload")
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
        "probe_id": BENCH_PROBE_ID,
        "measurement_method": measurement.measurement_method,
        "gen_tokens": measurement.gen_tokens,
        "prompt_tokens": measurement.prompt_tokens,
        "end_to_end_toks": measurement.end_to_end_toks,
    }
    validate_export_record(record)
    return record


def validate_export_record(record):
    if not isinstance(record, dict):
        raise ValueError("bench export record must be an object")
    version = record.get("schema_version")
    if version not in ("1.0", BENCH_EXPORT_SCHEMA_VERSION):
        raise ValueError("bench export record has an unsupported schema version")
    fields = {"schema_version", "repo", "chip", "runtime", "ttft_ms", "decode_toks",
              "prefill_toks", "measured_at", "tool_version"}
    if version == BENCH_EXPORT_SCHEMA_VERSION:
        fields |= {"probe_id", "measurement_method", "gen_tokens", "prompt_tokens", "end_to_end_toks"}
    if set(record) != fields:
        raise ValueError("bench export record fields do not match its schema")
    for key in ("repo", "chip", "runtime", "measured_at", "tool_version"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError("bench export record {0} must be a non-empty string".format(key))
    if len(record["repo"]) > 300 or len(record["chip"]) > 120:
        raise ValueError("bench export record identifiers exceed their bounds")
    _export_timestamp(record["measured_at"])
    for key in ("ttft_ms", "decode_toks", "prefill_toks", "end_to_end_toks"):
        value = record.get(key)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("bench export record {0} must be numeric or null".format(key))
        if not 0 <= value <= 1e9:
            raise ValueError("bench export record {0} is outside bench bounds".format(key))
    if version == "1.0" and record.get("decode_toks") is None:
        raise ValueError("bench export record requires decode_toks")
    if version == BENCH_EXPORT_SCHEMA_VERSION:
        method = record["measurement_method"]
        if record["probe_id"] != BENCH_PROBE_ID or method not in MEASUREMENT_METHODS:
            raise ValueError("bench export record has an unsupported measurement method")
        gen = record["gen_tokens"]
        if type(gen) not in (int, float) or not GEN_TOKENS_MIN <= gen <= GEN_TOKENS_MAX or gen != int(gen):
            raise ValueError("bench export record gen_tokens is outside bench bounds")
        prompt = record["prompt_tokens"]
        if prompt is not None and (type(prompt) not in (int, float) or not 0 <= prompt <= 1e9 or prompt != int(prompt)):
            raise ValueError("bench export record prompt_tokens must be an integer or null")
        stages = record["decode_toks"] is not None or record["prefill_toks"] is not None
        if method == "server_timings" and not stages:
            raise ValueError("server_timings requires a measured stage rate")
        if method in ("client_wall_clock", "latency_only") and stages:
            raise ValueError("client timing methods cannot claim stage rates")
        if method == "client_wall_clock" and record["end_to_end_toks"] is None:
            raise ValueError("client_wall_clock requires end-to-end throughput")
        if method == "latency_only" and record["end_to_end_toks"] is not None:
            raise ValueError("latency_only cannot claim token throughput")
    return record


def _export_timestamp(value):
    from datetime import datetime

    if len(value) > 64 or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})", value):
        raise ValueError("bench export measured_at must include a date, time and timezone")
    if value[-1] not in "Zz" and (int(value[-5:-3]) > 23 or int(value[-2:]) > 59):
        raise ValueError("bench export measured_at has an invalid timezone offset")
    return datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))


def append_export(path, record):
    """Append one validated export line; never reads or rewrites the file."""
    validate_export_record(record)
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
    """Keep the latest validated export per runtime, method and workload."""
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
                key = (record["repo"], record["chip"], record["runtime"],
                       record.get("measurement_method", "legacy_unclassified"),
                       record.get("gen_tokens"), record.get("prompt_tokens"))
                if key not in newest or _export_timestamp(record["measured_at"]) > _export_timestamp(newest[key]["measured_at"]):
                    newest[key] = record
    entries = []
    # A runtime/method/workload group retains only its latest export. Combining
    # unlike servers or legacy rates would erase how a number was measured.
    for key, record in sorted(newest.items(), key=lambda item: json.dumps(item[0])):
        if len(entries) >= MAX_AGGREGATE_ENTRIES:
            break
        repo, chip, runtime, method, gen_tokens, prompt_tokens = key
        entries.append({
            "repo": repo,
            "chip": chip,
            "runtime": runtime,
            "measurement_method": method,
            "probe_id": record.get("probe_id"),
            "gen_tokens": gen_tokens,
            "prompt_tokens": prompt_tokens,
            "decode_toks": record["decode_toks"],
            "prefill_toks": record["prefill_toks"],
            "end_to_end_toks": record.get("end_to_end_toks"),
            "ttft_ms": record["ttft_ms"],
            "samples": 1,
            "updated_at": record["measured_at"],
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
        if (isinstance(value, dict) and value.get("schema_version") in ("1.0", BENCH_AGGREGATE_SCHEMA_VERSION)
                and isinstance(value.get("entries"), list)):
            ambiguous = set()
            for entry in value["entries"]:
                if not isinstance(entry, dict):
                    continue
                repo = entry.get("repo")
                chip = entry.get("chip")
                if isinstance(repo, str) and isinstance(chip, str):
                    key = (repo, chip)
                    if key in entries or key in ambiguous:
                        entries.pop(key, None)
                        ambiguous.add(key)
                        continue
                    entries[(repo, chip)] = {
                        key: entry.get(key)
                        for key in ("decode_toks", "prefill_toks", "end_to_end_toks", "ttft_ms", "samples",
                                    "runtime", "gen_tokens", "prompt_tokens")
                    }
                    entries[(repo, chip)]["measurement_method"] = entry.get("measurement_method", "legacy_unclassified")
    except (OSError, ValueError, json.JSONDecodeError):
        entries = {}
    if path is None:
        _community_bench_cache = entries
    return entries

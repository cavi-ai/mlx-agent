"""Bounded, read-only performance measurement of a locally served model.

Bench never installs, downloads, or starts anything. It measures a model that
is already served by an already-running loopback runtime, using the same
safety envelope as verification: validated loopback URLs, absolute monotonic
deadlines, bounded responses, and redacted errors.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": BENCH_PROBE_ID,
            "outcome": "measured",
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


def measure_runtime(repo, runtime, runs=RUNS_DEFAULT, gen_tokens=GEN_TOKENS_DEFAULT,
                    timeout=TIMEOUT_DEFAULT, chip=None, clock=time.monotonic,
                    now=None):
    """Measure one already-served model; never starts or downloads anything."""
    validate_bench_bounds(runs, gen_tokens, timeout)
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
    try:
        _timed_run(stream_generate, repo, gen_tokens, per_run_timeout, clock)
        samples = [
            _timed_run(stream_generate, repo, gen_tokens, per_run_timeout, clock)
            for _ in range(runs)
        ]
    except BenchError:
        raise
    except Exception as error:
        raise BenchError(
            "measurement_failed",
            "The measurement run failed: {0}".format(_safe_error(error)),
            "Confirm the model responds to a normal generation request, then retry.",
        )

    decode_values = [sample["decode_toks"] for sample in samples]
    median_decode = statistics.median(decode_values)
    spread = 0.0
    if median_decode > 0 and len(decode_values) > 1:
        spread = (max(decode_values) - min(decode_values)) / median_decode * 100.0
    ttft_values = [sample["ttft_ms"] for sample in samples if sample["ttft_ms"] is not None]
    prefill_values = [
        sample["prefill_toks"] for sample in samples if sample["prefill_toks"] is not None
    ]
    measured_at = now() if callable(now) else _utc_now()
    return BenchMeasurement(
        repo=repo,
        runtime=runtime_name,
        runs=runs,
        prompt_tokens=BENCH_PROMPT_TOKEN_ESTIMATE,
        gen_tokens=gen_tokens,
        ttft_ms=round(statistics.median(ttft_values), 1) if ttft_values else None,
        decode_toks=round(median_decode, 1),
        prefill_toks=round(statistics.median(prefill_values), 1) if prefill_values else None,
        spread_pct=round(spread, 1),
        chip=chip,
        measured_at=measured_at,
        samples=samples,
    )


def _timed_run(stream_generate, repo, gen_tokens, timeout, clock):
    started = clock()
    first_content_at = None
    last_event_at = None
    content_events = 0
    usage = None
    for event in stream_generate(repo, BENCH_PROMPT, gen_tokens, timeout=timeout):
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
            sample["prefill_toks"] = round(BENCH_PROMPT_TOKEN_ESTIMATE / ttft_seconds, 1)
    elif usage and usage.get("completion_tokens") and finished > started:
        sample["decode_toks"] = round(usage["completion_tokens"] / (finished - started), 1)
    else:
        sample["decode_toks"] = round(content_events / max(finished - started, 1e-9), 1)
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
    for key in ("eval_count", "eval_duration", "prompt_eval_count", "prompt_eval_duration"):
        value = event.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            normalized[key] = value
    return normalized or None


def _utc_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()

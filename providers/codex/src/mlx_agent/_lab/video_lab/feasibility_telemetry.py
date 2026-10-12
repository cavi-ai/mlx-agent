from __future__ import annotations

import importlib
import os
import resource
import sys
from dataclasses import dataclass
from time import monotonic
from typing import Any, cast

from video_lab.feasibility_models import FeasibilityStage, StageEvidence

_INFERENCE_STAGES = {
    FeasibilityStage.LOAD_TEXT_STACK,
    FeasibilityStage.LOAD_VAE,
    FeasibilityStage.LOAD_HIGH_TRANSFORMER,
    FeasibilityStage.LOAD_LOW_TRANSFORMER,
    FeasibilityStage.ASSEMBLE_PIPELINE,
    FeasibilityStage.GENERATE_MINIMUM_1,
    FeasibilityStage.GENERATE_MINIMUM_2,
    FeasibilityStage.QUALIFY_STANDARD,
    FeasibilityStage.VALIDATE_CONTROLLED_PAIR,
}
_MPS_RUNTIME: object | None = None


def mark_mps_context_ready(torch_runtime: object) -> None:
    global _MPS_RUNTIME
    _MPS_RUNTIME = torch_runtime


def _unavailable(reason: str) -> dict[str, object]:
    return {"availability": reason}


def _process_snapshot() -> dict[str, object]:
    try:
        psutil = importlib.import_module("psutil")
        process = psutil.Process()
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "rss_bytes": int(process.memory_info().rss),
            "system_used_bytes": int(virtual.used),
            "system_available_bytes": int(virtual.available),
            "swap_used_bytes": int(swap.used),
            "availability": "available",
        }
    except Exception as error:
        return _unavailable(type(error).__name__)


def _rss_peak() -> tuple[int | None, str]:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if not sys.platform.startswith("darwin"):
            value *= 1024
        return value, "available"
    except Exception as error:
        return None, type(error).__name__


def _mps_snapshot(stage: FeasibilityStage) -> dict[str, object]:
    if stage not in _INFERENCE_STAGES:
        return _unavailable("not_applicable")
    if _MPS_RUNTIME is None:
        return _unavailable("context_not_initialized")
    try:
        mps = cast(Any, _MPS_RUNTIME).mps
        return {
            "allocated": int(mps.current_allocated_memory()),
            "driver": int(mps.driver_allocated_memory()),
            "availability": "available",
        }
    except Exception as error:
        return _unavailable(type(error).__name__)


@dataclass
class StageTelemetryCapture:
    stage: FeasibilityStage
    started: float
    before_process: dict[str, object]
    before_mps: dict[str, object]

    @classmethod
    def start(cls, stage: FeasibilityStage) -> StageTelemetryCapture:
        return cls(stage, monotonic(), _process_snapshot(), _mps_snapshot(stage))

    def merge(self, evidence: StageEvidence) -> StageEvidence:
        after_process = _process_snapshot()
        after_mps = _mps_snapshot(self.stage)
        peak_rss, peak_availability = _rss_peak()
        samples = (self.before_process, after_process)
        before_allocated = self.before_mps.get("allocated")
        after_allocated = after_mps.get("allocated")
        before_driver = self.before_mps.get("driver")
        after_driver = after_mps.get("driver")

        def peak(first: object, second: object) -> int | None:
            values = [value for value in (first, second) if isinstance(value, int)]
            return max(values) if values else None

        current_rss = after_process.get("rss_bytes")
        telemetry: dict[str, Any] = {
            "pid": os.getpid(),
            "sample_count": len(samples),
            "samples": samples,
            "rss_bytes": {
                "current": current_rss if isinstance(current_rss, int) else None,
                "peak": peak_rss,
                "availability": peak_availability,
            },
            "mps_bytes": {
                "allocated_before": before_allocated if isinstance(before_allocated, int) else None,
                "allocated_endpoint_max": peak(before_allocated, after_allocated),
                "allocated_after": after_allocated if isinstance(after_allocated, int) else None,
                "driver_before": before_driver if isinstance(before_driver, int) else None,
                "driver_endpoint_max": peak(before_driver, after_driver),
                "driver_after": after_driver if isinstance(after_driver, int) else None,
                "availability": str(after_mps.get("availability", "unavailable")),
            },
        }
        measurements = dict(evidence.measurements)
        measurements["stage_wall_seconds"] = max(0.0, monotonic() - self.started)
        details = evidence.model_dump(mode="json")["details"]
        details["telemetry"] = telemetry
        return StageEvidence(measurements=measurements, details=details)

from datetime import UTC

import psutil

from video_lab.telemetry import TelemetrySampler


def test_sampler_records_process_system_swap_and_optional_thermal(monkeypatch) -> None:
    class Process:
        def memory_info(self):
            return type("Memory", (), {"rss": 101})()

    monkeypatch.setattr(psutil, "Process", Process)
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: type("Virtual", (), {"used": 202, "available": 303})()
    )
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("Swap", (), {"used": 404})())
    timeouts: list[float] = []

    def thermal(timeout_seconds: float) -> str:
        timeouts.append(timeout_seconds)
        return "nominal"

    sample = TelemetrySampler(thermal_provider=thermal, thermal_timeout_seconds=0.05).sample(
        "attempt_1", 1.25
    )

    assert sample.recorded_at.tzinfo is UTC
    assert sample.model_dump(exclude={"recorded_at"}) == {
        "id": None,
        "attempt_id": "attempt_1",
        "elapsed_seconds": 1.25,
        "process_rss_bytes": 101,
        "system_used_bytes": 202,
        "system_available_bytes": 303,
        "swap_used_bytes": 404,
        "thermal_pressure": "nominal",
    }
    assert timeouts == [0.05]


def test_sampler_treats_missing_or_failing_thermal_provider_as_unavailable() -> None:
    assert TelemetrySampler().sample("attempt_1", 0).thermal_pressure is None

    def fail(_timeout_seconds: float) -> str:
        raise OSError("unsupported")

    assert TelemetrySampler(thermal_provider=fail).sample("attempt_1", 0).thermal_pressure is None


def test_sampler_treats_each_unavailable_psutil_component_as_zero(monkeypatch) -> None:
    class BrokenProcess:
        def memory_info(self):
            raise PermissionError("rss denied")

    monkeypatch.setattr(psutil, "Process", BrokenProcess)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: (_ for _ in ()).throw(OSError("vm")))
    monkeypatch.setattr(psutil, "swap_memory", lambda: (_ for _ in ()).throw(OSError("swap")))

    sample = TelemetrySampler().sample("attempt_1", 0)

    assert sample.process_rss_bytes == 0
    assert sample.system_used_bytes == 0
    assert sample.system_available_bytes == 0
    assert sample.swap_used_bytes == 0


def test_sampler_treats_process_construction_failure_as_zero_rss(monkeypatch) -> None:
    monkeypatch.setattr(psutil, "Process", lambda: (_ for _ in ()).throw(RuntimeError("process")))

    assert TelemetrySampler().sample("attempt_1", 0).process_rss_bytes == 0

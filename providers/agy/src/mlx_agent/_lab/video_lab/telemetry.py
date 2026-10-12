from collections.abc import Callable

import psutil  # type: ignore[import-untyped]

from video_lab.domain import TelemetrySample, utc_now

ThermalProvider = Callable[[float], str | None]


class TelemetrySampler:
    def __init__(
        self,
        *,
        thermal_provider: ThermalProvider | None = None,
        thermal_timeout_seconds: float = 0.1,
    ) -> None:
        if thermal_timeout_seconds <= 0:
            raise ValueError("thermal timeout must be positive")
        self._thermal_provider = thermal_provider
        self._thermal_timeout_seconds = thermal_timeout_seconds

    def sample(self, attempt_id: str, elapsed_seconds: float) -> TelemetrySample:
        try:
            process_rss = int(psutil.Process().memory_info().rss)
        except Exception:
            process_rss = 0
        try:
            virtual = psutil.virtual_memory()
            system_used = int(virtual.used)
            system_available = int(virtual.available)
        except Exception:
            system_used = 0
            system_available = 0
        try:
            swap_used = int(psutil.swap_memory().used)
        except Exception:
            # Some sandboxed macOS processes cannot query sysctl swap data.
            swap_used = 0
        thermal_pressure: str | None = None
        if self._thermal_provider is not None:
            try:
                thermal_pressure = self._thermal_provider(self._thermal_timeout_seconds)
            except Exception:
                thermal_pressure = None
        return TelemetrySample(
            attempt_id=attempt_id,
            recorded_at=utc_now(),
            elapsed_seconds=elapsed_seconds,
            process_rss_bytes=process_rss,
            system_used_bytes=system_used,
            system_available_bytes=system_available,
            swap_used_bytes=swap_used,
            thermal_pressure=thermal_pressure,
        )

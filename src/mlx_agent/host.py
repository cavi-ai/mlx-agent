"""Local host inventory probes used by Scout."""

import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HostInventory:
    ram_gb: object = None
    chip: object = None
    ollama: bool = False
    lmstudio: bool = False

    @staticmethod
    def runtime_supports(runtime, role):
        """Whether a runtime can serve the requested model class, independent of install state."""
        if runtime == "mlx-vlm":
            return True
        if role == "vision":
            return False
        return runtime in ("mlx_lm", "ollama", "lmstudio", "litellm")

    @classmethod
    def detect(cls, http_get, check_output=None):
        check_output = check_output or subprocess.check_output
        values = {"ram_gb": None, "chip": None, "ollama": False, "lmstudio": False}
        try:
            values["ram_gb"] = round(int(check_output(["sysctl", "-n", "hw.memsize"]).strip()) / 1073741824)
        except Exception:
            pass
        try:
            values["chip"] = check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
        try:
            http_get("http://127.0.0.1:11434/api/tags", timeout=3)
            values["ollama"] = True
        except Exception:
            pass
        try:
            http_get("http://localhost:1234/v1/models", timeout=3)
            values["lmstudio"] = True
        except Exception:
            pass
        return cls(**values)

    def to_dict(self):
        return asdict(self)

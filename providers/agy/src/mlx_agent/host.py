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
            return role == "vision"
        if role == "vision":
            return False
        return runtime in ("mlx_lm", "ollama", "lmstudio", "litellm")

    @classmethod
    def detect(cls, http_get, check_output=None):
        return cls.inspect(http_get, check_output=check_output)[0]

    @classmethod
    def inspect(cls, http_get, check_output=None):
        """Return bounded local inventory plus classified unavailable-probe warnings."""
        check_output = check_output or subprocess.check_output
        values = {"ram_gb": None, "chip": None, "ollama": False, "lmstudio": False}
        warnings = []
        try:
            values["ram_gb"] = round(int(check_output(["sysctl", "-n", "hw.memsize"]).strip()) / 1073741824)
        except Exception:
            warnings.append({"code": "host_probe_unavailable", "probe": "memory", "message": "Host memory probe unavailable."})
        try:
            values["chip"] = check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            warnings.append({"code": "host_probe_unavailable", "probe": "chip", "message": "Host chip probe unavailable."})
        try:
            http_get("http://127.0.0.1:11434/api/tags", timeout=3)
            values["ollama"] = True
        except Exception:
            warnings.append({"code": "runtime_probe_unavailable", "probe": "ollama", "message": "Local runtime probe unavailable."})
        try:
            http_get("http://localhost:1234/v1/models", timeout=3)
            values["lmstudio"] = True
        except Exception:
            warnings.append({"code": "runtime_probe_unavailable", "probe": "lmstudio", "message": "Local runtime probe unavailable."})
        return cls(**values), warnings

    def to_dict(self):
        return asdict(self)

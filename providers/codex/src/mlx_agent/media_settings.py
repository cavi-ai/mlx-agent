"""Read-only, model-bound settings shared by media planners and clients."""

import hashlib
import json
import os
import stat
from pathlib import Path

from .task_contracts import TaskError


def supports_adaptive_reuse(config, kind):
    if kind == "mesh_generation":
        return config.get("model_type") == "triposg"
    if kind == "video_generation":
        return (config.get("model_type") == "t2v" and config.get("model_version") == "2.1"
                and not config.get("dual_model", False))
    return False


def model_settings(model_path):
    """Inspect only a bounded config file; never import or load a model runtime."""
    try:
        requested = Path(model_path)
        if not requested.is_absolute():
            raise ValueError("The model path must be absolute.")
        model = requested.resolve(strict=True)
        config_path = (model / "config.json").resolve(strict=True)
        descriptor = os.open(config_path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("The model config must be a regular file.")
            data = stream.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("The model config exceeds 1 MiB.")
        config = json.loads(data)
        if not isinstance(config, dict):
            raise ValueError("The model config must be an object.")
    except (OSError, ValueError, RuntimeError) as error:
        raise TaskError("model_unreadable", "Cannot inspect the local model config.",
                        "Select a local model with a readable config.json.") from error
    binding = {"path": str(model), "config_sha256": hashlib.sha256(data).hexdigest()}
    descriptors = {}
    for kind, reason in (("mesh_generation", "Adaptive reuse requires TripoSG."),
                         ("video_generation", "Adaptive reuse requires single-model Wan2.1 text-to-video.")):
        supported = supports_adaptive_reuse(config, kind)
        value = {"modes": ["off", "adaptive"] if supported else ["off"], "default": "off",
                 "approximate": True, "cache_memory_mib": {"minimum": 8, "maximum": 4096, "default": 256}}
        if not supported:
            value["unavailable_reason"] = reason
        descriptors[kind] = value
    return binding, descriptors

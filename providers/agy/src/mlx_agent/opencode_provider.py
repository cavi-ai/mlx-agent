"""OpenCode provider models for live mlx-agent chat endpoints."""

from __future__ import annotations

import json
import sys

from .inference_runner import InferenceError
from .jit_serve import public_model_name

CHAT_RUNTIMES = ("mlx_lm", "mlx-vlm", "strata", "splash")


def provider_models(discover=None, key=None):
    """One model per name a single live chat endpoint serves with a key readable here."""
    if discover is None:
        from .service_registry import discover_services as discover
    if key is None:
        from .served import _key as key
    table = {}
    for service in discover().get("services", []):
        port = service.get("port")
        if not (service.get("alive") and service.get("argv_match") and service.get("runtime") in CHAT_RUNTIMES
                and type(port) is int and 1 <= port <= 65535 and isinstance(service.get("path"), str)):
            continue
        name = public_model_name(service["path"])
        table[name] = None if name in table else service
    models = []
    for name, service in sorted(table.items()):
        if service is None:
            continue
        try:
            secret = key(service)
        except InferenceError:
            continue
        settings = {"baseURL": "http://127.0.0.1:{0}/v1".format(service["port"])}
        if secret is not None:
            settings["apiKey"] = secret
        models.append({"id": name, "runtime": service["runtime"], "settings": settings})
    return models


def main():
    json.dump({"models": provider_models()}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

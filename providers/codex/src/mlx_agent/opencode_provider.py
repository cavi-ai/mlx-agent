"""OpenCode provider models for live mlx-agent chat endpoints."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import sys

from .inference_runner import InferenceError
from .jit_serve import public_model_name

CHAT_RUNTIMES = ("mlx_lm", "mlx-vlm", "strata", "splash")


def _front_door_models(url, secret, discover):
    """Authenticate one explicitly selected loopback front door before listing models."""
    match = re.fullmatch(r"http://127\.0\.0\.1:([1-9][0-9]{0,4})/v1", url or "")
    if not match or int(match.group(1)) > 65535 or not isinstance(secret, str):
        return []
    try:
        token = secret.encode("ascii")
    except UnicodeEncodeError:
        return []
    if not 1 <= len(token) <= 512 or any(value < 33 or value > 126 for value in token):
        return []
    connection = http.client.HTTPConnection("127.0.0.1", int(match.group(1)), timeout=5)
    try:
        connection.request("GET", "/v1/models", headers={"Authorization": "Bearer " + secret})
        response = connection.getresponse()
        if response.status != 200:
            return []
        raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            return []
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("object") != "list" or not isinstance(value.get("data"), list):
            return []
    except (OSError, ValueError, http.client.HTTPException):
        return []
    finally:
        connection.close()
    from .front_door import route_table
    try:
        hints = route_table(discover().get("services", []), hashlib.sha256(token).hexdigest())
    except (OSError, ValueError, TypeError, KeyError):
        hints = {}
    counts = {}
    for row in value["data"]:
        if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"]:
            counts[row["id"]] = counts.get(row["id"], 0) + 1
    models = []
    for row in value["data"]:
        if not isinstance(row, dict):
            continue
        name = row.get("id")
        if (not isinstance(name, str) or not name or counts[name] != 1
                or row.get("object") != "model" or row.get("owned_by") != "mlx-agent"
                or "task_kinds" in row):
            continue
        if name in hints and (hints[name] is None or hints[name].get("runtime") not in CHAT_RUNTIMES):
            continue
        runtime = hints[name]["runtime"] if name in hints else "front_door"
        models.append({"id": name, "runtime": runtime, "settings": {"baseURL": url, "apiKey": secret}})
    return sorted(models, key=lambda item: item["id"])


def provider_models(discover=None, key=None):
    """List direct chat endpoints or routes from an explicitly selected front door."""
    if discover is None:
        from .service_registry import discover_services as discover
    if "MLX_AGENT_FRONT_DOOR_URL" in os.environ or "MLX_AGENT_FRONT_DOOR_KEY" in os.environ:
        return _front_door_models(os.environ.get("MLX_AGENT_FRONT_DOOR_URL"),
                                  os.environ.get("MLX_AGENT_FRONT_DOOR_KEY"), discover)
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

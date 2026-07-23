"""Dependency-free Hugging Face Hub access for Scout."""

import http.client
import json
import queue
import re
import threading
import time
import urllib.parse

from .models import (
    REASONER_HINTS,
    TEMPLATE_REASON,
    TEMPLATE_TOOL_USE,
    TOOL_USE_HINTS,
    TOOL_USE_TAGS,
)


HF_API = "https://huggingface.co/api/models"
HF_API_HOST = "huggingface.co"
UA = {"User-Agent": "mlx-scout/0.2 (+https://github.com/cavi-ai/mlx-agent)"}
HF_RESPONSE_MAX_BYTES = 8 * 1024 * 1024
_HTTP_READ_CHUNK_BYTES = 64 * 1024
HF_CARD_HOST = "huggingface.co"
MODEL_CARD_MAX_BYTES = 512 * 1024
_CARD_PATH_SUFFIX = "/raw/main/README.md"


def http_json(
    url,
    timeout=10.0,
    connection_factory=None,
    clock=time.monotonic,
    completion_wait=None,
):
    """Read bounded JSON from the fixed Hugging Face API under one deadline."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != HF_API_HOST:
        raise ValueError("Hugging Face URL must use the fixed HTTPS API host")
    if parsed.port not in (None, 443):
        raise ValueError("Hugging Face URL must use the default HTTPS port")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Hugging Face URL must not contain credentials")
    if parsed.fragment:
        raise ValueError("Hugging Face URL must not contain a fragment")
    if parsed.path != "/api/models" and not parsed.path.startswith("/api/models/"):
        raise ValueError("Hugging Face URL must target the models API")

    deadline = clock() + timeout
    remaining = _deadline_remaining(deadline, clock)
    if connection_factory is None:
        connection = http.client.HTTPSConnection(
            HF_API_HOST,
            443,
            timeout=remaining,
        )
    else:
        connection = connection_factory(HF_API_HOST, 443, remaining)
    target = parsed.path or "/"
    if parsed.query:
        target = "{0}?{1}".format(target, parsed.query)
    return _run_http_worker(
        connection,
        lambda: _http_json_operation(connection, target, deadline, clock),
        deadline,
        clock,
        completion_wait,
    )


def _run_http_worker(connection, operation, deadline, clock, completion_wait):
    results = queue.Queue(maxsize=1)
    completion = threading.Event()

    def run():
        try:
            result = (True, operation())
        except BaseException as error:  # noqa: BLE001 - propagated to caller
            result = (False, error)
        results.put_nowait(result)
        completion.set()

    worker = threading.Thread(
        target=run,
        name="mlx-agent-huggingface-request",
        daemon=True,
    )
    try:
        worker.start()
    except BaseException:
        connection.close()
        raise

    try:
        remaining = _deadline_remaining(deadline, clock)
    except TimeoutError:
        connection.close()
        raise
    try:
        if completion_wait is None:
            completed = completion.wait(remaining)
        else:
            completed = completion_wait(completion, remaining)
    except BaseException:
        connection.close()
        raise
    if not completed:
        connection.close()
        raise TimeoutError("Hugging Face request exceeded its overall deadline")

    try:
        remaining = _deadline_remaining(deadline, clock)
    except TimeoutError:
        connection.close()
        raise
    worker.join(timeout=remaining)
    if worker.is_alive():
        connection.close()
        raise TimeoutError("Hugging Face request exceeded its overall deadline")

    succeeded, value = results.get_nowait()
    if succeeded:
        return value
    raise value


def _http_json_operation(connection, target, deadline, clock):
    try:
        connection.request("GET", target, headers=UA)
        _set_connection_timeout(
            connection,
            _deadline_remaining(deadline, clock),
        )
        response = connection.getresponse()
        _deadline_remaining(deadline, clock)
        if 300 <= response.status < 400:
            raise http.client.HTTPException(
                "redirect responses are not allowed for Hugging Face requests"
            )
        if not 200 <= response.status < 300:
            raise http.client.HTTPException(
                "Hugging Face returned HTTP status {0}".format(response.status)
            )
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except (TypeError, ValueError) as error:
                raise ValueError("invalid Hugging Face Content-Length") from error
            if declared_length < 0:
                raise ValueError("invalid Hugging Face Content-Length")
            if declared_length > HF_RESPONSE_MAX_BYTES:
                raise ValueError("Hugging Face response exceeds size limit")
        body = _read_bounded_body(
            response,
            connection,
            deadline,
            clock,
        )
        return json.loads(body.decode("utf-8"))
    finally:
        connection.close()


def http_card_text(
    url,
    timeout=8.0,
    connection_factory=None,
    clock=time.monotonic,
    completion_wait=None,
):
    """Read bounded README/model-card text from the fixed card host."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != HF_CARD_HOST:
        raise ValueError("card URL must use the fixed HTTPS card host")
    if parsed.port not in (None, 443):
        raise ValueError("card URL must use the default HTTPS port")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("card URL must not contain credentials")
    if parsed.fragment:
        raise ValueError("card URL must not contain a fragment")
    if not parsed.path.endswith(_CARD_PATH_SUFFIX) or parsed.path.count("/") != 5:
        raise ValueError("card URL must target <owner>/<repo>/raw/main/README.md")

    deadline = clock() + timeout
    remaining = _deadline_remaining(deadline, clock)
    if connection_factory is None:
        connection = http.client.HTTPSConnection(HF_CARD_HOST, 443, timeout=remaining)
    else:
        connection = connection_factory(HF_CARD_HOST, 443, remaining)
    target = parsed.path
    if parsed.query:
        target = "{0}?{1}".format(target, parsed.query)
    return _run_http_worker(
        connection,
        lambda: _http_text_operation(connection, target, deadline, clock),
        deadline,
        clock,
        completion_wait,
    )


def _http_text_operation(connection, target, deadline, clock):
    try:
        connection.request("GET", target, headers=UA)
        _set_connection_timeout(connection, _deadline_remaining(deadline, clock))
        response = connection.getresponse()
        _deadline_remaining(deadline, clock)
        if 300 <= response.status < 400:
            raise http.client.HTTPException(
                "redirect responses are not allowed for card requests"
            )
        if not 200 <= response.status < 300:
            raise http.client.HTTPException(
                "card host returned HTTP status {0}".format(response.status)
            )
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except (TypeError, ValueError) as error:
                raise ValueError("invalid card Content-Length") from error
            if declared_length < 0 or declared_length > MODEL_CARD_MAX_BYTES:
                raise ValueError("card response exceeds size limit")
        body = _read_bounded_card_body(response, connection, deadline, clock)
        return body.decode("utf-8", errors="replace")
    finally:
        connection.close()


def _read_bounded_card_body(response, connection, deadline, clock):
    chunks = []
    total = 0
    while True:
        _set_connection_timeout(connection, _deadline_remaining(deadline, clock))
        chunk = response.read(
            min(_HTTP_READ_CHUNK_BYTES, MODEL_CARD_MAX_BYTES - total + 1)
        )
        _deadline_remaining(deadline, clock)
        if not chunk:
            break
        total += len(chunk)
        if total > MODEL_CARD_MAX_BYTES:
            raise ValueError("card response exceeds size limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _read_bounded_body(response, connection, deadline, clock):
    chunks = []
    total = 0
    while True:
        _set_connection_timeout(
            connection,
            _deadline_remaining(deadline, clock),
        )
        chunk = response.read(
            min(_HTTP_READ_CHUNK_BYTES, HF_RESPONSE_MAX_BYTES - total + 1)
        )
        _deadline_remaining(deadline, clock)
        if not chunk:
            break
        total += len(chunk)
        if total > HF_RESPONSE_MAX_BYTES:
            raise ValueError("Hugging Face response exceeds size limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _deadline_remaining(deadline, clock):
    remaining = deadline - clock()
    if remaining <= 0:
        raise TimeoutError("Hugging Face request exceeded its overall deadline")
    return remaining


def _set_connection_timeout(connection, timeout):
    sock = getattr(connection, "sock", None)
    if sock is not None:
        sock.settimeout(timeout)


class HuggingFaceClient:
    def __init__(self, http_get=http_json, card_get=http_card_text):
        self._http_get = http_get
        self._card_get = card_get

    @property
    def http_get(self):
        return self._http_get

    def fetch_model_card(self, repo, timeout=8):
        """Return bounded README/model-card text, or None on any failure."""
        quoted = "/".join(urllib.parse.quote(part) for part in repo.split("/"))
        url = "https://{0}/{1}/raw/main/README.md".format(HF_CARD_HOST, quoted)
        try:
            return self._card_get(url, timeout=timeout)
        except Exception:
            return None

    @staticmethod
    def list_models_url(sort="trendingScore", limit_fetch=300):
        query = urllib.parse.urlencode({"filter": "mlx", "sort": sort, "direction": "-1", "limit": limit_fetch})
        return "{0}?{1}".format(HF_API, query)

    def list_models(self, sort="trendingScore", limit_fetch=300):
        return self._http_get(self.list_models_url(sort=sort, limit_fetch=limit_fetch))

    def inspect_model_metadata(self, repo, timeout=8):
        """Inspect model metadata without fetching the recursive repository tree."""
        quoted = urllib.parse.quote(repo)
        model_url = "{0}/{1}".format(HF_API, quoted)
        tree_url = "{0}/{1}/tree/main?recursive=true".format(HF_API, quoted)
        output = {
            "weight_bytes": None,
            "tags": [],
            "gated": None,
            "license": None,
            "reasoning": None,
            "reason_src": None,
            "tool_use": None,
            "tool_use_src": None,
            "tool_use_confidence": "none",
            "params_total": None,
            "metadata_available": False,
            "tree_available": False,
            "metadata_url": model_url,
            "tree_url": tree_url,
            "repository_url": "https://huggingface.co/{0}".format(repo),
        }
        try:
            metadata = self._http_get(model_url, timeout=timeout)
            output["metadata_available"] = True
            tags = metadata.get("tags", []) or []
            output["tags"] = tags
            output["gated"] = bool(metadata.get("gated"))
            config = metadata.get("config") or {}
            card_data = metadata.get("cardData") or {}
            output["license"] = card_data.get("license") or next((tag.split("license:", 1)[1] for tag in tags if tag.startswith("license:")), None)
            output["params_total"] = (metadata.get("safetensors") or {}).get("total")
            template = ((config.get("tokenizer_config") or {}).get("chat_template") or "")
            lower_tags = [tag.lower() for tag in tags]
            normalized_tags = {
                re.sub(r"\s+", "-", str(tag).strip().lower())
                for tag in tags
            }
            if TEMPLATE_REASON.search(template):
                output["reasoning"], output["reason_src"] = True, "chat_template"
            elif any(tag in ("reasoning", "thinking", "chain-of-thought") for tag in lower_tags):
                output["reasoning"], output["reason_src"] = True, "tags"
            elif REASONER_HINTS.search(repo):
                output["reasoning"], output["reason_src"] = True, "name"
            else:
                output["reasoning"], output["reason_src"] = False, "checked"
            if TEMPLATE_TOOL_USE.search(template):
                output.update({
                    "tool_use": True,
                    "tool_use_src": "chat_template",
                    "tool_use_confidence": "explicit",
                })
            elif normalized_tags & TOOL_USE_TAGS:
                output.update({
                    "tool_use": True,
                    "tool_use_src": "tags",
                    "tool_use_confidence": "explicit",
                })
            elif TOOL_USE_HINTS.search(repo):
                output.update({
                    "tool_use": True,
                    "tool_use_src": "name",
                    "tool_use_confidence": "weak",
                })
            else:
                output.update({
                    "tool_use": False,
                    "tool_use_src": "checked",
                    "tool_use_confidence": "explicit",
                })
        except Exception:
            pass
        return output

    def inspect_model(self, repo):
        output = self.inspect_model_metadata(repo, timeout=8)
        tree_url = output["tree_url"]
        try:
            tree = self._http_get(tree_url, timeout=8)
            output["tree_available"] = True
            weights = sum(item.get("size", 0) for item in tree if item.get("path", "").endswith((".safetensors", ".gguf", ".bin")))
            output["weight_bytes"] = weights or None
        except Exception:
            pass
        return output

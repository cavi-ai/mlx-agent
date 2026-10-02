"""Dependency-free Hugging Face Hub access for Scout."""

import http.client
import json
import queue
import re
import struct
import threading
import time
import urllib.parse

from .contextfit import extract_architecture
from .models import (
    REASONER_HINTS,
    TEMPLATE_REASON,
    TEMPLATE_TOOL_USE,
    TOOL_USE_HINTS,
    TOOL_USE_TAGS,
)


HF_API = "https://huggingface.co/api/models"
HF_DATASETS_API = "https://huggingface.co/api/datasets"
HF_API_HOST = "huggingface.co"
UA = {"User-Agent": "mlx-scout/0.2 (+https://github.com/cavi-ai/mlx-agent)"}
HF_RESPONSE_MAX_BYTES = 8 * 1024 * 1024
_HTTP_READ_CHUNK_BYTES = 64 * 1024
HF_CARD_HOST = "huggingface.co"
MODEL_CARD_MAX_BYTES = 512 * 1024
_CARD_PATH_SUFFIX = "/raw/main/README.md"
RAW_TEXT_MAX_BYTES = 8 * 1024 * 1024
_RAW_JSON_FILES = frozenset({"config.json", "model.safetensors.index.json"})
_RAW_PY_FILE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\.py")
_RAW_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SAFETENSORS_FILE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\.safetensors")
SAFETENSORS_HEADER_MAX_BYTES = 16 * 1024 * 1024
_SAFETENSORS_PROBE_BYTES = 1024 * 1024


class HuggingFaceHTTPError(http.client.HTTPException):
    """A non-2xx Hugging Face response; carries the HTTP status."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _is_valid_raw_path(path):
    """Accept /<owner>/<repo>/raw/<revision>/<file> for intake files only."""
    parts = path.split("/")
    if len(parts) != 6 or parts[0] != "" or parts[3] != "raw":
        return False
    if not parts[1] or not parts[2] or not _RAW_REVISION.fullmatch(parts[4]):
        return False
    return parts[5] in _RAW_JSON_FILES or bool(_RAW_PY_FILE.fullmatch(parts[5]))


def _is_valid_resolve_path(path):
    """Accept /<owner>/<repo>/resolve/<revision>/<file>.safetensors for top-level weight files."""
    parts = path.split("/")
    if len(parts) != 6 or parts[0] != "" or parts[3] != "resolve":
        return False
    if not parts[1] or not parts[2] or not _RAW_REVISION.fullmatch(parts[4]):
        return False
    return bool(_SAFETENSORS_FILE.fullmatch(parts[5]))


def _is_hub_storage_host(host):
    """Hosts the Hub redirects weight downloads to (its CDN and Xet storage)."""
    host = (host or "").lower()
    return host == "hf.co" or host.endswith(".hf.co") or host.endswith(".huggingface.co")


def _is_allowed_api_path(path):
    if path == "/api/models" or path.startswith("/api/models/"):
        return True
    if path == "/api/datasets" or path.startswith("/api/datasets/"):
        return True
    return False


def _is_valid_card_path(path):
    """Accept model or dataset README raw paths on the fixed card host."""
    if not path.endswith(_CARD_PATH_SUFFIX):
        return False
    slash_count = path.count("/")
    if slash_count == 5 and not path.startswith("/datasets/"):
        return True
    if slash_count == 6 and path.startswith("/datasets/"):
        return True
    return False


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
    if not _is_allowed_api_path(parsed.path):
        raise ValueError("Hugging Face URL must target the models or datasets API")

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
            raise HuggingFaceHTTPError(
                response.status,
                "Hugging Face returned HTTP status {0}".format(response.status),
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
    if not _is_valid_card_path(parsed.path):
        raise ValueError(
            "card URL must target <owner>/<repo>/raw/main/README.md "
            "or datasets/<owner>/<repo>/raw/main/README.md"
        )

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


def http_raw_text(
    url,
    timeout=8.0,
    connection_factory=None,
    clock=time.monotonic,
    completion_wait=None,
):
    """Read one bounded intake file (config, index, or top-level .py) as text."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != HF_CARD_HOST:
        raise ValueError("raw URL must use the fixed HTTPS host")
    if parsed.port not in (None, 443):
        raise ValueError("raw URL must use the default HTTPS port")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("raw URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("raw URL must not contain a query or fragment")
    if not _is_valid_raw_path(parsed.path):
        raise ValueError("raw URL must target config.json, the safetensors index, or a top-level .py file")

    deadline = clock() + timeout
    remaining = _deadline_remaining(deadline, clock)
    if connection_factory is None:
        connection = http.client.HTTPSConnection(HF_CARD_HOST, 443, timeout=remaining)
    else:
        connection = connection_factory(HF_CARD_HOST, 443, remaining)
    return _run_http_worker(
        connection,
        lambda: _http_text_operation(
            connection, parsed.path, deadline, clock, max_bytes=RAW_TEXT_MAX_BYTES
        ),
        deadline,
        clock,
        completion_wait,
    )


def _http_text_operation(connection, target, deadline, clock, max_bytes=MODEL_CARD_MAX_BYTES):
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
            raise HuggingFaceHTTPError(
                response.status,
                "card host returned HTTP status {0}".format(response.status),
            )
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except (TypeError, ValueError) as error:
                raise ValueError("invalid card Content-Length") from error
            if declared_length < 0 or declared_length > max_bytes:
                raise ValueError("card response exceeds size limit")
        body = _read_bounded_body(
            response,
            connection,
            deadline,
            clock,
            max_bytes=max_bytes,
        )
        return body.decode("utf-8", errors="replace")
    finally:
        connection.close()


def http_safetensors_header(
    url,
    timeout=8.0,
    connection_factory=None,
    clock=time.monotonic,
    completion_wait=None,
):
    """Read only the JSON header of one safetensors file through ranged GETs.

    The Hub answers with one redirect to its storage host; that redirect is
    followed only to a Hub storage host over HTTPS, and every body is bounded
    by the requested range, so no weights are downloaded.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != HF_CARD_HOST:
        raise ValueError("safetensors URL must use the fixed HTTPS host")
    if parsed.port not in (None, 443):
        raise ValueError("safetensors URL must use the default HTTPS port")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("safetensors URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("safetensors URL must not contain a query or fragment")
    if not _is_valid_resolve_path(parsed.path):
        raise ValueError("safetensors URL must target a top-level .safetensors file")

    deadline = clock() + timeout

    def ranged(host, target, start, end):
        remaining = _deadline_remaining(deadline, clock)
        if connection_factory is None:
            connection = http.client.HTTPSConnection(host, 443, timeout=remaining)
        else:
            connection = connection_factory(host, 443, remaining)
        return _run_http_worker(
            connection,
            lambda: _http_range_operation(connection, target, start, end, deadline, clock),
            deadline,
            clock,
            completion_wait,
        )

    host, target = HF_CARD_HOST, parsed.path
    status, location, body = ranged(host, target, 0, _SAFETENSORS_PROBE_BYTES - 1)
    if location is not None:
        redirected = urllib.parse.urlsplit(location)
        if redirected.scheme != "https" or not _is_hub_storage_host(redirected.hostname):
            raise ValueError("safetensors redirect must stay on a Hub storage host")
        if redirected.port not in (None, 443) or redirected.username is not None or redirected.password is not None:
            raise ValueError("safetensors redirect must use the default HTTPS port without credentials")
        host = redirected.hostname
        target = redirected.path + ("?" + redirected.query if redirected.query else "")
        status, location, body = ranged(host, target, 0, _SAFETENSORS_PROBE_BYTES - 1)
        if location is not None:
            raise http.client.HTTPException("safetensors requests follow at most one redirect")
    if len(body) < 8:
        raise ValueError("safetensors file is shorter than its header length")
    (length,) = struct.unpack("<Q", body[:8])
    if length > SAFETENSORS_HEADER_MAX_BYTES:
        raise ValueError("safetensors header exceeds size limit")
    if len(body) < 8 + length:
        status, location, rest = ranged(host, target, len(body), 8 + length - 1)
        if location is not None:
            raise http.client.HTTPException("safetensors requests follow at most one redirect")
        body += rest
    header = json.loads(body[8: 8 + length].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    return header


def _http_range_operation(connection, target, start, end, deadline, clock):
    """One ranged GET: (status, redirect location or None, bounded body)."""
    try:
        headers = dict(UA)
        headers["Range"] = "bytes={0}-{1}".format(start, end)
        connection.request("GET", target, headers=headers)
        _set_connection_timeout(connection, _deadline_remaining(deadline, clock))
        response = connection.getresponse()
        _deadline_remaining(deadline, clock)
        if 300 <= response.status < 400:
            location = response.getheader("Location")
            if not location:
                raise http.client.HTTPException("redirect without a Location header")
            return response.status, location, b""
        if response.status not in (200, 206):
            raise HuggingFaceHTTPError(
                response.status,
                "safetensors host returned HTTP status {0}".format(response.status),
            )
        body = _read_bounded_body(response, connection, deadline, clock, max_bytes=end - start + 1)
        return response.status, None, body
    finally:
        connection.close()


def _read_bounded_body(response, connection, deadline, clock, max_bytes=HF_RESPONSE_MAX_BYTES):
    chunks = []
    total = 0
    while True:
        _set_connection_timeout(
            connection,
            _deadline_remaining(deadline, clock),
        )
        chunk = response.read(
            min(_HTTP_READ_CHUNK_BYTES, max_bytes - total + 1)
        )
        _deadline_remaining(deadline, clock)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
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
    def __init__(self, http_get=http_json, card_get=http_card_text, raw_get=http_raw_text,
                 header_get=http_safetensors_header):
        self._http_get = http_get
        self._card_get = card_get
        self._raw_get = raw_get
        self._header_get = header_get

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

    def fetch_dataset_card(self, repo, timeout=8):
        """Return bounded dataset README text, or None on any failure."""
        quoted = "/".join(urllib.parse.quote(part) for part in repo.split("/"))
        url = "https://{0}/datasets/{1}/raw/main/README.md".format(HF_CARD_HOST, quoted)
        try:
            return self._card_get(url, timeout=timeout)
        except Exception:
            return None

    def fetch_model_info(self, repo, revision="main", timeout=8):
        """Model API document with per-file sizes; raises on any failure."""
        quoted = "/".join(urllib.parse.quote(part) for part in repo.split("/"))
        if revision == "main":
            url = "{0}/{1}?blobs=true".format(HF_API, quoted)
        else:
            url = "{0}/{1}/revision/{2}?blobs=true".format(
                HF_API, quoted, urllib.parse.quote(revision)
            )
        return self._http_get(url, timeout=timeout)

    def fetch_raw_text(self, repo, revision, filename, timeout=8):
        """One intake file as text; raises on any failure."""
        quoted = "/".join(urllib.parse.quote(part) for part in repo.split("/"))
        url = "https://{0}/{1}/raw/{2}/{3}".format(
            HF_CARD_HOST, quoted, urllib.parse.quote(revision), urllib.parse.quote(filename)
        )
        return self._raw_get(url, timeout=timeout)

    def fetch_safetensors_header(self, repo, revision, filename, timeout=8):
        """One weight file's tensor header (names, dtypes, shapes); raises on any failure."""
        quoted = "/".join(urllib.parse.quote(part) for part in repo.split("/"))
        url = "https://{0}/{1}/resolve/{2}/{3}".format(
            HF_CARD_HOST, quoted, urllib.parse.quote(revision), urllib.parse.quote(filename)
        )
        return self._header_get(url, timeout=timeout)

    @staticmethod
    def list_models_url(sort="trendingScore", limit_fetch=300):
        query = urllib.parse.urlencode({"filter": "mlx", "sort": sort, "direction": "-1", "limit": limit_fetch})
        return "{0}?{1}".format(HF_API, query)

    @staticmethod
    def list_adapters_url(search="", limit_fetch=20):
        params = {
            "filter": "peft",
            "sort": "downloads",
            "direction": "-1",
            "limit": limit_fetch,
        }
        if search:
            params["search"] = search
        return "{0}?{1}".format(HF_API, urllib.parse.urlencode(params))

    @staticmethod
    def list_datasets_url(search="", limit_fetch=20):
        params = {
            "sort": "downloads",
            "direction": "-1",
            "limit": limit_fetch,
        }
        if search:
            params["search"] = search
        return "{0}?{1}".format(HF_DATASETS_API, urllib.parse.urlencode(params))

    def list_models(self, sort="trendingScore", limit_fetch=300):
        return self._http_get(self.list_models_url(sort=sort, limit_fetch=limit_fetch))

    def list_adapters(self, search="", limit_fetch=20, timeout=10):
        """List PEFT/LoRA adapter model rows from the Hub (read-only)."""
        rows = self._http_get(
            self.list_adapters_url(search=search, limit_fetch=limit_fetch),
            timeout=timeout,
        )
        return rows if isinstance(rows, list) else []

    def list_datasets(self, search="", limit_fetch=20, timeout=10):
        """List dataset rows from the Hub (read-only)."""
        rows = self._http_get(
            self.list_datasets_url(search=search, limit_fetch=limit_fetch),
            timeout=timeout,
        )
        return rows if isinstance(rows, list) else []

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
            "architecture": None,
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
            output["architecture"] = extract_architecture(config)
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

"""Deterministic, injection-safe configuration renderers for Wire."""

from __future__ import annotations

import ipaddress
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


_RUNTIMES = {"ollama", "lmstudio", "mlx_lm", "mlx-vlm", "litellm"}
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_SECRET_NAMES = {
    "accesskey", "accesstoken", "apikey", "auth", "authorization", "clientsecret", "credential",
    "credentials", "idtoken", "passwd", "password", "privatekey",
    "refreshtoken", "secret", "secretkey", "token",
}
_ASSIGNMENT_SECRET = re.compile(
    r"(?im)(^|[\s,{&?;])([\"']?(?:"
    r"[a-z0-9_-]*(?:token|secret|passw(?:or)?d|authorization|credentials?|auth)|"
    r"[a-z0-9_-]*(?:api|access|private)[_-]?key"
    r")[\"']?\s*[:=]\s*)"
    r"([^\s,}\]\r\n]+|\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*')"
)
_URL_USERINFO = re.compile(r"(https?://)([^/@\s]+@)", re.I)
_OLLAMA = re.compile(r"\A# MLX_AGENT_WIRE v1\nFROM (" + _MODEL.pattern[1:-1] + r")\n\Z")
_ENV_REFERENCE = re.compile(
    r"^(?:os\.environ(?:\[[\"'][A-Z][A-Z0-9_]{0,127}[\"']\]|/[A-Z][A-Z0-9_]{0,127})|"
    r"\$\{[A-Z][A-Z0-9_]{0,127}\}|\$[A-Z][A-Z0-9_]{0,127}|env:[A-Z][A-Z0-9_]{0,127})$"
)
_STATE_CHANGING_PATH_PARTS = {
    "admin", "create", "delete", "install", "kill", "mutate", "pull", "push",
    "reload", "remove", "reset", "restart", "shutdown", "update", "write",
}


def _secret_name(name):
    raw = str(name).lower()
    collapsed = re.sub(r"[^a-z0-9]", "", raw)
    if collapsed in _SECRET_NAMES:
        return True
    segments = [item for item in re.split(r"[^a-z0-9]+", raw) if item]
    if any(
        item in {"auth", "authorization", "credential", "credentials", "passwd", "password", "secret", "token"}
        for item in segments
    ):
        return True
    return collapsed.endswith(
        ("accesskey", "apikey", "authorization", "credential", "credentials", "passwd", "password", "privatekey", "secret", "token")
    )


def _environment_reference(value):
    return isinstance(value, str) and _ENV_REFERENCE.fullmatch(value.strip()) is not None


def _json_contains_resolved_secret(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if _secret_name(key) and not _environment_reference(item):
                return True
            if _json_contains_resolved_secret(item):
                return True
        return False
    if isinstance(value, list):
        return any(_json_contains_resolved_secret(item) for item in value)
    if isinstance(value, str):
        try:
            parsed = urllib.parse.urlsplit(value)
        except ValueError:
            return False
        if parsed.scheme not in {"http", "https"}:
            return False
        if parsed.username is not None or parsed.password is not None:
            return True
        return any(
            _secret_name(key) and not _environment_reference(item)
            for key, item in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        )
    return False


def contains_resolved_secrets(content):
    """Return whether configuration text contains a resolved credential value."""
    if not isinstance(content, str):
        raise TypeError("configuration content must be text")
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if parsed is not None and _json_contains_resolved_secret(parsed):
        return True
    if _URL_USERINFO.search(content):
        return True
    for match in _ASSIGNMENT_SECRET.finditer(content):
        raw = match.group(3).strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"\"", "'"}:
            raw = raw[1:-1]
        if not _environment_reference(raw):
            return True
    return False


def require_secret_free_config(content):
    """Fail closed without echoing a resolved secret-bearing value."""
    if contains_resolved_secrets(content):
        raise ValueError(
            "existing configuration contains resolved secret-bearing fields; "
            "move secrets to environment references or a secret-free managed fragment"
        )
    return content


def _redact_json(value):
    if isinstance(value, dict):
        return {key: "<redacted>" if _secret_name(key) else _redact_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_json(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"https?://[^\s\"']+", lambda match: redact_endpoint(match.group(0)), value)
    return value


def redact_endpoint(endpoint):
    """Return only a non-credential origin/path representation of an endpoint."""
    try:
        parsed = urllib.parse.urlsplit(str(endpoint))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return "<invalid-endpoint>"
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = "[{0}]".format(host)
        origin = "{0}://{1}".format(parsed.scheme, host)
        if parsed.port is not None:
            origin += ":{0}".format(parsed.port)
        return origin + (parsed.path or "/")
    except ValueError:
        return "<invalid-endpoint>"


def redact_secrets(content):
    """Redact common credential forms before they can enter previews or receipts."""
    text = str(content)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if parsed is not None:
        return json.dumps(_redact_json(parsed), sort_keys=True, separators=(",", ": "))
    text = _URL_USERINFO.sub(r"\1<redacted>@", text)
    def redact_assignment(match):
        value = match.group(3)
        replacement = value if "<redacted>" in value else "<redacted>"
        return match.group(1) + match.group(2) + replacement
    text = _ASSIGNMENT_SECRET.sub(redact_assignment, text)
    def redact_url(match):
        return redact_endpoint(match.group(0))
    text = re.sub(r"https?://[^\s\"']+", redact_url, text)
    return text


def validate_health_endpoint(endpoint):
    """Validate one credential-free, read-only loopback HTTP(S) health URL."""
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("health endpoint must be a non-empty URL")
    try:
        parsed = urllib.parse.urlsplit(endpoint)
        port = parsed.port
    except ValueError as error:
        raise ValueError("health endpoint has an invalid port") from error
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("health endpoint must use HTTP(S)")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("health endpoint must not contain credentials, query, or fragment")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("health endpoint port is outside 1-65535")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname != "localhost":
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError as error:
            raise ValueError("health endpoint must use localhost or a loopback IP literal") from error
        if not address.is_loopback:
            raise ValueError("health endpoint must resolve only to loopback")
    decoded_path = urllib.parse.unquote(parsed.path or "/").lower()
    parts = {part for part in decoded_path.split("/") if part}
    if parts & _STATE_CHANGING_PATH_PARTS:
        raise ValueError("health endpoint path appears state-changing")
    return endpoint


def _endpoint_origin(endpoint):
    parsed = urllib.parse.urlsplit(validate_health_endpoint(endpoint))
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.scheme.lower(), parsed.hostname.lower().rstrip("."), port


class _SameLoopbackOriginRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, origin):
        super().__init__()
        self.origin = origin

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        destination = urllib.parse.urljoin(request.full_url, new_url)
        if _endpoint_origin(destination) != self.origin:
            raise urllib.error.URLError("health redirect left the reviewed loopback origin")
        return super().redirect_request(
            request, file_pointer, code, message, headers, destination
        )


class ConfigAdapter:
    """A small format adapter using only documented, exact stdlib-safe subsets."""

    version = "2.0"

    def __init__(self, runtime, path=None):
        if runtime not in _RUNTIMES:
            raise ValueError("unsupported runtime: {0}".format(runtime))
        self.runtime = runtime
        self.path = Path(path) if path is not None else None

    @classmethod
    def detect(cls, path, runtime=None):
        location = Path(path)
        selected = runtime
        if selected is None:
            name = location.name.lower()
            if name == "modelfile" or location.suffix.lower() == ".modelfile":
                selected = "ollama"
            elif location.suffix.lower() in {".yaml", ".yml"}:
                selected = "litellm"
            elif "lmstudio" in name:
                selected = "lmstudio"
            else:
                selected = "mlx_lm"
        return cls(selected, location)

    def render(self, model, runtime=None, existing=""):
        selected = runtime or self.runtime
        if selected != self.runtime:
            return ConfigAdapter(selected, self.path).render(model, existing=existing)
        self._validate_model(model)
        if not isinstance(existing, str):
            raise TypeError("existing config content must be text")
        require_secret_free_config(existing)
        if selected == "ollama":
            if existing and existing != self._render_ollama(model):
                raise ValueError("Ollama Wire accepts only its exact managed configuration")
            return self._render_ollama(model)
        if selected == "litellm":
            if existing:
                self._validate_litellm(existing)
            return self._render_litellm(model)
        return self._render_json(model, existing)

    @staticmethod
    def _validate_model(model):
        if not isinstance(model, str) or not _MODEL.fullmatch(model):
            raise ValueError("model must be a safe publisher/model identifier")

    def _render_json(self, model, existing):
        if existing.strip():
            try:
                document = json.loads(existing)
            except json.JSONDecodeError as error:
                raise ValueError("existing JSON configuration is invalid: {0}".format(error))
            if not isinstance(document, dict):
                raise ValueError("existing JSON configuration must be an object")
        else:
            document = {}
        port = 8081 if self.runtime == "mlx-vlm" else 8080
        provider_id = "mlxvlm" if self.runtime == "mlx-vlm" else self.runtime.replace("_", "")
        document["mlx_agent_wire"] = {
            "marker": "MLX_AGENT_WIRE",
            "version": self.version,
            "runtime": self.runtime,
            "model": model,
            "provider": {
                "id": provider_id,
                "type": "openai",
                "base_url": "http://127.0.0.1:{0}/v1".format(port),
                "api_key_env": "MLX_AGENT_LOCAL_API_KEY",
            },
        }
        return json.dumps(document, indent=2, sort_keys=True) + "\n"

    @staticmethod
    def _render_ollama(model):
        return "# MLX_AGENT_WIRE v1\nFROM {0}\n".format(model)

    @staticmethod
    def _render_litellm(model):
        short = model.split("/", 1)[1].lower()
        return (
            "# MLX_AGENT_WIRE v1\n"
            "model_list:\n"
            "  - model_name: {0}\n"
            "    litellm_params:\n"
            "      model: openai/{1}\n"
            "      api_base: http://127.0.0.1:8080/v1\n"
            "      api_key: os.environ/MLX_AGENT_LOCAL_API_KEY\n"
        ).format(short, model)

    def validate(self, content):
        if not isinstance(content, str):
            raise TypeError("configuration content must be text")
        require_secret_free_config(content)
        if self.runtime in {"lmstudio", "mlx_lm", "mlx-vlm"}:
            value = json.loads(content)
            if not isinstance(value, dict):
                raise ValueError("JSON configuration must be an object")
            return True
        if self.runtime == "ollama":
            match = _OLLAMA.fullmatch(content)
            if match is None:
                raise ValueError("Ollama configuration must match the exact Wire grammar")
            self._validate_model(match.group(1))
            return True
        self._validate_litellm(content)
        return True

    @staticmethod
    def _validate_litellm(content):
        lines = content.splitlines()
        if len(lines) != 7 or lines[0] != "# MLX_AGENT_WIRE v1" or lines[1] != "model_list:":
            raise ValueError("LiteLLM configuration must use the supported Wire YAML subset")
        model_name = re.fullmatch(r"  - model_name: ([A-Za-z0-9][A-Za-z0-9._-]{0,191})", lines[2])
        model = re.fullmatch(r"      model: openai/(" + _MODEL.pattern[1:-1] + r")", lines[4])
        if lines[3] != "    litellm_params:" or lines[5] != "      api_base: http://127.0.0.1:8080/v1" or lines[6] != "      api_key: os.environ/MLX_AGENT_LOCAL_API_KEY" or not model_name or not model:
            raise ValueError("LiteLLM configuration must use the supported Wire YAML subset")
        ConfigAdapter._validate_model(model.group(1))
        if model_name.group(1) != model.group(1).split("/", 1)[1].lower():
            raise ValueError("LiteLLM model_name does not match the model identifier")

    @staticmethod
    def health_check(endpoint, timeout=2.0):
        if not endpoint:
            return True
        try:
            endpoint = validate_health_endpoint(endpoint)
            request = urllib.request.Request(endpoint, method="GET")
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}),
                _SameLoopbackOriginRedirect(_endpoint_origin(endpoint))
            )
            with opener.open(request, timeout=timeout) as response:
                return 200 <= response.status < 400
        except (OSError, ValueError, urllib.error.URLError):
            return False

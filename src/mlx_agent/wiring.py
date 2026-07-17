"""Deterministic, injection-safe configuration renderers for Wire."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


_RUNTIMES = {"ollama", "lmstudio", "mlx_lm", "mlx-vlm", "litellm"}
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_SECRET_NAMES = {"apikey", "token", "secret", "authorization"}
_ASSIGNMENT_SECRET = re.compile(
    r"(?im)(^|[\s,{&])([\"']?(?:api[_-]?key|token|secret|authorization)[\"']?\s*[:=]\s*)([^\s,}\]\r\n]+|\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*')"
)
_URL_USERINFO = re.compile(r"(https?://)([^/@\s]+@)", re.I)
_OLLAMA = re.compile(r"\A# MLX_AGENT_WIRE v1\nFROM (" + _MODEL.pattern[1:-1] + r")\n\Z")


def _secret_name(name):
    return re.sub(r"[_-]", "", str(name)).lower() in _SECRET_NAMES


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
            parsed = urllib.parse.urlsplit(endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                return False
            request = urllib.request.Request(endpoint, method="GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return 200 <= response.status < 400
        except (OSError, ValueError, urllib.error.URLError):
            return False

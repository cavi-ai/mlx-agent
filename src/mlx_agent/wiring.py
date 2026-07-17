"""Deterministic, dependency-free configuration renderers for Wire."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path


_RUNTIMES = {"ollama", "lmstudio", "mlx_lm", "mlx-vlm", "litellm"}
_SECRET_KEY = r"(?:api[_-]?key|token|secret|authorization)"
_JSON_SECRET = re.compile(r'("' + _SECRET_KEY + r'"\s*:\s*)"(?:\\.|[^"\\])*"', re.I)
_YAML_SECRET = re.compile(r'(^\s*' + _SECRET_KEY + r'\s*:\s*)([^#\r\n]+)', re.I | re.M)
_HEADER_SECRET = re.compile(r'(authorization\s*[:=]\s*)([^\s,;]+(?:\s+[^\s,;]+)?)', re.I)
_QUERY_SECRET = re.compile(r'([?&]' + _SECRET_KEY + r'=)([^&#\r\n]*)', re.I)


def redact_secrets(content):
    """Remove values associated with common credential keys from display data."""
    text = str(content)
    text = _JSON_SECRET.sub(r'\1"<redacted>"', text)
    text = _YAML_SECRET.sub(r'\1<redacted>', text)
    text = _HEADER_SECRET.sub(r'\1<redacted>', text)
    return _QUERY_SECRET.sub(r'\1<redacted>', text)


class ConfigAdapter:
    """A small format adapter with no provider-specific runtime dependency."""

    version = "1.0"

    def __init__(self, runtime, path=None):
        if runtime not in _RUNTIMES:
            raise ValueError("unsupported runtime: {0}".format(runtime))
        self.runtime = runtime
        self.path = Path(path) if path is not None else None

    @classmethod
    def detect(cls, path, runtime=None):
        """Detect a safe adapter from an explicit runtime or conventional path."""
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
        """Render a stable managed block without resolving or storing secrets."""
        selected = runtime or self.runtime
        if selected != self.runtime:
            return ConfigAdapter(selected, self.path).render(model, existing=existing)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(existing, str):
            raise TypeError("existing config content must be text")
        if selected == "ollama":
            return self._render_ollama(model, existing)
        if selected == "litellm":
            return self._render_litellm(model, existing)
        return self._render_json(model, existing)

    def _render_json(self, model, existing):
        source = existing.strip()
        if source:
            try:
                document = json.loads(source)
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

    def _render_ollama(self, model, existing):
        clean = self._remove_marked_block(existing, "# MLX_AGENT_WIRE BEGIN", "# MLX_AGENT_WIRE END")
        if clean and not clean.endswith("\n"):
            clean += "\n"
        return clean + "# MLX_AGENT_WIRE BEGIN\n# runtime: ollama\nFROM {0}\n# MLX_AGENT_WIRE END\n".format(model)

    def _render_litellm(self, model, existing):
        self._validate_yaml(existing) if existing.strip() else None
        clean = self._remove_marked_block(existing, "# MLX_AGENT_WIRE BEGIN", "# MLX_AGENT_WIRE END")
        if clean and not clean.endswith("\n"):
            clean += "\n"
        name = model.split("/")[-1].lower()
        block = (
            "# MLX_AGENT_WIRE BEGIN\n"
            "# runtime: litellm\n"
            "model_list:\n"
            "  - model_name: {0}\n"
            "    litellm_params:\n"
            "      model: openai/{1}\n"
            "      api_base: http://127.0.0.1:8080/v1\n"
            "      api_key: os.environ/MLX_AGENT_LOCAL_API_KEY\n"
            "# MLX_AGENT_WIRE END\n"
        ).format(name, model)
        return clean + block

    @staticmethod
    def _remove_marked_block(content, start, end):
        expression = re.compile(re.escape(start) + r".*?" + re.escape(end) + r"\s*", re.S)
        return expression.sub("", content)

    def validate(self, content):
        """Parse or conservatively validate a renderer's target format."""
        if not isinstance(content, str):
            raise TypeError("configuration content must be text")
        if self.runtime in {"lmstudio", "mlx_lm", "mlx-vlm"}:
            value = json.loads(content)
            if not isinstance(value, dict):
                raise ValueError("JSON configuration must be an object")
            return True
        if self.runtime == "ollama":
            if "# MLX_AGENT_WIRE BEGIN" not in content or not re.search(r"^FROM\s+\S+", content, re.M):
                raise ValueError("Ollama configuration does not contain a Wire FROM directive")
            return True
        self._validate_yaml(content)
        if "# MLX_AGENT_WIRE BEGIN" not in content or "model_list:" not in content:
            raise ValueError("LiteLLM configuration does not contain a Wire model_list block")
        return True

    @staticmethod
    def _validate_yaml(content):
        if "\t" in content:
            raise ValueError("YAML configuration must not use tabs")
        if content.count("[") != content.count("]") or content.count("{") != content.count("}"):
            raise ValueError("YAML configuration has unbalanced flow delimiters")
        for number, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("-"):
                continue
            if ":" not in stripped:
                raise ValueError("YAML configuration line {0} has no mapping separator".format(number))
        return True

    @staticmethod
    def health_check(endpoint, timeout=2.0):
        """Return a boolean only; endpoint credentials are never logged or persisted."""
        if not endpoint:
            return True
        request = urllib.request.Request(endpoint, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return 200 <= response.status < 400
        except (OSError, ValueError, urllib.error.URLError):
            return False

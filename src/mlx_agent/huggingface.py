"""Dependency-free Hugging Face Hub access for Scout."""

import json
import urllib.parse
import urllib.request

from .models import REASONER_HINTS, TEMPLATE_REASON


HF_API = "https://huggingface.co/api/models"
UA = {"User-Agent": "mlx-scout/0.2 (+https://github.com/sasan1200/mlx-agent)"}


def http_json(url, timeout=10.0):
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class HuggingFaceClient:
    def __init__(self, http_get=http_json):
        self._http_get = http_get

    @property
    def http_get(self):
        return self._http_get

    def list_models(self, sort="trendingScore", limit_fetch=300):
        query = urllib.parse.urlencode({"filter": "mlx", "sort": sort, "direction": "-1", "limit": limit_fetch})
        return self._http_get("{0}?{1}".format(HF_API, query))

    def inspect_model(self, repo):
        output = {"weight_bytes": None, "tags": [], "gated": False, "license": None, "reasoning": None, "reason_src": None, "params_total": None}
        quoted = urllib.parse.quote(repo)
        try:
            metadata = self._http_get("{0}/{1}".format(HF_API, quoted), timeout=8)
            tags = metadata.get("tags", []) or []
            output["tags"] = tags
            output["gated"] = bool(metadata.get("gated"))
            config = metadata.get("config") or {}
            card_data = metadata.get("cardData") or {}
            output["license"] = card_data.get("license") or next((tag.split("license:", 1)[1] for tag in tags if tag.startswith("license:")), None)
            output["params_total"] = (metadata.get("safetensors") or {}).get("total")
            template = ((config.get("tokenizer_config") or {}).get("chat_template") or "")
            lower_tags = [tag.lower() for tag in tags]
            if TEMPLATE_REASON.search(template):
                output["reasoning"], output["reason_src"] = True, "chat_template"
            elif any(tag in ("reasoning", "thinking", "chain-of-thought") for tag in lower_tags):
                output["reasoning"], output["reason_src"] = True, "tags"
            elif REASONER_HINTS.search(repo):
                output["reasoning"], output["reason_src"] = True, "name"
            else:
                output["reasoning"], output["reason_src"] = False, "checked"
        except Exception:
            pass
        try:
            tree = self._http_get("{0}/{1}/tree/main?recursive=true".format(HF_API, quoted), timeout=8)
            weights = sum(item.get("size", 0) for item in tree if item.get("path", "").endswith((".safetensors", ".gguf", ".bin")))
            output["weight_bytes"] = weights or None
        except Exception:
            pass
        return output

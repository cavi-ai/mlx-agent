"""Model discovery, ranking, and structured Scout results."""

from dataclasses import dataclass

from .contracts import ResultEnvelope
from .host import HostInventory
from .huggingface import HuggingFaceClient
from .models import REPUTABLE, ROLES, base_name, classify, quant_rank, resolve_ram, resolve_reasoning, wiring


@dataclass(frozen=True)
class DiscoveryRequest:
    limit: int = 6
    role: object = None
    new: bool = False
    fast: bool = False


class DiscoveryService:
    def __init__(self, host=None, huggingface=None):
        self._host = host
        self._huggingface = huggingface or HuggingFaceClient()

    @property
    def host(self):
        if self._host is None:
            self._host = HostInventory.detect(self._huggingface.http_get)
        return self._host

    def discover(self, request):
        host = self.host
        try:
            raw = self._huggingface.list_models(sort="lastModified" if request.new else "trendingScore")
        except Exception as error:
            return ResultEnvelope.fail("discover", "network_unavailable", "HuggingFace query failed: {0}".format(error), "Check network access to huggingface.co and retry the discovery command.", retryable=True)

        buckets = {role: [] for role, _keywords, _label in ROLES}
        seen_repo, seen_base = set(), {}
        for model in raw:
            repo = model.get("id") or model.get("modelId")
            if not repo or repo in seen_repo:
                continue
            seen_repo.add(repo)
            role = classify(repo)
            if request.role and role != request.role:
                continue
            item = {"repo": repo, "downloads": model.get("downloads", 0), "likes": model.get("likes", 0), "trusted": repo.split("/")[0].lower() in REPUTABLE, "base": base_name(repo), "qrank": quant_rank(repo)}
            key = (role, item["base"])
            if key in seen_base:
                current = seen_base[key]
                if (item["trusted"], item["qrank"], item["downloads"]) > (current["trusted"], current["qrank"], current["downloads"]):
                    buckets[role].remove(current)
                    buckets[role].append(item)
                    seen_base[key] = item
                continue
            seen_base[key] = item
            buckets[role].append(item)

        for role in buckets:
            buckets[role].sort(key=lambda item: (item["trusted"], item["downloads"], item["likes"]), reverse=True)
            buckets[role] = buckets[role][:request.limit]

        host_data = host.to_dict()
        for role, items in buckets.items():
            for item in items:
                enrichment = {} if request.fast else self._huggingface.inspect_model(item["repo"])
                ram, ram_source = resolve_ram(item["repo"], enrichment)
                reasoning, reason_source = resolve_reasoning(item["repo"], enrichment)
                item["est_ram_gb"] = ram
                item["ram_src"] = "est" if request.fast else ram_source
                item["fits"] = ram is None or host_data.get("ram_gb") is None or ram < host_data["ram_gb"] * 0.8
                item["reasoning"] = reasoning
                item["reason_src"] = reason_source
                item["gated"] = enrichment.get("gated", False)
                item["license"] = enrichment.get("license")
                item["wiring"] = wiring(item["repo"], role, host_data)

        return ResultEnvelope.ok("discover", {"host": host_data, "fast": request.fast, "roles": {role: items for role, items in buckets.items() if items}})

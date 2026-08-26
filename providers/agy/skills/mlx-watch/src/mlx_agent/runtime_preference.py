"""Justified MLX-native runtime preference for research packs and discovery wiring.

Reporting only: does not change scoring. Uses existing HostInventory flags
(ollama, lmstudio) — no new runtime probes. Ollama remains a valid alternate
for text workloads and is never removed from wire/CLI surfaces.

This module stays free of interview/models imports to avoid cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple


_VISION_MODALITIES = frozenset({"document-vision", "video"})


@dataclass(frozen=True)
class RuntimePreference:
    preferred: str
    alternates: Tuple[str, ...]
    rationale: str
    host_snapshot: Mapping[str, object]

    def to_dict(self):
        return {
            "preferred": self.preferred,
            "alternates": list(self.alternates),
            "rationale": self.rationale,
            "host_snapshot": {
                "ollama": bool(self.host_snapshot.get("ollama")),
                "lmstudio": bool(self.host_snapshot.get("lmstudio")),
                "chip": self.host_snapshot.get("chip"),
                "ram_gb": self.host_snapshot.get("ram_gb"),
            },
        }


def _needs_vision_path(roles: Sequence[str], modalities: Sequence[str]) -> bool:
    if "vision" in roles:
        return True
    return any(modality in _VISION_MODALITIES for modality in modalities)


def prefer_runtime(
    host: Mapping[str, object],
    roles: Sequence[str] = (),
    modalities: Sequence[str] = (),
) -> RuntimePreference:
    """Return a deterministic preferred runtime with alternates and rationale."""
    ollama = bool(host.get("ollama"))
    lmstudio = bool(host.get("lmstudio"))
    snapshot = {
        "ollama": ollama,
        "lmstudio": lmstudio,
        "chip": host.get("chip"),
        "ram_gb": host.get("ram_gb"),
    }

    if _needs_vision_path(roles, modalities):
        alternates = []
        if lmstudio:
            alternates.append("lmstudio")
        return RuntimePreference(
            preferred="mlx-vlm",
            alternates=tuple(alternates),
            rationale=(
                "Vision / document / video workloads need native mlx-vlm; "
                "Ollama does not run VLMs. LM Studio can help download weights "
                "when its server is up."
            ),
            host_snapshot=snapshot,
        )

    audio = "audio" in modalities
    if lmstudio:
        alternates = ["mlx_lm"]
        if ollama:
            alternates.append("ollama")
        rationale = (
            "LM Studio is the lowest-friction native MLX path for arbitrary Hub "
            "repos when its local server is up."
        )
        if audio:
            rationale += (
                " Audio packs still prefer native MLX weights; Ollama is limited "
                "to curated tags."
            )
        return RuntimePreference(
            preferred="lmstudio",
            alternates=tuple(alternates),
            rationale=rationale,
            host_snapshot=snapshot,
        )

    alternates = []
    if ollama:
        alternates.append("ollama")
    rationale = (
        "Prefer headless mlx_lm for native MLX quants and Hub repos when LM Studio "
        "is not detected."
    )
    if audio:
        rationale += (
            " Audio packs benefit from native MLX serving; Ollama remains a valid "
            "alternate for curated tags only."
        )
    elif ollama:
        rationale += (
            " Ollama remains a valid first choice for curated library tags."
        )
    else:
        rationale += " Start LM Studio or install mlx-lm to serve local models."

    return RuntimePreference(
        preferred="mlx_lm",
        alternates=tuple(alternates),
        rationale=rationale,
        host_snapshot=snapshot,
    )


def wiring_for_preference(repo: str, role: str, preference: RuntimePreference) -> str:
    """Render a suggested wiring string for the preferred runtime."""
    del role
    short = repo.split("/")[-1].lower()
    name = repo.split("/")[-1]
    preferred = preference.preferred
    if preferred == "mlx-vlm":
        return "mlx-vlm server → `mlxvlm/{0}`".format(name)
    if preferred == "lmstudio":
        return "LM Studio → `lmstudio/{0}`".format(short)
    if preferred == "ollama":
        return "Ollama → `ollama/<tag>` (curated / GGUF; not arbitrary mlx-community)"
    return "`mlx_lm.server --model {0}` → custom provider".format(repo)


def prefer_runtime_for_role(role: str, host: Mapping[str, object]) -> RuntimePreference:
    """Scout/discovery helper: preference from a single role + host inventory."""
    modalities = ("document-vision",) if role == "vision" else ()
    return prefer_runtime(host, roles=(role or "general",), modalities=modalities)

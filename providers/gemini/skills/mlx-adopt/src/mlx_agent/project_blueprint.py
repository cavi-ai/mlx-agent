"""MLX project design packs: guidance-only blueprints for downstream agents.

No scaffolding, no downloads, no training. Deterministic templates from a short
project brief. Distinct from dataset blueprints in ``blueprint.py``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from .modality import validate_modalities
from .research import slugify


@dataclass(frozen=True)
class ProjectBrief:
    goal: str
    modalities: Tuple[str, ...] = ()
    notes: str = ""
    memory_gb: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "goal": self.goal,
            "modalities": list(self.modalities),
            "notes": self.notes,
            "memory_gb": self.memory_gb,
        }


@dataclass(frozen=True)
class ProjectDesignPack:
    brief: ProjectBrief
    generated_at: str
    quantization_ideas: Tuple[str, ...]
    training_loop: Tuple[str, ...]
    lora_notes: Tuple[str, ...]
    mtx_notes: Tuple[str, ...]
    study_materials: Tuple[str, ...]
    stack_path: Tuple[str, ...]
    next_steps: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "brief": self.brief.to_dict(),
            "generated_at": self.generated_at,
            "quantization_ideas": list(self.quantization_ideas),
            "training_loop": list(self.training_loop),
            "lora_notes": list(self.lora_notes),
            "mtx_notes": list(self.mtx_notes),
            "study_materials": list(self.study_materials),
            "stack_path": list(self.stack_path),
            "next_steps": list(self.next_steps),
        }


def _resolve_memory(raw: object) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError("memory_gb must be a number") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("memory_gb must be a positive number")
    return value


def build_brief(answers: Mapping[str, object]) -> ProjectBrief:
    goal = str(answers.get("goal") or answers.get("domain") or "").strip()
    if not goal:
        raise ValueError("goal is required")
    modalities_raw = answers.get("modalities") or ()
    if isinstance(modalities_raw, str):
        modalities_raw = [part.strip() for part in modalities_raw.split(",") if part.strip()]
    modalities = validate_modalities(list(modalities_raw)) if modalities_raw else ()
    return ProjectBrief(
        goal=goal,
        modalities=modalities,
        notes=str(answers.get("notes", "")).strip(),
        memory_gb=_resolve_memory(answers.get("memory_gb")),
    )


def generate_design_pack(
    brief: ProjectBrief,
    now: Optional[datetime] = None,
) -> ProjectDesignPack:
    """Build a deterministic project design pack. Guidance only."""
    moment = now or datetime.now(timezone.utc)
    visionish = any(item in brief.modalities for item in ("document-vision", "video"))
    audio = "audio" in brief.modalities
    memory = brief.memory_gb

    quant: List[str] = [
        "Start from published MLX community 4-bit / 8-bit weights when available.",
        "Prefer 4-bit for chat/tooling under tight unified memory; keep 8-bit for quality-sensitive OCR or reasoning.",
        "Record the exact Hub repo + quant tag in the research pack before fine-tuning.",
    ]
    if memory is not None:
        quant.append(
            "Respect the {0:g} GB budget with ~20% runtime headroom beyond weight size.".format(memory)
        )
    if visionish:
        quant.append(
            "Vision stacks need mlx-vlm-compatible VLM quants; do not plan Ollama-only vision paths."
        )
    if audio:
        quant.append(
            "Audio ASR/TTS often ships as specialty checkpoints — verify native MLX serve path before training plans."
        )

    training = (
        "Define a tiny offline JSONL/Parquet dataset before any train loop.",
        "Dry-run one batch on Apple Silicon with mlx-lm / project tooling; abort on OOM.",
        "Use LoRA/QLoRA-style adapters before full fine-tunes; keep base weights frozen.",
        "Hold out 10–20% validation with no near-duplicate leakage.",
        "This blueprint does not train or download — execution stays outside mlx-agent.",
    )

    lora = (
        "Search adapters in `mlx-agent research` (PEFT/LoRA section) before training new ones.",
        "Serve adapters with mlx_lm `--adapter-path` after verify; keep base + adapter versions pinned.",
        "If no Hub adapter fits, use the research dataset blueprint to draft labels locally.",
    )

    mtx = (
        "Treat experimental MTX / mixed-precision experiments as optional research notes, not production defaults.",
        "Document any experimental kernel or quant recipe separately from the shipping wire config.",
        "Prefer proven mlx-community quants for the first vertical slice; revisit MTX only after a working baseline.",
    )

    study = (
        "Apple MLX documentation: model convert / quantize / generate flows.",
        "mlx-lm and mlx-vlm server READMEs for OpenAI-compatible local serving.",
        "mlx-scout runtimes reference: LM Studio vs mlx_lm vs Ollama trade-offs.",
        "PEFT/LoRA primers focused on low-rank adapters for domain dial-in.",
    )

    stack = (
        "1. Run `mlx-agent research` with the same goal/modalities to rank models, adapters, and datasets.",
        "2. Follow the pack's Runtime preference (mlx-vlm / LM Studio / mlx_lm; Ollama only as curated alternate).",
        "3. `mlx-agent adopt start` to verify a candidate on a local runtime.",
        "4. `mlx-agent wire` to apply a confirmation-gated config — never skip preview hashes.",
    )

    next_steps = (
        "Freeze the goal and modalities above; do not expand scope mid-slice.",
        "Generate a research pack and pick one base model + optional adapter.",
        "Create or obtain a small labeled dataset offline.",
        "Implement training outside mlx-agent; keep this design pack as the agent-ingestible plan.",
    )
    if brief.notes:
        next_steps = next_steps + ("Honor project notes: {0}".format(brief.notes),)

    return ProjectDesignPack(
        brief=brief,
        generated_at=moment.isoformat(),
        quantization_ideas=tuple(quant),
        training_loop=training,
        lora_notes=lora,
        mtx_notes=mtx,
        study_materials=study,
        stack_path=stack,
        next_steps=next_steps,
    )


def render_design_pack(pack: ProjectDesignPack) -> str:
    brief = pack.brief
    lines = [
        "# MLX Project Design Pack: {0}".format(brief.goal),
        "",
        "- Generated: {0}".format(pack.generated_at),
        "- Modalities: {0}".format(", ".join(brief.modalities) or "none"),
        "- Memory budget (GB): {0}".format(
            brief.memory_gb if brief.memory_gb is not None else "none"
        ),
    ]
    if brief.notes:
        lines.append("- Notes: {0}".format(brief.notes))
    lines.extend([
        "",
        "## Goal",
        "",
        brief.goal,
        "",
        "## Recommended stack path",
        "",
    ])
    for item in pack.stack_path:
        lines.append("- {0}".format(item))
    lines.extend(["", "## Quantization ideas", ""])
    for item in pack.quantization_ideas:
        lines.append("- {0}".format(item))
    lines.extend(["", "## Training loop sketch", ""])
    for item in pack.training_loop:
        lines.append("- {0}".format(item))
    lines.extend(["", "## LoRA / adapter notes", ""])
    for item in pack.lora_notes:
        lines.append("- {0}".format(item))
    lines.extend(["", "## Experimental MTX notes", ""])
    for item in pack.mtx_notes:
        lines.append("- {0}".format(item))
    lines.extend(["", "## Study materials", ""])
    for item in pack.study_materials:
        lines.append("- {0}".format(item))
    lines.extend(["", "## Next steps", ""])
    for index, item in enumerate(pack.next_steps, start=1):
        lines.append("{0}. {1}".format(index, item))
    lines.extend([
        "",
        "## Notes",
        "",
        "This pack is guidance only for downstream agents. "
        "mlx-agent blueprint does not scaffold repositories, download weights, or run training.",
        "",
    ])
    return "\n".join(lines)


def write_design_pack(
    markdown: str,
    brief: ProjectBrief,
    root: Optional[object] = None,
    now: Optional[datetime] = None,
    pack: Optional[ProjectDesignPack] = None,
) -> Path:
    """Write under ``<root>/mlx-blueprints`` and never outside it."""
    moment = now or datetime.now(timezone.utc)
    base = Path(root) if root is not None else Path.cwd()
    out_dir = (base / "mlx-blueprints").resolve()
    unresolved_dir = base / "mlx-blueprints"
    if unresolved_dir.is_symlink():
        raise ValueError("refusing to write design pack through a symlinked folder")
    timestamp = moment.strftime("%Y%m%dT%H%M%SZ")
    filename = "{0}-{1}.md".format(slugify(brief.goal), timestamp)
    path = (out_dir / filename).resolve()
    if os.path.commonpath([str(out_dir), str(path)]) != str(out_dir):
        raise ValueError("refusing to write design pack outside the project folder")
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    if pack is not None:
        sidecar = path.with_suffix(".json")
        if os.path.commonpath([str(out_dir), str(sidecar.resolve())]) != str(out_dir):
            raise ValueError("refusing to write design pack outside the project folder")
        sidecar.write_text(json.dumps(pack.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path

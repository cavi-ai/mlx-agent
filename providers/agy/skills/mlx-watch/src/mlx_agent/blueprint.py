"""Deterministic dataset-creation blueprint from a DomainIntent.

Guidance only: no download, no training, no execution. Templates are fixed so
the same intent always yields the same blueprint text and structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .interview import DomainIntent


@dataclass(frozen=True)
class DatasetBlueprint:
    goal: str
    schema_fields: Tuple[str, ...]
    labeling_notes: str
    split_guidance: str
    license_privacy: str
    mlx_next_steps: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "goal": self.goal,
            "schema_fields": list(self.schema_fields),
            "labeling_notes": self.labeling_notes,
            "split_guidance": self.split_guidance,
            "license_privacy": self.license_privacy,
            "mlx_next_steps": self.mlx_next_steps,
        }

    def to_markdown(self) -> str:
        lines = [
            "### Goal",
            "",
            self.goal,
            "",
            "### Suggested schema",
            "",
        ]
        for field_name in self.schema_fields:
            lines.append("- `{0}`".format(field_name))
        lines.extend([
            "",
            "### Labeling notes",
            "",
            self.labeling_notes,
            "",
            "### Train / val split",
            "",
            self.split_guidance,
            "",
            "### License and privacy",
            "",
            self.license_privacy,
            "",
            "### MLX fine-tune next steps",
            "",
            self.mlx_next_steps,
            "",
        ])
        return "\n".join(lines)


_DEFAULT_FIELDS = (
    "id",
    "input_text",
    "label",
    "source",
    "license",
    "notes",
)


def build_dataset_blueprint(intent: DomainIntent) -> DatasetBlueprint:
    """Build a deterministic dataset blueprint for an empty catalog result."""
    roles = ", ".join(intent.roles) or "general"
    keywords = ", ".join(intent.keywords) or "none"
    licenses = ", ".join(intent.license_allow) or "document and respect upstream licenses"
    notes = intent.notes.strip() or "none recorded"

    schema = list(_DEFAULT_FIELDS)
    if "vision" in intent.roles:
        schema.insert(1, "image_path")
    if "embedding" in intent.roles:
        schema.insert(-2, "embedding_text")

    goal = (
        "Create a small, domain-specific dataset for **{0}** covering roles "
        "({1}) and keywords ({2}). Prefer quality over volume; keep examples "
        "representative of real production inputs."
    ).format(intent.domain, roles, keywords)

    labeling = (
        "Define clear positive/negative or multi-class labels tied to the domain. "
        "Double-check borderline cases. Keywords to cover: {0}. Extra notes: {1}."
    ).format(keywords, notes)

    split = (
        "Hold out 10–20% for validation with no leakage from near-duplicates. "
        "Stratify by label when classes are imbalanced. Keep a tiny smoke-test "
        "slice for local MLX dry-runs."
    )

    license_privacy = (
        "Allowed licenses for this intent: {0}. Do not include secrets, credentials, "
        "or personal data without consent and redaction. Record provenance per row."
    ).format(licenses)

    next_steps = (
        "1. Draft the schema above as JSONL or Parquet locally (no Hub upload required).\n"
        "2. Re-run `mlx-agent research` after datasets appear on the Hub, or keep this "
        "blueprint as the creation guide.\n"
        "3. When a base MLX model is chosen, use `mlx-agent adopt start` to verify and "
        "wire it — research/blueprint never downloads or trains.\n"
        "4. Fine-tune offline with your preferred MLX tooling against the local dataset."
    )

    return DatasetBlueprint(
        goal=goal,
        schema_fields=tuple(schema),
        labeling_notes=labeling,
        split_guidance=split,
        license_privacy=license_privacy,
        mlx_next_steps=next_steps,
    )

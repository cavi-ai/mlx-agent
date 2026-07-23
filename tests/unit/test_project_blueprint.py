"""Unit tests for MLX project design blueprints (guidance only)."""

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.project_blueprint import (
    ProjectBrief,
    ProjectDesignPack,
    build_brief,
    generate_design_pack,
    render_design_pack,
    write_design_pack,
)


class BuildBriefTests(unittest.TestCase):
    def test_requires_goal(self):
        with self.assertRaises(ValueError):
            build_brief({"goal": "  "})

    def test_accepts_modalities_and_notes(self):
        brief = build_brief({
            "goal": "On-device legal OCR assistant",
            "modalities": ["document-vision", "audio"],
            "notes": "offline only",
            "memory_gb": "32",
        })
        self.assertEqual(brief.goal, "On-device legal OCR assistant")
        self.assertEqual(brief.modalities, ("document-vision", "audio"))
        self.assertEqual(brief.notes, "offline only")
        self.assertEqual(brief.memory_gb, 32.0)

    def test_rejects_unknown_modality(self):
        with self.assertRaises(ValueError):
            build_brief({"goal": "x", "modalities": ["smell"]})


class GenerateDesignPackTests(unittest.TestCase):
    def test_deterministic_sections(self):
        brief = ProjectBrief(goal="Local ASR notes app", modalities=("audio",))
        first = generate_design_pack(brief, now=datetime(2026, 7, 23, tzinfo=timezone.utc))
        second = generate_design_pack(brief, now=datetime(2026, 7, 23, tzinfo=timezone.utc))
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertIsInstance(first, ProjectDesignPack)
        md = render_design_pack(first)
        for heading in (
            "# MLX Project Design Pack:",
            "## Goal",
            "## Recommended stack path",
            "## Quantization ideas",
            "## Training loop sketch",
            "## LoRA / adapter notes",
            "## Experimental MTX notes",
            "## Study materials",
            "## Next steps",
        ):
            self.assertIn(heading, md)
        self.assertIn("audio", md)
        self.assertIn("mlx-agent research", md)
        self.assertIn("does not train", md.lower())

    def test_vision_modalities_mention_mlx_vlm(self):
        pack = generate_design_pack(
            ProjectBrief(goal="PDF OCR", modalities=("document-vision",)),
            now=datetime(2026, 7, 23, tzinfo=timezone.utc),
        )
        md = render_design_pack(pack)
        self.assertIn("mlx-vlm", md)


class WriteDesignPackTests(unittest.TestCase):
    def test_writes_md_and_json_inside_project(self):
        brief = ProjectBrief(goal="Chat helper", modalities=())
        pack = generate_design_pack(brief, now=datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc))
        with TemporaryDirectory() as root:
            path = write_design_pack(
                render_design_pack(pack),
                brief,
                root=root,
                now=datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc),
                pack=pack,
            )
            self.assertEqual(path.parent.name, "mlx-blueprints")
            self.assertTrue(path.name.startswith("chat-helper-"))
            self.assertTrue(path.with_suffix(".json").is_file())
            payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(payload["brief"]["goal"], "Chat helper")
            self.assertIn("quantization_ideas", payload)


if __name__ == "__main__":
    unittest.main()

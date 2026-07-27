import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.fleet import (
    FleetConfigAdapter,
    FleetError,
    assignments_from_adoption,
    parse_assignments,
    parse_runtime_map,
)


class ParseAssignmentsTests(unittest.TestCase):
    def test_valid_assignments(self):
        result = parse_assignments(["coding=pub/coder", "vision=pub/see"])
        self.assertEqual(result, {"coding": "pub/coder", "vision": "pub/see"})

    def test_missing_assignments(self):
        with self.assertRaises(FleetError) as caught:
            parse_assignments(None)
        self.assertEqual(caught.exception.code, "missing_assignments")

    def test_unknown_role(self):
        with self.assertRaises(FleetError) as caught:
            parse_assignments(["wizard=pub/model"])
        self.assertEqual(caught.exception.code, "invalid_role")

    def test_duplicate_role(self):
        with self.assertRaises(FleetError) as caught:
            parse_assignments(["coding=pub/a", "coding=pub/b"])
        self.assertEqual(caught.exception.code, "duplicate_role")

    def test_unsafe_repo(self):
        with self.assertRaises(FleetError) as caught:
            parse_assignments(["coding=../evil"])
        self.assertEqual(caught.exception.code, "invalid_repo")


class RuntimeMapTests(unittest.TestCase):
    def test_valid_override(self):
        self.assertEqual(parse_runtime_map(["coding=mlx-vlm"]), {"coding": "mlx-vlm"})

    def test_unknown_runtime(self):
        with self.assertRaises(FleetError) as caught:
            parse_runtime_map(["coding=ollama"])
        self.assertEqual(caught.exception.code, "unsupported_runtime")


class AdoptionAssignmentsTests(unittest.TestCase):
    def test_reads_recommendations(self):
        with TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            state.write_text(json.dumps({
                "recommendations": [
                    {"role": "coding", "repo": "pub/coder"},
                    {"role": "vision", "repo": "pub/see"},
                ]
            }))
            result = assignments_from_adoption(state)
            self.assertEqual(result, {"coding": "pub/coder", "vision": "pub/see"})

    def test_empty_recommendations(self):
        with TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            state.write_text(json.dumps({"recommendations": []}))
            with self.assertRaises(FleetError) as caught:
                assignments_from_adoption(state)
            self.assertEqual(caught.exception.code, "empty_recommendations")


class FleetAdapterTests(unittest.TestCase):
    def test_render_assigns_default_ports(self):
        adapter = FleetConfigAdapter()
        content = adapter.render({"coding": "pub/coder", "vision": "pub/see"})
        self.assertIn("model_name: coding", content)
        self.assertIn("api_base: http://127.0.0.1:8080/v1", content)
        self.assertIn("model_name: vision", content)
        self.assertIn("api_base: http://127.0.0.1:8083/v1", content)
        self.assertTrue(content.startswith("# MLX_AGENT_WIRE v1\n"))

    def test_render_honors_runtime_override(self):
        adapter = FleetConfigAdapter()
        content = adapter.render({"coding": "pub/coder"}, {"coding": "mlx-vlm"})
        self.assertIn("api_base: http://127.0.0.1:8083/v1", content)

    def test_roundtrip_validate(self):
        adapter = FleetConfigAdapter()
        content = adapter.render({
            "general": "pub/a", "coding": "pub/b", "embedding": "pub/c",
        })
        self.assertTrue(adapter.validate(content))

    def test_validate_rejects_tampered_entry(self):
        adapter = FleetConfigAdapter()
        content = adapter.render({"coding": "pub/coder"})
        tampered = content.replace("8080", "9999")
        with self.assertRaises(ValueError):
            adapter.validate(tampered)

    def test_validate_rejects_duplicate_roles(self):
        adapter = FleetConfigAdapter()
        content = adapter.render({"coding": "pub/coder"})
        doubled = content + content.split("\n", 2)[2]
        with self.assertRaises(ValueError):
            adapter.validate(doubled)

    def test_render_refuses_unmanaged_existing_config(self):
        adapter = FleetConfigAdapter()
        with self.assertRaises(FleetError) as caught:
            adapter.render({"coding": "pub/coder"}, existing="model_list: []\n")
        self.assertEqual(caught.exception.code, "existing_config_unmanaged")

    def test_render_refuses_secrets_in_existing(self):
        adapter = FleetConfigAdapter()
        with self.assertRaises(ValueError):
            adapter.render({"coding": "pub/coder"}, existing='{"api_key": "sk-live"}')


if __name__ == "__main__":
    unittest.main()

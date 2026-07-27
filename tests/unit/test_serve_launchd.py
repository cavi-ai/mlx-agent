import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.serve import (
    LaunchdPlistAdapter,
    launchd_label,
    load_recipes,
    plan_start,
    render_launchd_plist,
)


RECIPES = load_recipes()


def _plan(**overrides):
    values = {"repo": "pub/model", "runtime": "mlx_lm"}
    values.update(overrides)
    return plan_start(values.pop("repo"), values.pop("runtime"), RECIPES, **values)


class LaunchdRenderTests(unittest.TestCase):
    def test_label_uses_port(self):
        self.assertEqual(launchd_label(8080), "com.mlx-agent.serve.8080")

    def test_plist_contains_plan_argv(self):
        plist = render_launchd_plist(_plan(), log_path="/tmp/serve.log")
        self.assertIn("<string>com.mlx-agent.serve.8080</string>", plist)
        self.assertIn("<string>mlx_lm.server</string>", plist)
        self.assertIn("<string>pub/model</string>", plist)
        self.assertIn("<key>RunAtLoad</key>", plist)
        self.assertIn("<string>/tmp/serve.log</string>", plist)
        self.assertTrue(plist.endswith("</plist>\n"))

    def test_special_characters_are_escaped(self):
        plan = _plan(repo="pub/model", adapter_path="/tmp/a & b")
        plist = render_launchd_plist(plan, log_path="/tmp/x.log")
        self.assertIn("/tmp/a &amp; b", plist)

    def test_adapter_validates_rendered_plist(self):
        adapter = LaunchdPlistAdapter()
        plist = render_launchd_plist(_plan(), log_path="/tmp/serve.log")
        self.assertTrue(adapter.validate(plist))

    def test_adapter_rejects_foreign_label(self):
        adapter = LaunchdPlistAdapter()
        plist = render_launchd_plist(_plan(), log_path="/tmp/x.log")
        tampered = plist.replace("com.mlx-agent.serve.8080", "com.evil.daemon")
        with self.assertRaises(ValueError):
            adapter.validate(tampered)

    def test_adapter_rejects_non_plist(self):
        adapter = LaunchdPlistAdapter()
        with self.assertRaises(ValueError):
            adapter.validate("not a plist")


if __name__ == "__main__":
    unittest.main()

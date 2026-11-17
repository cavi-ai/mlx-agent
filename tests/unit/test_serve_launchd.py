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
    def setUp(self):
        from unittest.mock import patch
        resolver = patch('mlx_agent.model_serving.resolve_model', side_effect=lambda value, *_: value if value.startswith('/') else '/models/example')
        resolver.start()
        self.addCleanup(resolver.stop)

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
        plan = _plan(repo=None, path="/tmp/a & b")
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

class LaunchdOwnershipTests(unittest.TestCase):
    def test_plist_routes_runtime_through_model_claim_entrypoint(self):
        import plistlib
        from unittest.mock import patch
        plan = _plan()
        with patch('mlx_agent.model_serving.resolve_model', return_value='/models/example'):
            arguments = plistlib.loads(render_launchd_plist(plan).encode())['ProgramArguments']
        self.assertTrue(arguments[1].endswith('/model_serving_entry.py'))
        self.assertIn('/models/example', arguments)
        self.assertEqual(arguments[arguments.index('--') + 1:], plan['argv'])

    def test_entrypoint_claims_before_exec_and_cleans_exec_failure(self):
        from unittest.mock import Mock, patch
        from mlx_agent import model_serving_entry
        claim = Mock()
        command = ['--artifact', '/models/example', '--runtime', 'mlx_lm', '--port', '8080',
                   '--auth-required', 'false', '--', 'mlx_lm.server', '--model', '/models/example', '--port', '8080']
        with patch.object(model_serving_entry, 'resolve_model', side_effect=lambda value, *_: value), \
             patch.object(model_serving_entry, 'acquire', return_value=claim), \
             patch.object(model_serving_entry, 'process_birth_identity', return_value='birth'), \
             patch.object(model_serving_entry.os, 'getpgrp', return_value=77), \
             patch.object(model_serving_entry.os, 'getpid', return_value=77), \
             patch.object(model_serving_entry.os, 'execvpe', side_effect=OSError('exec failed')):
            with self.assertRaisesRegex(OSError, 'exec failed'):
                model_serving_entry.main(command)
        claim.update.assert_called_once_with(state='active', pid=77, birth='birth', pgid=77, jit=False, ready=False)
        claim.cancel_failed_exec.assert_called_once()

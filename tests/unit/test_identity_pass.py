"""Both start checks use one identity pass."""
import inspect
import unittest
from unittest.mock import patch

from mlx_agent.serve import recheck_bound_identity, start_serve

_CHECKS = (
    "_require_auth_identity",
    "_require_task_identity",
    "_require_training_binding",
    "_require_adapter_identity",
    "_require_prefill_identity",
    "_require_decode_identity",
    "_require_vision_identity",
    "_require_splash_identity",
    "_require_access_identity",
)


class IdentityPassTests(unittest.TestCase):
    def test_start_calls_one_pass_before_spawn_and_after_readiness(self):
        source = inspect.getsource(start_serve)
        self.assertEqual(source.count("recheck_bound_identity(plan)"), 2)
        for name in _CHECKS:
            self.assertNotIn(name + "(plan)", source)

    def test_the_pass_runs_every_bound_check_in_order(self):
        seen = []

        def record(name):
            def check(plan):
                del plan
                seen.append(name)
            return check

        with patch.multiple("mlx_agent.serve", **{name: record(name) for name in _CHECKS}):
            recheck_bound_identity({"runtime": "mlx_lm"})
        self.assertEqual(seen, list(_CHECKS))

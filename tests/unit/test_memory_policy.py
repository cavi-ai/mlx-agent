"""Memory policy has one translation and one set of runtime gates."""
import unittest
from types import SimpleNamespace

from mlx_agent.jit_serve import validate_memory_policy
from mlx_agent.memory_policy import (PolicyRejected, cli_supplied, enforce_configure, enforce_launch, for_start,
                                     from_cli, policy_from_profile, reservation, validate)


def _args(**overrides):
    values = dict(idle_timeout=None, keep_loaded=None, min_headroom_gb=None, context_tokens=None,
                  unload_under_pressure=None, memory_budget_gb=None, execution_wait_seconds=None,
                  release_after_request=None, allocator_cache_limit_mib=None, cache_reclaim=None,
                  clear_allocator_cache_limit=False, clear_memory_budget=False)
    values.update(overrides)
    return SimpleNamespace(**values)


class MemoryPolicyTests(unittest.TestCase):
    def test_validate_matches_the_gateway_name(self):
        policy = {"execution_wait_seconds": 30, "minimum_headroom_gb": 2}
        self.assertEqual(validate(policy), validate_memory_policy(policy))
        self.assertEqual(validate(None), validate_memory_policy(None))

    def test_lone_execution_wait_is_a_start_policy(self):
        arguments = _args(execution_wait_seconds=30)
        self.assertTrue(cli_supplied(arguments))
        self.assertEqual(for_start(arguments, "mlx_lm")["execution_wait_seconds"], 30)
        self.assertIsNone(for_start(_args(), "mlx_lm"))

    def test_task_execution_wait_keeps_release_after_request(self):
        policy = for_start(_args(execution_wait_seconds=30), "tasks")
        self.assertEqual(policy["execution_wait_seconds"], 30)
        self.assertTrue(policy["release_after_request"])

    def test_task_idle_timeout_does_not_invent_release(self):
        policy = for_start(_args(idle_timeout=60), "tasks")
        self.assertEqual(policy["idle_timeout_seconds"], 60)
        self.assertNotIn("release_after_request", policy)

    def test_profile_names_validate_through_the_same_function(self):
        settings = {"min_headroom_gb": 4, "idle_timeout": 10, "keep_loaded": False, "unload_under_pressure": False,
                    "execution_wait_seconds": 5}
        validated = validate(policy_from_profile(settings))
        self.assertEqual(validated["minimum_headroom_gb"], 4)
        self.assertEqual(validated["execution_wait_seconds"], 5)
        self.assertEqual(from_cli(_args(min_headroom_gb=4, idle_timeout=10, execution_wait_seconds=5))["minimum_headroom_gb"], 4)

    def test_launch_and_configure_share_the_cache_and_budget_rules(self):
        cache = validate({"allocator_cache_limit_mib": 32, "minimum_headroom_gb": 1})
        with self.assertRaises(PolicyRejected) as launch:
            enforce_launch(cache, "mlx_lm", True)
        with self.assertRaises(PolicyRejected) as configure:
            enforce_configure(cache, "mlx_lm", {})
        self.assertEqual(launch.exception.code, configure.exception.code)
        budget = validate({"model_memory_budget_gb": 8})
        with self.assertRaises(PolicyRejected):
            enforce_launch(budget, "mlx_lm", True)
        enforce_launch(budget, "strata", True)
        enforce_configure(budget, "tasks", {})
        with self.assertRaises(PolicyRejected) as task_context:
            enforce_configure(validate({"minimum_headroom_gb": 1}), "tasks", {"context_tokens": 2048})
        self.assertEqual(str(task_context.exception), "Task endpoints do not use text KV context settings.")

    def test_splash_reservation_uses_the_reviewed_cap(self):
        outcome = reservation(validate(None), {"runtime": "splash", "splash_weight_bytes": 10,
                                                "splash_max_memory_gb": 4, "local_path": "/missing"})
        self.assertEqual(outcome["size"], 4_000_000_000)

    def test_strata_reservation_uses_the_stored_sizes(self):
        policy = validate({"minimum_headroom_gb": 2, "model_memory_budget_gb": 8})
        outcome = reservation(policy, {"runtime": "strata", "local_path": "/missing",
                                        "strata_memory_allocation": {"estimated_resident_bytes": 1234},
                                        "strata_memory_growth_bytes": 50})
        self.assertEqual(outcome["size"], 1234)
        self.assertEqual(outcome["reserve_bytes"] + outcome["workspace_bytes"] + outcome["growth_bytes"], 1234)

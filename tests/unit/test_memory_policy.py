"""Memory policy has one translation and one set of runtime gates."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mlx_agent.jit_serve import validate_memory_policy
from mlx_agent.memory_policy import (PolicyRejected, cli_supplied, enforce_configure, enforce_launch, for_start,
                                     from_cli, policy_from_profile, reservation, validate)
from mlx_agent.serve import configure_serve_memory


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

    def test_validate_omits_default_execution_wait_and_release_after_request(self):
        base = validate(None)
        self.assertEqual(validate({"execution_wait_seconds": 0}), base)
        self.assertEqual(validate({"release_after_request": False}), base)
        self.assertEqual(
            validate({"execution_wait_seconds": 0, "release_after_request": False, "keep_loaded": True}),
            {"idle_timeout_seconds": 0, "keep_loaded": True, "minimum_headroom_gb": None},
        )
        for seconds in (1, 30, 3600):
            with self.subTest(seconds=seconds):
                self.assertEqual(validate({"execution_wait_seconds": seconds})["execution_wait_seconds"], seconds)
        enabled = validate({"execution_wait_seconds": 45, "release_after_request": True})
        self.assertEqual(enabled["execution_wait_seconds"], 45)
        self.assertIs(enabled["release_after_request"], True)

    def test_validate_rejects_invalid_execution_wait_and_release_after_request(self):
        for bad_wait in (None, True, False, 1.5, -1, 3601, "30"):
            with self.subTest(bad_wait=bad_wait), self.assertRaises(ValueError):
                validate({"execution_wait_seconds": bad_wait})
        for bad_release in (None, 0, 1, "true", "false"):
            with self.subTest(bad_release=bad_release), self.assertRaises(ValueError):
                validate({"release_after_request": bad_release})
        with self.assertRaises(ValueError):
            validate({"keep_loaded": True, "release_after_request": True})
        # for_start keeps raw from_cli values and task default residency.
        self.assertEqual(for_start(_args(execution_wait_seconds=0), "mlx_lm")["execution_wait_seconds"], 0)
        self.assertIs(for_start(_args(release_after_request=False), "tasks")["release_after_request"], False)
        self.assertIs(for_start(_args(execution_wait_seconds=10), "tasks")["release_after_request"], True)

    def test_configure_serve_memory_resets_omitted_wait_and_release_without_clearing_budget(self):
        existing = validate({
            "minimum_headroom_gb": 2,
            "model_memory_budget_gb": 8,
            "execution_wait_seconds": 30,
            "release_after_request": True,
        })
        with patch("mlx_agent.serve._configure_owned_jit", return_value={}) as send:
            configure_serve_memory(19080, {"keep_loaded": True, "minimum_headroom_gb": 2})
            payload = send.call_args.args[4]
        self.assertEqual(payload["execution_wait_seconds"], 0)
        self.assertIs(payload["release_after_request"], False)
        self.assertNotIn("model_memory_budget_gb", payload)
        merged = validate(dict(existing, **payload))
        self.assertEqual(
            merged,
            {"idle_timeout_seconds": 0, "keep_loaded": True, "minimum_headroom_gb": 2, "model_memory_budget_gb": 8},
        )

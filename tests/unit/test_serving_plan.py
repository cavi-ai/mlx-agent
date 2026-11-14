"""The serving plan's projections share one field set."""
import unittest

from mlx_agent.serving_plan import Contribution, memory_basis, preview_hash, receipt_fields, worker_config


def _plan(**extra):
    plan = {
        "runtime": "mlx_lm", "worker_argv": ["mlx_lm.server", "--model", "local"],
        "local_path": "/model", "fingerprint": "abc", "max_tokens": 8, "repo": None, "path": "/model",
    }
    plan.update(extra)
    return plan


class ServingPlanTests(unittest.TestCase):
    def test_preview_hash_ignores_its_own_key_and_dict_order(self):
        plan = _plan(preview_hash="stale")
        again = {"preview_hash": "other", "path": "/model", "repo": None, "max_tokens": 8,
                 "fingerprint": "abc", "local_path": "/model", "worker_argv": plan["worker_argv"], "runtime": "mlx_lm"}
        self.assertEqual(preview_hash(plan), preview_hash(again))
        self.assertNotEqual(preview_hash(plan), "stale")

    def test_contribution_retargets_the_worker_program_only(self):
        contribution = Contribution(["--draft", "x"], {"executable": "splash"}, {"splash_served_model": "local/model"},
                                    worker_entry=["/python", "/launcher"])
        values, argv, plan = {}, ["recipe"], {}
        contribution.apply_values(values)
        contribution.extend_argv(argv)
        contribution.apply_fields(plan)
        worker = ["splash", "--model", "local/model"]
        self.assertTrue(contribution.retarget_worker(worker))
        self.assertEqual(values["executable"], "splash")
        self.assertEqual(argv, ["recipe", "--draft", "x"])
        self.assertEqual(plan["splash_served_model"], "local/model")
        self.assertEqual(worker, ["/python", "/launcher", "--model", "local/model"])
        self.assertFalse(Contribution([], {}, {}).retarget_worker(["mlx_lm.server"]))

    def test_splash_projection_is_one_field_set(self):
        plan = _plan(runtime="splash", splash_served_model="local/model", splash_weight_bytes=10,
                     splash_max_memory_gb=3, splash_version="1.6.0", jit=True)
        config = worker_config(plan, control_token="token", state_path="/state")
        receipt = receipt_fields(plan)
        basis = memory_basis(plan)
        self.assertEqual(config["splash_served_model"], "local/model")
        self.assertEqual(config["splash_weight_bytes"], 10)
        self.assertEqual(config["splash_max_memory_gb"], 3)
        self.assertEqual(receipt["splash_served_model"], "local/model")
        self.assertEqual(receipt["splash_version"], "1.6.0")
        self.assertEqual(receipt["jit"], True)
        self.assertEqual((basis.kind, basis.weights, basis.cap_gb), ("splash", 10, 3))

    def test_strata_projection_keeps_allocation_on_the_worker_and_the_receipt(self):
        allocation = {"estimated_resident_bytes": 40, "mode": "resident"}
        plan = _plan(runtime="strata", strata_decoding="mtp", strata_memory_allocation=allocation,
                     strata_memory_estimate_bytes=40, strata_memory_growth_bytes=5, strata_context_tokens=1024)
        config = worker_config(plan, control_token="token", state_path="/state")
        self.assertEqual(config["strata_decoding"], "mtp")
        self.assertEqual(config["strata_memory_allocation"], allocation)
        self.assertEqual(config["strata_context_tokens"], 1024)
        self.assertEqual(receipt_fields(plan)["strata_memory_allocation"], allocation)
        basis = memory_basis(plan)
        self.assertEqual((basis.kind, basis.allocation_bytes, basis.growth_bytes), ("strata", 40, 5))

    def test_chat_plan_does_not_require_runtime_sizes(self):
        self.assertEqual(memory_basis(_plan()).kind, "local")
        self.assertNotIn("splash_served_model", receipt_fields(_plan()))
        config = worker_config(_plan(memory_policy={"keep_loaded": False}), control_token="token", state_path="/state")
        self.assertEqual(config["memory_policy"]["keep_loaded"], False)
        self.assertNotIn("log_path", config)

    def test_task_worker_config_takes_its_state_directory_from_the_caller(self):
        plan = _plan(runtime="tasks", task_identity={"backend": "embed"}, task_kinds=["embedding"], port=9,
                     task_runtime_fingerprint="abc")
        config = worker_config(plan, control_token="token", state_path="/state", task_state_dir="/tasks")
        self.assertEqual(config["task_state_dir"], "/tasks")
        self.assertEqual(receipt_fields(plan)["task_kinds"], ["embedding"])

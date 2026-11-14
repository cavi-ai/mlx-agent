"""One transport choice, and one gateway config."""
import unittest

from mlx_agent.worker_transport import choose, gateway_config


def _plan(**extra):
    plan = {"runtime": "mlx_lm", "worker_argv": ["mlx_lm.server", "--model", "local"],
            "local_path": "/model", "fingerprint": "abc", "max_tokens": 8, "repo": None, "path": "/model"}
    plan.update(extra)
    return plan


class WorkerTransportTests(unittest.TestCase):
    def test_choose_names_the_four_transports(self):
        self.assertEqual(choose(_plan()).name, "open")
        self.assertEqual(choose(_plan(api_key_sha256="a" * 64)).name, "inherited-unix")
        marked = _plan(api_key_sha256="a" * 64)
        choose(marked).mark(marked)
        self.assertEqual(marked["private_worker"]["protocol"], "inherited-unix-v1")
        accelerated = _plan(api_key_sha256="a" * 64, prefill_kernel={"adapter_enabled": True})
        self.assertEqual(choose(accelerated).name, "loopback-key")
        choose(accelerated).mark(accelerated)
        self.assertNotIn("private_worker", accelerated)
        self.assertEqual(choose(_plan(runtime="tasks", api_key_sha256="a" * 64)).name, "resident")

    def test_gateway_config_keeps_memory_policy_for_every_caller(self):
        plan = _plan(memory_policy={"idle_timeout_seconds": 0, "keep_loaded": False, "minimum_headroom_gb": None},
                     private_worker={"protocol": "inherited-unix-v1"}, api_key_sha256="a" * 64)
        config = gateway_config(plan, control_token="token", state_path="/tmp/state.json")
        self.assertEqual(config["memory_policy"]["keep_loaded"], False)
        self.assertEqual(config["private_worker"]["protocol"], "inherited-unix-v1")
        self.assertNotIn("log_path", config)
        supervised = gateway_config(plan, control_token="token", state_path="/tmp/state.json",
                                    log_path="/tmp/worker.log", config_path="/tmp/port.jit-config.json")
        self.assertEqual(supervised["log_path"], "/tmp/worker.log")
        self.assertEqual(supervised["config_path"], "/tmp/port.jit-config.json")

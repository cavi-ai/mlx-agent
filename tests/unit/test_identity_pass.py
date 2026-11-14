"""The identity pass refuses a serving plan whose bound files changed."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.identity_pass import recheck_bound_identity
from mlx_agent.jit_serve import model_fingerprint
from mlx_agent.lora import adapter_identity
from mlx_agent.serve import ServeError


class IdentityPassTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}\n")
        (self.model / "model.safetensors").write_bytes(b"weights")

    def plan(self, **extra):
        plan = {"runtime": "mlx_lm", "jit": True, "local_path": str(self.model),
                "fingerprint": model_fingerprint(self.model)}
        plan.update(extra)
        return plan

    def test_a_matching_model_directory_passes(self):
        recheck_bound_identity(self.plan())

    def test_a_changed_model_directory_is_refused(self):
        plan = self.plan()
        (self.model / "config.json").write_text('{"changed": true}\n')
        with self.assertRaises(ServeError) as caught:
            recheck_bound_identity(plan)
        self.assertEqual(caught.exception.code, "model_changed")

    def test_a_changed_adapter_is_refused(self):
        adapter = self.root / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        tensor = adapter / "adapters.safetensors"
        tensor.write_bytes(len(header).to_bytes(8, "little") + header + b"1234")
        plan = self.plan(adapter_path=str(adapter), adapter_identity=adapter_identity(adapter),
                         worker_argv=["server", "--adapter-path", str(adapter)])
        recheck_bound_identity(plan)
        tensor.write_bytes(tensor.read_bytes()[:-4] + b"5678")
        with self.assertRaises(ServeError) as caught:
            recheck_bound_identity(plan)
        self.assertEqual(caught.exception.code, "adapter_changed")

    def test_an_adapter_named_only_on_the_worker_argv_is_refused(self):
        plan = self.plan(worker_argv=["server", "--adapter-path", str(self.root / "missing")])
        with self.assertRaises(ServeError) as caught:
            recheck_bound_identity(plan)
        self.assertEqual(caught.exception.code, "adapter_changed")

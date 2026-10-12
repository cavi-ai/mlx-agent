import io
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from mlx_agent import lab
from mlx_agent.compute import ComputeBusy


class LabComputeRoutingTests(unittest.TestCase):
    def test_nested_feasibility_resume_obeys_existing_compute_owner(self):
        with patch("mlx_agent.compute.ComputeLease") as lease, patch("mlx_agent.lab._run_child") as child:
            child.return_value = subprocess.CompletedProcess([], 0, '{"ok":true,"data":{}}', "")
            lease.return_value.__enter__.side_effect = ComputeBusy("owned by training")
            with redirect_stdout(io.StringIO()):
                code = lab.run(["adapter", "feasibility", "resume", "fixture-run", "--cache", "/fixture/cache", "--json"])
            self.assertEqual(code, 2)
            child.assert_not_called()

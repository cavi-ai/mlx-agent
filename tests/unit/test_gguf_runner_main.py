import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mlx_agent import gguf_runner


class GgufRunnerMainTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.gguf = self.root / "model.gguf"
        self.gguf.write_bytes(b"GGUF" + b"\0" * 60)
        self.out = self.root / "converted"

    def main(self, *extra, dequantize=None, quantize=None, missing=()):
        output = io.StringIO()
        with patch.object(gguf_runner, "_missing_modules", return_value=list(missing)), \
                patch.object(gguf_runner, "dequantize", side_effect=dequantize), \
                patch.object(gguf_runner, "quantize", side_effect=quantize), redirect_stdout(output):
            code = gguf_runner.main(["--gguf", str(self.gguf), "--out", str(self.out), "--q-bits", "4", *extra])
        return code, output.getvalue()

    def test_refuses_missing_sources_existing_outputs_and_missing_modules(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(2, gguf_runner.main(["--gguf", str(self.root / "absent.gguf"), "--out", str(self.out), "--q-bits", "4"]))
        self.out.mkdir()
        self.assertEqual(2, self.main()[0])
        self.out.rmdir()
        code, log = self.main(missing=("torch",))
        self.assertEqual(2, code)
        self.assertIn("missing required modules: torch", log)
        intermediate = self.root / "converted.hf-intermediate"
        intermediate.mkdir()
        self.assertEqual(2, self.main()[0])

    def test_converts_through_an_intermediate_and_records_provenance(self):
        def quantize(work_dir, out_dir, q_bits):
            self.assertTrue(work_dir.is_dir())
            out_dir.mkdir()

        code, log = self.main("--signature", "sig", quantize=quantize)
        self.assertEqual(0, code)
        marker = json.loads((self.out / gguf_runner.PROVENANCE_NAME).read_text())
        self.assertEqual((4, "sig", "model.gguf"), (marker["q_bits"], marker["source"]["signature"], marker["source"]["name"]))
        self.assertFalse((self.root / "converted.hf-intermediate").exists())
        self.assertIn("removed intermediate", log)

    def test_failures_return_one_and_intermediate_can_be_kept(self):
        code, log = self.main(quantize=RuntimeError("mlx_lm.convert exited 1"))
        self.assertEqual(1, code)
        self.assertIn("conversion failed: mlx_lm.convert exited 1", log)
        self.assertFalse((self.root / "converted.hf-intermediate").exists())
        work = self.root / "work"
        code, _log = self.main("--work-dir", str(work), "--keep-intermediate", quantize=RuntimeError("boom"))
        self.assertEqual(1, code)
        self.assertTrue(work.is_dir())

    def test_quantize_requires_the_converter_and_its_success(self):
        with patch.object(gguf_runner.shutil, "which", return_value=None), self.assertRaisesRegex(RuntimeError, "not on PATH"):
            gguf_runner.quantize(self.root, self.out, 4)
        calls = []

        def run(argv, **options):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, len(calls) - 1)

        with patch.object(gguf_runner.shutil, "which", return_value="/bin/mlx_lm.convert"), \
                patch.object(gguf_runner.subprocess, "run", run), redirect_stdout(io.StringIO()):
            gguf_runner.quantize(self.root / "hf", self.out, 8)
            with self.assertRaisesRegex(RuntimeError, "exited 1"):
                gguf_runner.quantize(self.root / "hf", self.out, 8)
        self.assertEqual(["/bin/mlx_lm.convert", "--hf-path", str(self.root / "hf"), "--mlx-path", str(self.out),
                          "-q", "--q-bits", "8"], calls[0])

    def test_signature_samples_both_ends_of_large_files(self):
        data = bytes(range(40))
        self.gguf.write_bytes(data)
        expected = hashlib.sha256(b"40" + data[:4] + data[-4:]).hexdigest()
        self.assertEqual(expected, gguf_runner._signature(self.gguf, chunk_bytes=4))
        self.assertEqual(["mlx_agent_no_such_module"], gguf_runner._missing_modules(("json", "mlx_agent_no_such_module")))


if __name__ == "__main__":
    unittest.main()

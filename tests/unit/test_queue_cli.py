import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mlx_agent import queue_cli


class QueueCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.queue = self.root / "queues" / "convert-queue.json"

    def cli(self, request):
        path = self.root / "request.json"
        path.write_text(json.dumps(request))
        output = io.StringIO()
        with redirect_stdout(output):
            code = queue_cli.main(["--path", str(self.queue), "--receipts-dir", str(self.root / "receipts"),
                                   "queue", "--request", str(path), "--json"])
        return code, json.loads(output.getvalue())

    def test_default_queue_sits_beside_the_compute_root(self):
        with patch("mlx_agent.queue_cli.compute_root", return_value=Path("/state/compute")):
            self.assertEqual(Path("/state/queues/convert-queue.json"), queue_cli.default_path())

    def test_snapshot_and_clear_report_the_queue(self):
        code, payload = self.cli({"operation": "snapshot"})
        self.assertEqual(0, code)
        self.assertEqual({"schema_version": "1.1", "items": []}, payload["data"]["conversion_queue"])
        self.assertEqual(str(self.queue), payload["data"]["queue_path"])
        code, payload = self.cli({"operation": "clear"})
        self.assertEqual((0, 0), (code, payload["data"]["removed_count"]))

    def test_requests_outside_the_operation_grammar_are_rejected(self):
        for request, message in (([], "Unknown conversion queue operation."),
                                 ({"operation": "drop"}, "Unknown conversion queue operation."),
                                 ({"operation": "snapshot", "id": "x"}, "Unexpected conversion queue fields."),
                                 ({"operation": "move", "id": "x"}, "Missing conversion queue fields."),
                                 ({"operation": "adopt", "source": "x"}, "Missing conversion queue fields.")):
            with self.subTest(request=request):
                code, payload = self.cli(request)
                self.assertEqual((2, False), (code, payload["ok"]))
                self.assertEqual({"code": "queue_rejected", "message": message,
                                  "remediation": "Inspect the queue request and retained state."}, payload["error"])

    def test_oversized_queue_state_is_refused(self):
        self.queue.parent.mkdir(parents=True)
        self.queue.write_text("[]" + " " * 98)
        with patch.object(queue_cli, "MAXIMUM_BYTES", 50):
            code, payload = self.cli({"operation": "snapshot"})
        self.assertEqual(2, code)
        self.assertEqual("Queue state exceeds the supported size.", payload["error"]["message"])

    def test_reads_are_bounded_regular_files(self):
        with self.assertRaisesRegex(ValueError, "regular file"):
            queue_cli._read(self.root)
        large = self.root / "large.json"
        large.write_text("x" * 32)
        with patch.object(queue_cli, "MAXIMUM_BYTES", 16), self.assertRaisesRegex(ValueError, "bounded regular file"):
            queue_cli._read(large)
        self.assertEqual(b"x" * 32, queue_cli._read(large))

    def test_adoption_never_replaces_an_existing_queue(self):
        self.queue.parent.mkdir(parents=True)
        self.queue.write_text('{"schema_version": "1.1", "items": []}')
        source = self.root / "legacy.json"
        source.write_text("[]")
        code, payload = self.cli({"operation": "adopt-preview", "source": str(source)})
        self.assertEqual(2, code)
        self.assertEqual("Queue adoption never replaces an existing queue.", payload["error"]["message"])

    def test_unreadable_request_reports_an_error_envelope(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = queue_cli.main(["queue", "--request", str(self.root / "missing.json")])
        self.assertEqual(2, code)
        self.assertFalse(json.loads(output.getvalue())["ok"])


if __name__ == "__main__":
    unittest.main()

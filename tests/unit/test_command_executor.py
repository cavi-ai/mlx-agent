"""Tests for the stdin-only OpenCode custom-tool executor."""

import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mlx_agent.command_executor import CommandExecutorError, execute_command, read_command_arguments


class CommandExecutorTests(unittest.TestCase):
    def test_reads_bounded_utf8_arguments_and_passes_only_allowlisted_argv_to_core(self):
        received = []

        def core(argv):
            received.append(argv)
            return 0

        result = execute_command("opencode", "scout", io.BytesIO(b"--role coding --limit 2 --json"), core=core)
        self.assertEqual({"status": "ok", "provider": "opencode", "capability": "scout", "exit_code": 0}, result)
        self.assertEqual([["discover", "--role", "coding", "--limit", "2", "--json"]], received)

    def test_hostile_stdin_is_rejected_without_invoking_core_or_exposing_arguments(self):
        marker = Path(tempfile.gettempdir()) / "mlx-agent-command-executor-hostile-marker"
        if marker.exists():
            marker.unlink()
        called = []
        hostile = "--role coding; touch {0}".format(marker)
        with self.assertRaises(CommandExecutorError) as captured:
            execute_command("opencode", "scout", io.BytesIO(hostile.encode("utf-8")), core=lambda argv: called.append(argv))
        self.assertEqual([], called)
        self.assertFalse(marker.exists())
        self.assertNotIn(hostile, str(captured.exception))

    def test_subprocess_transport_never_interprets_hostile_stdin_as_a_shell_command(self):
        marker = Path(tempfile.gettempdir()) / "mlx-agent-command-executor-subprocess-marker"
        if marker.exists():
            marker.unlink()
        hostile = "--role coding; touch {0}".format(marker)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlx_agent.command_executor",
                "--provider",
                "opencode",
                "--capability",
                "scout",
            ],
            input=hostile.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            shell=False,
            check=False,
        )
        self.assertEqual(2, result.returncode)
        self.assertFalse(marker.exists())
        self.assertNotIn(hostile.encode("utf-8"), result.stdout)
        self.assertNotIn(hostile.encode("utf-8"), result.stderr)

    def test_rejects_untrusted_provider_invalid_utf8_and_oversized_stdin(self):
        with self.assertRaises(CommandExecutorError):
            execute_command("gemini", "scout", io.BytesIO(b"--limit 1"))
        with self.assertRaises(CommandExecutorError):
            read_command_arguments(io.BytesIO(b"\xff"))
        with self.assertRaises(CommandExecutorError):
            read_command_arguments(io.BytesIO(b"x" * 4097))


if __name__ == "__main__":
    unittest.main()

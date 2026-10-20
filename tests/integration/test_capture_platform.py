"""Real unprivileged platform boundaries; no model or coding client runs."""

import errno
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from mlx_agent.task_contracts import TaskError

try:
    from mlx_agent import capture_platform
except ImportError:
    capture_platform = None


SERVER = """
import json, socket, sys
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(('127.0.0.1', int(sys.argv[1])))
listener.listen(1)
connection = None
print(listener.getsockname()[1], flush=True)
for line in sys.stdin:
    command = line.strip()
    if command == 'accept':
        connection, _ = listener.accept()
        print('accepted', flush=True)
    elif command == 'close':
        listener.close()
        if connection is not None:
            connection.close()
        print('closed', flush=True)
    elif command == 'read':
        connection.settimeout(.15)
        try:
            payload = connection.recv(4096)
        except socket.timeout:
            payload = b''
        print(len(payload), flush=True)
    elif command == 'stop':
        break
"""


class OwnedServer:
    def __init__(self, port=0):
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-c", SERVER, str(port)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        self.port = int(self.response())

    def response(self):
        if not select.select([self.process.stdout], [], [], 5)[0]:
            raise AssertionError("Owned TCP fixture did not respond")
        line = self.process.stdout.readline()
        if not line:
            raise AssertionError("Owned TCP fixture exited")
        return line.strip()

    def command(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        return self.response()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()


@unittest.skipUnless(sys.platform == "darwin", "Darwin capture platform")
class CapturePeerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(capture_platform, "Capture platform implementation is missing")

    def server(self, port=0):
        server = OwnedServer(port)
        self.addCleanup(server.close)
        return server

    def connect(self, server):
        connection = socket.create_connection(("127.0.0.1", server.port), timeout=5)
        self.addCleanup(connection.close)
        self.assertEqual(server.command("accept"), "accepted")
        return connection

    def test_connected_peer_matches_accepted_socket(self):
        server = self.server()
        allowed = capture_platform.capture_process_identity(server.process.pid)
        connection = self.connect(server)
        evidence = capture_platform.verify_connected_peer(connection, [allowed])
        self.assertEqual(evidence["pid"], server.process.pid)
        self.assertEqual(evidence["identity"], allowed)
        self.assertEqual(evidence["local"], list(connection.getpeername()))
        self.assertEqual(evidence["peer"], list(connection.getsockname()))
        self.assertGreater(evidence["socket_id"], 0)
        self.assertEqual(server.command("read"), "0", "Verification must send no application bytes")

    def test_rebound_peer_blocks_body(self):
        original = self.server()
        allowed = capture_platform.capture_process_identity(original.process.pid)
        self.assertEqual(original.command("close"), "closed")
        replacement = self.server(original.port)
        connection = self.connect(replacement)
        error = None
        try:
            capture_platform.verify_connected_peer(connection, [allowed])
        except TaskError as caught:
            error = caught
        else:
            connection.sendall(b"project-contents-must-not-reach-rebound-port")
        self.assertEqual(replacement.command("read"), "0")
        self.assertIsNotNone(error)
        self.assertEqual(error.code, "capture_peer_mismatch")

    def test_peer_query_failure_is_unavailable(self):
        server = self.server()
        allowed = capture_platform.capture_process_identity(server.process.pid)
        connection = self.connect(server)
        with patch.object(capture_platform, "_socket_connections", side_effect=PermissionError("denied")):
            with self.assertRaises(TaskError) as caught:
                capture_platform.verify_connected_peer(connection, [allowed])
        self.assertEqual(caught.exception.code, "capture_peer_unavailable")
        self.assertEqual(server.command("read"), "0")

    def test_process_start_identity_drift_is_refused(self):
        server = self.server()
        allowed = capture_platform.capture_process_identity(server.process.pid)
        allowed["started_at"] = [0, 0]
        connection = self.connect(server)
        with self.assertRaises(TaskError) as caught:
            capture_platform.verify_connected_peer(connection, [allowed])
        self.assertEqual(caught.exception.code, "capture_peer_mismatch")
        self.assertEqual(server.command("read"), "0")


@unittest.skipUnless(sys.platform == "darwin", "Darwin capture platform")
class CaptureVolumeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(capture_platform, "Capture platform implementation is missing")
        self.assertNotEqual(os.geteuid(), 0, "This proof must run without elevated privileges")
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name).resolve()
        self.identifier = str(uuid4())
        self.root = self.parent / self.identifier
        self.capacity = 128 * 1024 * 1024 - 65536

    def create(self):
        volume = capture_platform.CaptureVolume.create(self.root, self.identifier, self.capacity)
        self.addCleanup(self.remove_if_present)
        return volume

    def remove_if_present(self):
        if self.root.exists():
            capture_platform.CaptureVolume.remove(self.root)

    def test_unprivileged_volume_has_hard_capacity(self):
        self.create()
        written = 0
        with capture_platform.CaptureVolume.open_owned(self.root, readonly=False) as mounted:
            capacity = os.statvfs(mounted)
            self.assertLessEqual(capacity.f_blocks * capacity.f_frsize + 65536, 128 * 1024 * 1024)
            exhausted = False
            for index in range(32):
                descriptor = os.open(mounted / ("payload-" + str(index)), os.O_CREAT | os.O_WRONLY, 0o600)
                try:
                    for _ in range(16):
                        written += os.write(descriptor, b"x" * (512 * 1024))
                    os.fsync(descriptor)
                except OSError as error:
                    self.assertIn(error.errno, (errno.ENOSPC, errno.EDQUOT))
                    exhausted = True
                    break
                finally:
                    os.close(descriptor)
            self.assertTrue(exhausted, "The filesystem must enforce its capacity")
            self.assertGreater(written, 0)
            self.assertLessEqual(written + 65536, 128 * 1024 * 1024)
            for path in mounted.glob("payload-*"):
                path.unlink()
            (mounted / "retained.txt").write_text("retained owned session")
        code = """
import errno, sys
from pathlib import Path
from mlx_agent.capture_platform import CaptureVolume
with CaptureVolume.open_owned(Path(sys.argv[1]), readonly=True) as mounted:
    try:
        (mounted / 'must-not-write').write_text('no')
    except OSError as error:
        assert error.errno == errno.EROFS, error
    else:
        raise AssertionError('Read-only recovery allowed a write')
    print((mounted / 'retained.txt').read_text())
"""
        reopened = subprocess.run([sys.executable, "-c", code, str(self.root)],
                                  capture_output=True, text=True, timeout=30)
        self.assertEqual(reopened.returncode, 0, reopened.stderr)
        self.assertEqual(reopened.stdout.strip(), "retained owned session")

    def test_many_small_files_cannot_escape_capacity(self):
        self.create()
        with capture_platform.CaptureVolume.open_owned(self.root, readonly=False) as mounted:
            exhausted = False
            for index in range(100000):
                try:
                    (mounted / ("item-" + str(index))).write_bytes(b"x")
                except OSError as error:
                    self.assertIn(error.errno, (errno.ENOSPC, errno.EDQUOT))
                    exhausted = True
                    break
            self.assertTrue(exhausted, "File/catalog growth must stay inside the same capacity")

    def test_wrong_volume_identity_refuses_detach_or_remove(self):
        self.create()
        marker = self.root / "volume.json"
        original = marker.read_bytes()
        value = json.loads(original)
        value["volume_uuid"] = str(uuid4())
        marker.write_text(json.dumps(value))
        try:
            with self.assertRaises(TaskError):
                capture_platform.CaptureVolume.remove(self.root)
            self.assertTrue(self.root.exists())
        finally:
            marker.write_bytes(original)

    def test_symlink_root_cannot_touch_an_unrelated_directory(self):
        unrelated = self.parent / "unrelated"
        unrelated.mkdir()
        sentinel = unrelated / "keep"
        sentinel.write_text("unchanged")
        self.root.symlink_to(unrelated, target_is_directory=True)
        with self.assertRaises(TaskError):
            capture_platform.CaptureVolume.remove(self.root)
        self.assertEqual(sentinel.read_text(), "unchanged")

    def test_attach_failure_retains_owned_recovery_data(self):
        self.create()
        from mlx_agent import capture_volume
        original = capture_volume._run

        def fail_attach(arguments):
            if arguments[1] == "attach":
                raise TaskError("capture_volume_unavailable", "Attach failed")
            return original(arguments)

        with patch.object(capture_volume, "_run", side_effect=fail_attach):
            with self.assertRaises(TaskError):
                with capture_platform.CaptureVolume.open_owned(self.root, readonly=False):
                    self.fail("Failed attachment cannot yield an artifact root")
        self.assertTrue((self.root / "volume.json").exists())

    def test_parent_traversal_is_refused_before_creation(self):
        target = self.parent / "missing" / ".." / self.identifier
        with self.assertRaises(TaskError):
            capture_platform.CaptureVolume.create(target, self.identifier, self.capacity)
        self.assertFalse((self.parent / "missing").exists())
        self.assertFalse(self.root.exists())

    def test_attach_error_after_mount_does_not_leave_an_untracked_mount(self):
        self.create()
        from mlx_agent import capture_volume
        original = capture_volume._run

        def fail_after_attach(arguments):
            result = original(arguments)
            if arguments[1] == "attach":
                raise TaskError("capture_volume_unavailable", "Attach response failed")
            return result

        with patch.object(capture_volume, "_run", side_effect=fail_after_attach):
            with self.assertRaises(TaskError):
                with capture_platform.CaptureVolume.open_owned(self.root, readonly=False):
                    self.fail("A failed attachment cannot yield storage")
        self.assertIsNone(capture_volume._attached(self.root))

    def test_detach_failure_retains_owned_recovery_data(self):
        self.create()
        from mlx_agent import capture_volume
        original = capture_volume._run

        def fail_detach(arguments):
            if arguments[1] == "detach":
                raise TaskError("capture_volume_unresolved", "Detach failed")
            return original(arguments)

        with patch.object(capture_volume, "_run", side_effect=fail_detach):
            with self.assertRaises(TaskError):
                with capture_platform.CaptureVolume.open_owned(self.root, readonly=False) as mounted:
                    (mounted / "retained.txt").write_text("keep on failure")
        self.assertTrue((self.root / "volume.json").exists())


if __name__ == "__main__":
    unittest.main()

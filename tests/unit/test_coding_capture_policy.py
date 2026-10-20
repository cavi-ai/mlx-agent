"""Policy admission observes existing state without starting or migrating it."""

import importlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from mlx_agent.task_contracts import TaskError

try:
    policy = importlib.import_module("mlx_agent.coding_capture_policy")
except ModuleNotFoundError:
    policy = None


class ReadOnlyPolicyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(policy, "Read-only policy admission is missing")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.database = self.root / "opencode.db"

    def create_database(self, credential=None, sources=None):
        with sqlite3.connect(str(self.database)) as connection:
            connection.execute("CREATE TABLE credential (id TEXT, integration_id TEXT, label TEXT, value TEXT, "
                               "connector_id TEXT, method_id TEXT, active INTEGER, time_created INTEGER, time_updated INTEGER)")
            connection.execute("CREATE TABLE kv (key TEXT, value TEXT, time_created INTEGER, time_updated INTEGER)")
            if credential is not None:
                connection.execute("INSERT INTO credential VALUES ('fixture','opencode','fixture','secret-not-selected',NULL,'key',?,1,1)",
                                   (credential,))
            if sources is not None:
                connection.execute("INSERT INTO kv VALUES ('wellknown:sources',?,1,1)", (json.dumps(sources),))

    def inspect(self):
        return policy.inspect_policy_sources(self.root, [self.database])

    def test_absent_database_and_neutral_legacy_auth_do_not_start_a_client(self):
        (self.root / "auth.json").write_text(json.dumps({"local": {"type": "api", "key": "not-returned"}}))
        with patch("subprocess.Popen") as start:
            result = self.inspect()
        start.assert_not_called()
        self.assertEqual(result["managed_policy"], "no_console_connection")
        self.assertNotIn("not-returned", json.dumps(result))
        self.assertFalse(self.database.exists())

    def test_legacy_console_and_wellknown_inputs_are_refused_before_bootstrap(self):
        for value in ({"opencode": {"type": "api", "key": "secret"}},
                      {"https://fixture.invalid": {"type": "wellknown", "key": "secret", "token": "secret"}}):
            (self.root / "auth.json").write_text(json.dumps(value))
            with self.subTest(value=list(value)), patch("subprocess.Popen") as start, self.assertRaises(TaskError):
                self.inspect()
            start.assert_not_called()
        self.assertFalse(self.database.exists())

    def test_known_empty_policy_schema_is_read_without_writes(self):
        self.create_database(sources=[])
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        result = self.inspect()
        self.assertEqual(result["managed_policy"], "no_console_connection")
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})
        self.assertNotIn("secret", json.dumps(result))

    def test_any_console_candidate_requires_an_authoritative_policy_snapshot(self):
        for active in (0, 1, None):
            with self.subTest(active=active):
                self.database.unlink(missing_ok=True)
                self.create_database(credential=active if active is not None else 0)
                if active is None:
                    with sqlite3.connect(str(self.database)) as connection:
                        connection.execute("UPDATE credential SET active=NULL")
                with self.assertRaises(TaskError) as failure:
                    self.inspect()
                self.assertEqual(failure.exception.code, "capture_client_unavailable")

    def test_wellknown_sources_remain_unavailable_without_credentials(self):
        self.create_database(sources=["https://fixture.invalid"])
        with self.assertRaises(TaskError):
            self.inspect()

    def test_credential_secret_column_is_never_read(self):
        self.create_database()
        connect = sqlite3.connect
        def metadata_only(*args, **kwargs):
            connection = connect(*args, **kwargs)
            connection.set_authorizer(lambda action, table, column, *_:
                sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and table == "credential" and column == "value"
                else sqlite3.SQLITE_OK)
            return connection
        with patch.object(policy.sqlite3, "connect", side_effect=metadata_only):
            self.assertEqual(self.inspect()["managed_policy"], "no_console_connection")

    def test_journal_unknown_schema_and_malformed_auth_are_unavailable(self):
        self.create_database()
        journal = self.root / "opencode.db-wal"
        journal.write_bytes(b"nonempty journal")
        with self.assertRaises(TaskError):
            self.inspect()
        journal.unlink()
        with sqlite3.connect(str(self.database)) as connection:
            connection.execute("DROP TABLE credential")
        with self.assertRaises(TaskError):
            self.inspect()
        self.database.unlink()
        (self.root / "auth.json").write_text('{"opencode":')
        with self.assertRaises(TaskError):
            self.inspect()

    def test_changed_policy_input_requires_new_preview(self):
        result = self.inspect()
        policy.verify_policy_sources(result)
        (self.root / "auth.json").write_text('{"opencode":{"type":"api","key":"secret"}}')
        with self.assertRaises(TaskError) as failure:
            policy.verify_policy_sources(result)
        self.assertEqual(failure.exception.code, "capture_context_changed")

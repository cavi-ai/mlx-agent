"""Read-only admission of OpenCode's persisted managed-policy inputs."""

import hashlib
import json
from pathlib import Path
import sqlite3

from .coding_capture_plugins import file_identity
from .task_contracts import MAX_REQUEST_BYTES, TaskError, encoded


def _unavailable(message):
    return TaskError("capture_client_unavailable", message,
                     "Capture requires bound local policy sources and an authoritative snapshot of any remote policy.")


def _snapshot(path):
    path = Path(path)
    resolved = path.resolve(strict=False)
    try:
        item = file_identity(resolved)
    except FileNotFoundError:
        return {"path": str(path), "resolved": str(resolved), "exists": False}
    return {"path": str(path), "resolved": str(resolved), "exists": True,
            "identity": item["identity"], "sha256": item["sha256"]}


def verify_policy_sources(manifest):
    try:
        if any(_snapshot(Path(item["path"])) != item for item in manifest["sources"]):
            raise ValueError("changed source")
    except (OSError, ValueError, KeyError, TaskError) as error:
        raise TaskError("capture_context_changed", "A persisted client policy input changed.",
                        "Create a new preview before starting the coding task.") from error


def _json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=pairs)
    encoded(value)
    return value


def inspect_policy_sources(data_root: Path, database_paths):
    """Static input facts only; the owned instance still needs runtime checks.

    An absent DB is insufficient because startup may import legacy auth.json.
    Immutable SQLite reads cannot create or update the original WAL/SHM files.
    A nonempty journal requires a separate authoritative snapshot instead.
    """
    data_root = Path(data_root)
    paths = [Path(path) for path in database_paths]
    if (not data_root.is_absolute() or len(paths) > 32 or len(set(paths)) != len(paths)
            or any(not path.is_absolute() for path in paths)):
        raise _unavailable("Client policy sources need explicit absolute local paths.")
    auth = data_root / "auth.json"
    inputs = [auth]
    for path in paths:
        inputs.extend((path, Path(str(path) + "-wal"), Path(str(path) + "-journal")))
    try:
        if auth.exists() and auth.stat().st_size > MAX_REQUEST_BYTES:
            raise _unavailable("Legacy client authentication metadata exceeds its supported size.")
        before = [_snapshot(path) for path in inputs]
        if auth.exists():
            entries = _json(auth.read_bytes())
            if not isinstance(entries, dict) or not all(isinstance(value, dict) for value in entries.values()):
                raise _unavailable("Legacy client authentication metadata is not understood.")
            # The pinned migration trims trailing slashes from integration IDs.
            # Reject candidates before a fresh DB could import their secrets.
            if any(key.rstrip("/") == "opencode" or value.get("type") == "wellknown"
                   for key, value in entries.items()):
                raise _unavailable("Legacy client state can introduce Console policy or remote configuration.")
        for path in paths:
            for suffix in ("-wal", "-journal"):
                journal = Path(str(path) + suffix)
                if journal.exists() and journal.stat().st_size:
                    raise _unavailable("Client policy has nonempty journal data and needs a stable read-only snapshot.")
            if not path.exists() or path.stat().st_size == 0:
                continue
            connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
            try:
                expected = {
                    "credential": {"id", "integration_id", "label", "value", "connector_id", "method_id",
                                   "active", "time_created", "time_updated"},
                    "kv": {"key", "value", "time_created", "time_updated"},
                }
                for table, columns in expected.items():
                    kind = connection.execute("SELECT type FROM sqlite_master WHERE name=?", (table,)).fetchone()
                    actual = {row[1] for row in connection.execute("PRAGMA table_info(" + table + ")")}
                    if kind != ("table",) or actual != columns:
                        raise _unavailable("The persisted client policy schema is unsupported.")
                # Never select credential.value. Even false/NULL active rows
                # can become the selected connection under the pinned resolver.
                candidate = connection.execute("SELECT 1 FROM credential WHERE integration_id='opencode' LIMIT 1").fetchone()
                if candidate is not None:
                    raise _unavailable("A Console connection needs its authoritative managed-policy snapshot.")
                size = connection.execute("SELECT length(CAST(value AS BLOB)) FROM kv WHERE key='wellknown:sources'").fetchone()
                if size is not None and (type(size[0]) is not int or not 0 <= size[0] <= MAX_REQUEST_BYTES):
                    raise _unavailable("Persisted remote configuration metadata exceeds its supported size.")
                source = connection.execute("SELECT value FROM kv WHERE key='wellknown:sources'").fetchone()
                if source is not None and _json(source[0]) != []:
                    raise _unavailable("Persisted remote configuration needs its authoritative policy snapshot.")
            finally:
                connection.close()
        result = {"schema": "coding-capture-policy-sources/1", "basis": "static_inputs", "sources": before,
                  "managed_policy": "no_console_connection", "remote_configuration": "no_wellknown_sources"}
        verify_policy_sources(result)
        result["sha256"] = hashlib.sha256(encoded(result)).hexdigest()
        return result
    except TaskError as error:
        if error.code in ("capture_client_unavailable", "capture_context_changed"):
            raise
        raise _unavailable("Persisted client policy inputs cannot be inspected safely.") from error
    except (OSError, ValueError, TypeError, UnicodeError, sqlite3.Error) as error:
        raise _unavailable("Persisted client policy inputs cannot be inspected safely.") from error

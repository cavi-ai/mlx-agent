import hashlib
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.coding_capture import OwnedSession
from mlx_agent.coding_capture_plugins import file_identity
from mlx_agent.task_contracts import TaskError

CLIENT = r'''
import base64, json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
args = sys.argv[1:]
if args[0] == "serve":
    if os.environ.get("FIXTURE_NO_URL"):
        raise SystemExit(1)
    expected = "Basic " + base64.b64encode(("opencode:" + os.environ["OPENCODE_SERVER_PASSWORD"]).encode()).decode()
    class API(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_GET(self):
            if self.path != "/api/plugin" or self.headers.get("Authorization") != expected:
                self.send_error(401)
                return
            status = os.environ.get("FIXTURE_PLUGIN_STATUS", "active")
            raw = json.dumps({"data": [{"id": "bundle-1", "state": {"status": status}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
    server = ThreadingHTTPServer(("127.0.0.1", 0), API)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(json.dumps({"url": "http://127.0.0.1:" + str(server.server_port)}), flush=True)
    sys.stdin.read()
elif args[0] == "run":
    assert sys.stdin.read() == "Reply with fixture text."
    for kind in ("step_start", "text", "step_finish"):
        print(json.dumps({"type": kind, "sessionID": "ses_fixture", "timestamp": 1760000000000}), flush=True)
else:
    assert args[:3] == ["session", "export", "ses_fixture"]
    print(json.dumps({"info": {"id": "ses_fixture", "projectID": "prj_fixture",
                               "time": {"created": 1760000000000, "updated": 1760000000001},
                               "location": {"directory": os.environ["PWD"]}},
                      "messages": [{"id": "msg_1", "type": "assistant", "content": [],
                                    "model": {"providerID": "capture", "id": "local"}}]}))
'''


class FakeHost:
    def __init__(self, client_path, manifest, profile_root, configuration, project_path, environment=None):
        self.profile = {"environment": dict(environment or {}, PWD=str(project_path))}
        self.server_fds = ()
        self.entered = self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *_):
        self.exited = True


class Artifacts:
    def __init__(self):
        self.written = {}

    def write_artifact(self, name, raw):
        self.written[name] = raw
        return {"name": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


@unittest.skipUnless(sys.platform == "darwin", "Owned process identities use Darwin libproc.")
class OwnedSessionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        self.client = root / "opencode"
        self.client.write_text("#!" + sys.executable + "\n" + CLIENT)
        self.client.chmod(0o700)
        self.bundle = root / "bundle-1.mjs"
        self.bundle.write_text("export default {};\n")
        self.manifest = {"client_path": str(self.client), "client_alias": str(self.client),
                         "inputs": [file_identity(self.client)],
                         "bundles": [dict(file_identity(self.bundle), id="bundle-1", legacy=True)]}
        self.profile, self.project = root / "profile", root / "project"
        self.profile.mkdir(mode=0o700)
        self.project.mkdir(mode=0o700)
        self.processes = {}

    def session(self, **environment):
        hosts = []

        def host(*args):
            hosts.append(FakeHost(*args))
            return hosts[-1]

        session = OwnedSession(self.manifest, self.profile, {}, self.project, "capture/local",
                               environment=dict(os.environ, **environment),
                               on_process=lambda role, identity: self.processes.update({role: identity}),
                               host_factory=host)
        return session, hosts

    def test_runs_one_prompt_exports_its_session_and_stops_the_server(self):
        session, hosts = self.session()
        storage = Artifacts()
        with session:
            result = session.run("Reply with fixture text.", storage)
            server = session.server
        self.assertIsNone(result["run"]["error"], result)
        self.assertEqual((result["run"]["session_id"], result["run"]["exit_code"]), ("ses_fixture", 0))
        self.assertIsNone(result["export"]["error"], result)
        self.assertEqual(result["export"]["session"]["info"]["location"]["directory"], str(self.project))
        self.assertEqual(set(self.processes), {"private_server", "client", "exporter"})
        self.assertEqual(set(storage.written), {"client.ndjson", "client-stderr.log", "session.json", "export-stderr.log"})
        self.assertIsNotNone(server.poll())
        self.assertTrue(hosts[0].entered and hosts[0].exited)

    def test_a_changed_extension_blocks_the_prompt(self):
        session, _ = self.session()
        with session:
            self.bundle.write_text("export default {changed: true};\n")
            with self.assertRaises(TaskError) as caught:
                session.run("Reply with fixture text.", Artifacts())
        self.assertEqual(caught.exception.code, "capture_context_changed")
        self.assertNotIn("client", self.processes)

    def test_an_inactive_extension_or_silent_server_fails_closed_and_cleans_up(self):
        for environment in ({"FIXTURE_PLUGIN_STATUS": "failed"}, {"FIXTURE_NO_URL": "1"}):
            with self.subTest(environment=environment):
                (self.profile / "server-stderr.log").unlink(missing_ok=True)
                session, hosts = self.session(**environment)
                with self.assertRaises(TaskError) as caught:
                    session.__enter__()
                self.assertEqual(caught.exception.code, "capture_client_unavailable")
                self.assertIsNotNone(session.server.poll())
                self.assertTrue(hosts[0].exited)


if __name__ == "__main__":
    unittest.main()

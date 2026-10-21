"""Compile existing local extensions without evaluating them or installing packages."""

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import socket

from .task_contracts import TaskError, encoded

_COMPILER = Path(__file__).parent / "resources" / "coding_capture" / "freeze.mjs"
_MAX_BYTES = 32 * 1024 * 1024


def _unavailable(message):
    return TaskError("capture_client_unavailable", message,
                     "Use existing absolute local extensions with supported hooks and fully resolved imports.")


def file_identity(path):
    """Path, inode identity and SHA-256 of one existing regular file."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise _unavailable("Extension inputs must be existing regular local files.")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "identity": [str(value) for value in (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)],
            "sha256": digest.hexdigest()}


def verify_extensions(manifest):
    """Recheck both original inputs and the immutable owned bundles before bootstrap."""
    try:
        if str(Path(manifest["client_alias"]).resolve(strict=True)) != manifest["client_path"]:
            raise ValueError("changed client alias")
        for item in manifest["inputs"] + manifest["bundles"]:
            current = file_identity(Path(item["path"]))
            if any(current[key] != item[key] for key in ("identity", "sha256")):
                raise ValueError("changed input")
    except (OSError, ValueError, KeyError, TaskError) as error:
        raise TaskError("capture_context_changed", "An extension input or frozen bundle changed.",
                        "Create a new preview before starting the coding task.") from error


def freeze_extensions(client_path: Path, targets, output_root: Path):
    # Reject package operations before even invoking the static compiler. A
    # cached npm package is still a package operation in OpenCode's supervisor.
    if not isinstance(targets, list) or not 0 <= len(targets) <= 64:
        raise _unavailable("Too many or invalid extension targets.")
    sources = []
    try:
        for target in targets:
            if not isinstance(target, str) or not Path(target).is_absolute():
                raise _unavailable("Package and relative extension operations cannot bootstrap a capture.")
            source = Path(target)
            file_identity(source)
            if source.suffix not in (".ts", ".js", ".mjs", ".mts", ".tsx", ".jsx"):
                raise _unavailable("Unsupported local extension source format.")
            sources.append(str(source))
    except OSError as error:
        raise _unavailable("A required local extension is missing.") from error
    if len(set(sources)) != len(sources):
        raise _unavailable("Repeated extension inputs are ambiguous.")
    alias = Path(client_path)
    if not alias.is_absolute():
        raise _unavailable("Select an absolute installed client executable.")
    try:
        client = alias.resolve(strict=True)
        file_identity(client)
    except OSError as error:
        raise _unavailable("The selected client executable is missing.") from error
    output_root = Path(output_root)
    if not output_root.is_absolute() or any(p.is_symlink() for p in (output_root, *output_root.parents)):
        raise _unavailable("Owned extension output cannot traverse symbolic links.")
    output_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("BUN_", "NODE_")) and key not in ("LD_PRELOAD", "DYLD_INSERT_LIBRARIES")}
    environment.update(BUN_BE_BUN="1", TMPDIR=str(output_root), XDG_CACHE_HOME=str(output_root / "cache"))
    try:
        run = subprocess.run([str(client), "--no-install", "--no-env-file", "--no-macros", str(_COMPILER)],
                             input=encoded({"sources": sources, "output": str(output_root), "maxBytes": _MAX_BYTES}),
                             cwd=output_root, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if run.returncode or len(run.stdout) > 1024 * 1024:
            raise _unavailable("Local extension imports could not be frozen without installation.")
        result = json.loads(run.stdout)
        result.update(client_path=str(client), client_alias=str(alias))
        result["inputs"].append(file_identity(client))
        result["inputs"].append(file_identity(_COMPILER))
        result["inputs"].extend(file_identity(Path(source)) for source in sources)
        verify_extensions(result)
        return result
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        raise _unavailable("The installed client's static compiler is unavailable or incompatible.") from error


def prepare_extension_bootstrap(manifest, profile_root: Path, configuration, project_path: Path, environment=None):
    """Prepare an explicit Start's owned host; this function executes no plugin."""
    verify_extensions(manifest)
    if any(not bundle["legacy"] for bundle in manifest["bundles"]):
        raise _unavailable("Native extension definitions require an adapter that can bind their bootstrap policy.")
    if not isinstance(configuration, dict) or configuration.get("plugins") or configuration.get("plugin"):
        raise _unavailable("Configured plugin operations must be resolved before preparing an owned profile.")
    if len(encoded(configuration)) > 1024 * 1024 or configuration.get("mcp") or configuration.get("providers"):
        raise _unavailable("Unsupported configuration sources need a new client context.")
    for provider in configuration.get("provider", {}).values():
        if provider.get("npm") != "@ai-sdk/openai-compatible":
            raise _unavailable("Capture bootstrap admits only the bundled compatible provider.")
        if any(model.get("provider", {}).get("npm") for model in provider.get("models", {}).values()):
            raise _unavailable("Model-specific provider packages are not admitted by this adapter.")
    profile_root = Path(profile_root)
    project_path = Path(project_path)
    if not profile_root.is_absolute() or any(p.is_symlink() for p in (profile_root, *profile_root.parents)):
        raise _unavailable("Owned profile roots cannot traverse symbolic links.")
    if not project_path.is_absolute() or not project_path.is_dir():
        raise _unavailable("An existing explicit project is needed for bootstrap.")
    profile_root.mkdir(mode=0o700, exist_ok=True)
    if profile_root.stat().st_mode & 0o077 or profile_root.stat().st_uid != os.getuid():
        raise _unavailable("Owned profile permissions changed.")
    directories = {name: profile_root / name for name in ("config", "data", "cache", "state", "tmp")}
    for path in directories.values():
        path.mkdir(mode=0o700, exist_ok=True)
        if path.is_symlink() or path.stat().st_mode & 0o077 or path.stat().st_uid != os.getuid():
            raise _unavailable("Owned profile directory permissions changed.")
    plugin_root = directories["config"] / "plugins"
    plugin_root.mkdir(mode=0o700, exist_ok=False)

    def write(path, raw):
        with path.open("xb") as output:
            os.chmod(path, 0o600)
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())

    def resource(name):
        target = profile_root / name
        write(target, (_COMPILER.parent / name).read_bytes())
        return target

    for name in ("proxy.mjs", "rpc.mjs", "worker.mjs"):
        resource(name)
    for index, bundle in enumerate(manifest["bundles"]):
        source = "import {proxyPlugin} from " + json.dumps(str(profile_root / "proxy.mjs")) + ";\n"
        source += "export default proxyPlugin(" + json.dumps(bundle["id"]) + ");\n"
        write(plugin_root / ("extension-" + str(index) + ".ts"), source.encode("utf-8"))
    original = profile_root / "original-config.json"
    effective = profile_root / "effective-config.json"
    receipt = profile_root / "bootstrap-receipt.json"
    plan = profile_root / "bootstrap-plan.json"
    write(original, encoded(configuration))
    write(plan, encoded({"schema": "coding-capture-bootstrap-plan/1", "bundles": manifest["bundles"],
                         "original": str(original), "effective": str(effective), "receipt": str(receipt),
                         "project": str(project_path), "config_root": str(directories["config"])}))
    base = os.environ if environment is None else environment
    child = {key: value for key, value in base.items()
             if not key.startswith(("BUN_", "NODE_", "OPENCODE_"))
             and key not in ("LD_PRELOAD", "DYLD_INSERT_LIBRARIES")}
    child.update({"XDG_CONFIG_HOME": str(directories["config"]), "XDG_DATA_HOME": str(directories["data"]),
                  "PWD": str(project_path),
                  "XDG_CACHE_HOME": str(directories["cache"]), "XDG_STATE_HOME": str(directories["state"]),
                  "TMPDIR": str(directories["tmp"]), "OPENCODE_CONFIG_DIR": str(directories["config"]),
                  "OPENCODE_DB": str(directories["data"] / "opencode" / "opencode.db"),
                  "OPENCODE_DISABLE_PROJECT_CONFIG": "1", "OPENCODE_DISABLE_MODELS_FETCH": "1",
                  "OPENCODE_DISABLE_AUTOUPDATE": "1",
                  "OPENCODE_DISABLE_FFF": "1", "MLX_AGENT_CAPTURE_BOOTSTRAP": str(plan),
                  "OPENCODE_CONFIG": str(effective)})
    return {"environment": child, "bootstrap_plan": str(plan), "receipt_path": str(receipt),
            "effective_config_path": str(effective), "config_root": str(directories["config"])}


class ExtensionHost:
    """One checked Bun child holds the original closures over an inherited socket."""

    def __init__(self, client_path, manifest, profile_root, configuration, project_path, environment=None):
        if str(Path(client_path).resolve()) != manifest["client_path"]:
            raise TaskError("capture_context_changed", "The selected client executable changed.")
        self.client_path = Path(manifest["client_path"])
        self.manifest, self.root = manifest, Path(profile_root)
        self.configuration, self.project_path = configuration, Path(project_path)
        self.environment = environment
        self.profile = self.process = self.channel = self.identity = None
        self.logs = []

    def __enter__(self):
        from .capture_platform import capture_process_identity
        self.profile = prepare_extension_bootstrap(self.manifest, self.root, self.configuration,
                                                   self.project_path, self.environment)
        worker, self.channel = socket.socketpair()
        self.channel.settimeout(30)
        try:
            for name in ("extension-stdout.log", "extension-stderr.log"):
                descriptor = os.open(self.root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                self.logs.append(os.fdopen(descriptor, "wb"))
            child = dict(self.profile["environment"], BUN_BE_BUN="1",
                         MLX_AGENT_CAPTURE_EXTENSION_FD=str(worker.fileno()))
            self.process = subprocess.Popen([str(self.client_path), "--no-install", "--no-env-file", "--no-macros",
                                             str(self.root / "worker.mjs")], cwd=self.project_path, env=child,
                                            stdin=subprocess.PIPE, stdout=self.logs[0], stderr=self.logs[1],
                                            pass_fds=(worker.fileno(),), start_new_session=True)
            worker.close()
            self.identity = capture_process_identity(self.process.pid)
            self.channel.sendall(encoded({"id": "engine:1", "method": "bootstrap", "params": {}}) + b"\n")
            raw = bytearray()
            while b"\n" not in raw:
                chunk = self.channel.recv(4096)
                if not chunk or len(raw) + len(chunk) > 1024 * 1024:
                    raise _unavailable("Owned extension bootstrap ended without a bounded reply.")
                raw.extend(chunk)
            reply = json.loads(raw)
            if reply.get("id") != "engine:1" or "error" in reply:
                raise _unavailable("A legacy extension has unsupported hooks or could not bootstrap.")
            result = reply["result"]
            effective = encoded(result["configuration"])
            for path, content in ((self.profile["effective_config_path"], effective),
                                  (self.profile["receipt_path"], encoded({"schema": "coding-capture-bootstrap/1",
                                   "plugins": result["plugins"], "effective_config_sha256": hashlib.sha256(effective).hexdigest()}))):
                with Path(path).open("xb") as output:
                    os.chmod(path, 0o600)
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
            verify_extensions(self.manifest)
            # settimeout uses O_NONBLOCK on the shared file description; the
            # child's stream reader needs a blocking descriptor after handoff.
            self.channel.setblocking(True)
            self.profile["environment"]["MLX_AGENT_CAPTURE_EXTENSION_FD"] = str(self.channel.fileno())
            return self
        except BaseException:
            worker.close()
            self.close()
            raise

    @property
    def server_fds(self):
        return (self.channel.fileno(),)

    def close(self):
        from .capture_platform import stop_owned
        try:
            if self.process is not None:
                stop_owned(self.process, self.identity)
        finally:
            if self.process is not None and self.process.stdin is not None:
                self.process.stdin.close()
            if self.channel is not None:
                self.channel.close()
            for stream in self.logs:
                stream.close()

    def __exit__(self, *_):
        self.close()

"""Confirmation-gated launcher for local MLX serving runtimes.

Serve is the only component that spawns processes, so every mutation goes
through the wire-style preview -> confirm -> receipt flow. Serve never
installs runtimes, never downloads models, binds only to 127.0.0.1, and stops
only processes that it started itself, verified against their receipts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .model_doctor import scan_wired_configs
from .transactions import _atomic_in_directory
from .wiring import require_secret_free_config, validate_health_endpoint


SERVE_RECEIPT_SCHEMA_VERSION = "1.0"
SERVE_RECEIPT_KIND = "serve"
MAX_TOKENS_MIN, MAX_TOKENS_MAX, MAX_TOKENS_DEFAULT = 256, 65536, 8192
PORT_MIN, PORT_MAX = 1, 65535
READINESS_DEADLINE_DEFAULT = 60.0
STOP_DEADLINE_SECONDS = 10.0
MAX_RECEIPTS = 50

_RECIPES_PATH = Path(__file__).resolve().parent / "resources" / "serve-recipes.json"


class ServeError(RuntimeError):
    """Classified serve failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def load_recipes(path=None):
    location = Path(path) if path is not None else _RECIPES_PATH
    value = json.loads(location.read_text(encoding="utf-8"))
    recipes = value.get("recipes")
    if not isinstance(recipes, dict) or not recipes:
        raise ValueError("serve recipe table is missing its recipes")
    for name, recipe in recipes.items():
        for key in ("executable", "argv", "default_port", "readiness", "install_hint"):
            if key not in recipe:
                raise ValueError("serve recipe {0} is missing {1}".format(name, key))
    return recipes


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _preview_hash(plan):
    canonical = json.dumps(
        {key: plan[key] for key in sorted(plan) if key != "preview_hash"},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def receipts_root(root=None):
    base = Path(root) if root is not None else Path.cwd()
    return base / ".mlx-agent-receipts" / "serve"


def plan_start(repo, runtime, recipes, port=None, max_tokens=MAX_TOKENS_DEFAULT,
               adapter_path=None):
    """Render the exact start plan; pure and side-effect free."""
    if not isinstance(repo, str) or not repo.strip() or "/" not in repo:
        raise ServeError(
            "invalid_repo",
            "serve requires a publisher/model repository identifier.",
            "Pass --repo as publisher/model exactly as it appears in the Hugging Face cache.",
        )
    recipe = recipes.get(runtime)
    if recipe is None:
        raise ServeError(
            "unsupported_runtime",
            "serve does not launch the {0} runtime.".format(runtime),
            "Ollama and LM Studio manage their own servers; serve supports: {0}.".format(
                ", ".join(sorted(recipes))
            ),
        )
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise ServeError(
            "invalid_arguments", "max_tokens must be an integer.",
            "Pass --max-tokens between {0} and {1}.".format(MAX_TOKENS_MIN, MAX_TOKENS_MAX),
        )
    if not MAX_TOKENS_MIN <= max_tokens <= MAX_TOKENS_MAX:
        raise ServeError(
            "invalid_arguments",
            "max_tokens is outside {0}-{1}.".format(MAX_TOKENS_MIN, MAX_TOKENS_MAX),
            "Pass a bounded --max-tokens value.",
        )
    selected_port = recipe["default_port"] if port is None else port
    if not isinstance(selected_port, int) or isinstance(selected_port, bool):
        raise ServeError(
            "invalid_arguments", "port must be an integer.",
            "Pass --port between {0} and {1}.".format(PORT_MIN, PORT_MAX),
        )
    if not PORT_MIN <= selected_port <= PORT_MAX:
        raise ServeError(
            "invalid_arguments",
            "port is outside {0}-{1}.".format(PORT_MIN, PORT_MAX),
            "Pass a valid loopback port.",
        )
    if adapter_path is not None and not recipe.get("adapter_argv"):
        raise ServeError(
            "unsupported_runtime",
            "The {0} recipe does not support adapter serving.".format(runtime),
            "Serve adapters with a runtime whose recipe declares adapter support.",
        )
    values = {
        "executable": recipe["executable"],
        "repo": repo,
        "port": str(selected_port),
        "max_tokens": str(max_tokens),
    }
    argv = [part.format(**values) for part in recipe["argv"]]
    if adapter_path is not None:
        argv.extend(
            part.format(adapter_path=str(adapter_path)) for part in recipe["adapter_argv"]
        )
    plan = {
        "repo": repo,
        "runtime": runtime,
        "port": selected_port,
        "max_tokens": max_tokens,
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "argv": argv,
        "readiness": recipe["readiness"].format(port=selected_port),
        "bind": "127.0.0.1",
    }
    plan["preview_hash"] = _preview_hash(plan)
    return plan


def _default_which(executable):
    return shutil.which(executable)


def _default_port_free(port):
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) != 0
    finally:
        probe.close()


def _default_spawn(argv, log_path):
    handle = open(log_path, "ab", buffering=0)
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process.pid


def _default_readiness(url, deadline_seconds, clock=time.monotonic, sleep=time.sleep):
    validate_health_endpoint(url)
    deadline = clock() + deadline_seconds
    while clock() < deadline:
        try:
            request = urllib.request.Request(url, method="GET")
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=2.0) as response:
                if 200 <= response.status < 400:
                    return True
        except (OSError, ValueError, urllib.error.URLError):
            pass
        sleep(0.5)
    return False


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pid_command(pid):
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def start_serve(plan, receipts_dir=None, confirm=False, preview_hash=None,
                which=None, model_present=None, port_free=None,
                wired_claims=None, spawn=None, readiness=None,
                readiness_deadline=READINESS_DEADLINE_DEFAULT, now=_utc_now,
                pid_alive=None):
    """Execute a reviewed start plan; the only mutating serve entry point."""
    which = which or _default_which
    port_free = port_free or _default_port_free
    spawn = spawn or _default_spawn
    readiness = readiness or _default_readiness
    pid_alive = pid_alive or _pid_alive
    root = receipts_root(receipts_dir)

    if not confirm:
        return {"status": "preview", "plan": plan, "requires_confirmation": True}
    if not preview_hash:
        raise ServeError(
            "preview_hash_required",
            "--confirm requires the hash from a reviewed serve preview.",
            "Run serve start without --confirm, inspect the plan, then pass --preview-hash.",
        )
    if preview_hash != plan["preview_hash"]:
        raise ServeError(
            "preview_stale",
            "The supplied preview hash does not match this start plan.",
            "Re-run serve start without --confirm and review the fresh plan.",
        )

    if which(plan["argv"][0]) is None:
        recipe_hint = load_recipes()[plan["runtime"]]["install_hint"]
        raise ServeError(
            "runtime_not_installed",
            "The {0} executable is not installed.".format(plan["argv"][0]),
            "Install it yourself ({0}); serve never installs runtimes.".format(recipe_hint),
        )
    if model_present is not None and not model_present(plan["repo"]):
        raise ServeError(
            "model_not_local",
            "The model is not present in a local inventory.",
            "Download it with the runtime's own pull command first; serve never downloads models.",
        )
    if not port_free(plan["port"]):
        raise ServeError(
            "port_in_use",
            "Port {0} is already bound on 127.0.0.1.".format(plan["port"]),
            "Pick a free --port, or stop the process that owns this one.",
        )
    if wired_claims is not None:
        claimed = wired_claims(plan["port"], plan["runtime"])
        if claimed:
            raise ServeError(
                "port_in_use",
                "A wired config already claims port {0}.".format(plan["port"]),
                "Move this server to its own --port.",
            )
    if plan["adapter_path"] is not None and not Path(plan["adapter_path"]).is_dir():
        raise ServeError(
            "invalid_arguments",
            "The adapter path does not exist: {0}".format(plan["adapter_path"]),
            "Pass an existing --adapter-path directory produced by mlx_lm.lora.",
        )

    receipt_path = root / "{0}.json".format(plan["port"])
    if receipt_path.exists():
        existing = _read_receipt(receipt_path)
        if existing is not None and pid_alive(existing.get("pid", -1)):
            raise ServeError(
                "port_in_use",
                "A serve receipt for port {0} is still live.".format(plan["port"]),
                "Run serve stop --port {0} first.".format(plan["port"]),
            )

    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "{0}.log".format(plan["port"])
    pid = spawn(plan["argv"], str(log_path))
    if not readiness(plan["readiness"], readiness_deadline):
        _terminate_pid(pid)
        raise ServeError(
            "readiness_timeout",
            "The server did not answer its readiness endpoint in time.",
            "Inspect the serve log at {0}.".format(log_path),
        )

    receipt = {
        "schema_version": SERVE_RECEIPT_SCHEMA_VERSION,
        "kind": SERVE_RECEIPT_KIND,
        "repo": plan["repo"],
        "runtime": plan["runtime"],
        "port": plan["port"],
        "argv": list(plan["argv"]),
        "pid": pid,
        "log_path": str(log_path),
        "readiness": plan["readiness"],
        "started_at": now(),
        "preview_hash": plan["preview_hash"],
    }
    content = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    _atomic_in_directory(root, receipt_path.name, content, 0o600)
    return {"status": "started", "receipt": receipt}


def _read_receipt(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("kind") != SERVE_RECEIPT_KIND:
        return None
    if not isinstance(value.get("pid"), int) or isinstance(value.get("pid"), bool):
        return None
    if not isinstance(value.get("argv"), list) or not isinstance(value.get("port"), int):
        return None
    return value


def status_serve(receipts_dir=None, pid_alive=_pid_alive, pid_command=_pid_command):
    """Cross-check serve receipts against live processes; read-only."""
    root = receipts_root(receipts_dir)
    entries = []
    if not root.is_dir():
        return entries
    for path in sorted(root.glob("*.json")):
        if len(entries) >= MAX_RECEIPTS:
            break
        receipt = _read_receipt(path)
        if receipt is None:
            continue
        alive = pid_alive(receipt["pid"])
        command = pid_command(receipt["pid"]) if alive else None
        entries.append({
            "receipt": str(path),
            "repo": receipt["repo"],
            "runtime": receipt["runtime"],
            "port": receipt["port"],
            "pid": receipt["pid"],
            "alive": alive,
            "argv_match": _argv_matches(receipt, command) if alive else False,
            "log_path": receipt.get("log_path"),
            "started_at": receipt.get("started_at"),
        })
    return entries


def _argv_matches(receipt, command, require_port=True):
    if not command:
        return False
    argv = receipt.get("argv") or []
    if not argv:
        return False
    executable = Path(str(argv[0])).name
    if executable not in command or str(receipt.get("repo")) not in command:
        return False
    if require_port:
        return "--port {0}".format(receipt.get("port")) in command
    return True


def _terminate_pid(pid, sig=signal.SIGTERM):
    try:
        os.kill(pid, sig)
    except OSError:
        return False
    return True


def stop_serve(port, receipts_dir=None, pid_alive=_pid_alive, pid_command=_pid_command,
               terminate=_terminate_pid, clock=time.monotonic, sleep=time.sleep):
    """Stop exactly the process a serve receipt owns; never kills by port scan."""
    root = receipts_root(receipts_dir)
    receipt_path = root / "{0}.json".format(port)
    receipt = _read_receipt(receipt_path) if receipt_path.is_file() else None
    if receipt is None:
        raise ServeError(
            "receipt_not_found",
            "No serve receipt exists for port {0}.".format(port),
            "serve stop only stops processes that serve started; stop foreign processes yourself.",
        )
    pid = receipt["pid"]
    if not pid_alive(pid):
        receipt_path.unlink()
        return {"status": "already_stopped", "port": port, "pid": pid}
    command = pid_command(pid)
    if not _argv_matches(receipt, command):
        raise ServeError(
            "pid_argv_mismatch",
            "The live pid {0} does not match the serve receipt; refusing to signal it.".format(pid),
            "Inspect the process yourself; the receipt is retained at {0}.".format(receipt_path),
        )
    terminate(pid, signal.SIGTERM)
    deadline = clock() + STOP_DEADLINE_SECONDS
    while clock() < deadline:
        if not pid_alive(pid):
            receipt_path.unlink()
            return {"status": "stopped", "port": port, "pid": pid}
        sleep(0.2)
    terminate(pid, signal.SIGKILL)
    if pid_alive(pid):
        raise ServeError(
            "stop_failed",
            "The process did not exit after SIGTERM and SIGKILL.",
            "Inspect pid {0} yourself; the receipt is retained at {1}.".format(pid, receipt_path),
        )
    receipt_path.unlink()
    return {"status": "stopped", "port": port, "pid": pid, "forced": True}


def wired_port_claim(roots):
    """Build a wired_claims(port, runtime) gate from Wire-managed configs."""
    wired = scan_wired_configs(roots)
    claims = {}
    for config in wired:
        endpoint = config.get("endpoint")
        if not endpoint:
            continue
        try:
            parsed = validate_health_endpoint(endpoint)
            from urllib.parse import urlsplit

            parts = urlsplit(parsed)
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError:
            continue
        claims.setdefault(port, config["runtime"])

    def check(port, runtime):
        owner = claims.get(port)
        return owner is not None and owner != runtime

    return check


def default_model_present(hf_cache=None):
    from .model_doctor import default_hf_cache, inspect_hf_cache

    cache = Path(hf_cache) if hf_cache is not None else default_hf_cache()
    known = {item["id"].casefold() for item in inspect_hf_cache(cache)}

    def check(repo):
        return repo.casefold() in known

    return check


LAUNCHD_LABEL_PREFIX = "com.mlx-agent.serve."
_LAUNCHD_LABEL = re.compile(r"\A" + LAUNCHD_LABEL_PREFIX.replace(".", r"\.") + r"(\d{1,5})\Z")


def launchd_label(port):
    return "{0}{1}".format(LAUNCHD_LABEL_PREFIX, port)


def render_launchd_plist(plan, log_path=None):
    """Render a deterministic launchd plist for a reviewed serve plan."""
    from xml.sax.saxutils import escape

    label = launchd_label(plan["port"])
    log = log_path or str(receipts_root() / "{0}.log".format(plan["port"]))
    arguments = "\n".join(
        "        <string>{0}</string>".format(escape(str(part))) for part in plan["argv"]
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        '<dict>\n'
        '    <key>Label</key>\n'
        '    <string>{label}</string>\n'
        '    <key>ProgramArguments</key>\n'
        '    <array>\n'
        '{arguments}\n'
        '    </array>\n'
        '    <key>RunAtLoad</key>\n'
        '    <true/>\n'
        '    <key>KeepAlive</key>\n'
        '    <false/>\n'
        '    <key>StandardOutPath</key>\n'
        '    <string>{log}</string>\n'
        '    <key>StandardErrorPath</key>\n'
        '    <string>{log}</string>\n'
        '</dict>\n'
        '</plist>\n'
    ).format(label=escape(label), arguments=arguments, log=escape(str(log)))


class LaunchdPlistAdapter:
    """Validate the exact launchd plist subset serve renders."""

    version = "1.0"
    runtime = "launchd"

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None

    def validate(self, content):
        if not isinstance(content, str):
            raise TypeError("plist content must be text")
        require_secret_free_config(content)
        lines = content.splitlines()
        if len(lines) < 8 or not lines[0].startswith("<?xml") or lines[2] != '<plist version="1.0">':
            raise ValueError("launchd plist must use the managed XML subset")
        if lines[-1] != "</plist>" or lines[-2] != "</dict>":
            raise ValueError("launchd plist must close its dict and plist elements")
        text = content
        for required in ("<key>Label</key>", "<key>ProgramArguments</key>", "<key>RunAtLoad</key>"):
            if required not in text:
                raise ValueError("launchd plist is missing {0}".format(required))
        label_match = re.search(r"<key>Label</key>\s*\n\s*<string>([^<]+)</string>", text)
        if label_match is None or _LAUNCHD_LABEL.fullmatch(label_match.group(1)) is None:
            raise ValueError("launchd plist label must use the managed serve prefix")
        return True

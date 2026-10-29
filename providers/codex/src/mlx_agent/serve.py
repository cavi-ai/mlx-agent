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
import sys
import secrets
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


def _local_weight_evidence(path):
    """Describe only locally declared weight facts; never resolve or fetch a repo."""
    root = Path(path)
    try:
        location = root / "config.json"
        if location.stat().st_size > 1024 * 1024:
            return {}
        raw = location.read_bytes()
        config = json.loads(raw)
        if not isinstance(config, dict):
            return {}
        evidence = {"model_config_fingerprint": hashlib.sha256(raw).hexdigest()}
        if any(root.glob("*.safetensors")):
            evidence["weight_format"] = "safetensors"
        quantization = config.get("quantization")
        if isinstance(quantization, dict):
            mode = quantization.get("mode", "affine")
            if mode not in ("affine", "mxfp4", "mxfp8", "nvfp4"):
                mode = "quantized"
            if config.get("vq_modules") or config.get("vq_ple"):
                description = "VQ + per-tensor " + mode
            elif any(isinstance(value, dict) for value in quantization.values()):
                description = "per-tensor " + mode
            else:
                bits = quantization.get("bits")
                description = mode + (" {}-bit".format(bits) if type(bits) is int and 1 <= bits <= 16 else "")
            evidence["weight_quantization"] = description
        return evidence
    except (OSError, ValueError):
        return {}


def resolve_serve_runtime(repo, path, runtime, recipes, hf_cache=None,
                          manifests=None, registries=None, backend_root=None):
    """Select a declared local loader and its isolated executable; never fetch or import model code."""
    from .backends import backend_target, choose_backend, load_manifests, load_registries, lookup
    from .convert import cached_snapshot

    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests, root=backend_root) if registries is None else registries
    local = Path(path).expanduser() if path else cached_snapshot(repo, hf_cache) if repo else None
    config = {}
    try:
        location = local / "config.json"
        if location.stat().st_size > 1024 * 1024:
            raise ValueError("config too large")
        config = json.loads(location.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
    except (AttributeError, OSError, ValueError, TypeError):
        config = {}
        if runtime == "auto":
            raise ServeError("model_config_unavailable", "Automatic runtime selection needs a readable local config.json.",
                             "Select a downloaded local model or explicitly review its serving runtime.") from None
    if runtime == "strata" or (runtime == "auto" and config.get("serving_runtime") == "strata"):
        from .strata_runtime import StrataError, installed_runtime, read_model
        try:
            if path is None:
                raise StrataError("strata_model_required", "Strata serves prepared local GGUF profiles only.")
            read_model(path)
            return "strata", installed_runtime()["python"]
        except StrataError as error:
            raise ServeError(error.code, str(error), error.remediation) from error
    hits = lookup(config.get("model_type"), manifests, registries) if config.get("model_type") else []
    if runtime == "auto":
        backend = choose_backend(hits, "text_llm", "vision_config" in config)
        if backend not in ("mlx-lm", "mlx-vlm"):
            raise ServeError("unsupported_runtime", "No declared serving backend implements this model architecture.",
                             "Install a backend with a vetted model adapter; remote runtime code is never executed.")
        runtime = "mlx_lm" if backend == "mlx-lm" else backend
    else:
        backend = "mlx-lm" if runtime == "mlx_lm" else runtime
        if config.get("model_type") and not any(hit["backend"] == backend for hit in hits):
            raise ServeError("unsupported_runtime", "The selected runtime does not implement this model architecture.",
                             "Review a fresh serve plan with --runtime auto.")
    recipe = recipes.get(runtime)
    if recipe is None:
        raise ServeError("unsupported_runtime", "No serving recipe exists for this runtime.", "Choose a declared serving runtime.")
    manifest = manifests[backend]
    if not manifest["builtin"] and not registries.get(backend, {}).get("installed"):
        raise ServeError("runtime_not_installed", "The declared {0} backend is not installed.".format(backend),
                         "Preview and confirm backend install {0}, then review serving again.".format(backend))
    # Optional backends are deliberately outside PATH. Bind the exact reviewed
    # executable into both eager and JIT plans instead of changing global PATH.
    executable = recipe["executable"] if manifest["builtin"] else str(
        backend_target(manifest, backend_root) / "bin" / recipe["executable"])
    return runtime, executable


def _adapter_evidence(path):
    from .lora import adapter_identity
    try:
        local = Path(path).expanduser().absolute()
        identity = adapter_identity(local)
    except (TypeError, ValueError, OSError):
        identity = None
    if identity is None:
        raise ServeError("invalid_adapter", "Adapter files are missing, invalid or linked.",
                         "Select a completed adapter with adapter_config.json and adapters.safetensors.")
    return str(local), identity


def _require_adapter_identity(plan):
    if plan.get("adapter_path") is None:
        return
    try:
        path, identity = _adapter_evidence(plan["adapter_path"])
        if path == plan["adapter_path"] and identity == plan.get("adapter_identity"):
            return
    except ServeError:
        pass
    raise ServeError("adapter_changed", "Adapter files changed after the serving preview.",
                     "Review a fresh serve plan before starting this adapter.")


def _require_training_binding(plan):
    binding = plan.get("training_binding")
    if binding is None:
        return
    try:
        from .training_artifacts import read_trained_adapter
        _, current = read_trained_adapter(binding["training_receipt"])
        if current == binding:
            return
    except (OSError, ValueError, TypeError, KeyError):
        pass
    raise ServeError("training_receipt_changed", "The training receipt or its pinned artifacts changed.",
                     "Review a fresh serving preview for the completed training receipt.")


def _require_prefill_identity(plan):
    if "prefill_kernel" in plan:
        from .prefill_runtime import can_wrap_runtime, identity
        configured = plan["prefill_kernel"]
        facts = {k: v for k, v in configured.items() if k not in ("adapter_enabled", "runtime_executable")}
        if (facts != identity() or (configured.get("adapter_enabled")
                and not can_wrap_runtime(configured["runtime_executable"]))):
            raise ServeError("prefill_kernel_changed", "Prompt kernel source or dependencies changed.",
                             "Review a fresh serving preview.")


def _require_decode_identity(plan):
    if "decode_profile" in plan:
        from .prompt_lookup_runtime import can_wrap_runtime, identity
        configured = plan["decode_profile"]
        facts = {k: v for k, v in configured.items() if k not in ("adapter_enabled", "runtime_executable")}
        if (facts != identity() or (configured.get("adapter_enabled")
                and not can_wrap_runtime(configured["runtime_executable"]))):
            raise ServeError("decode_profile_changed", "Code-edit runtime source or dependencies changed.",
                             "Review a fresh serving preview.")


def plan_start(repo, runtime, recipes, port=None, max_tokens=MAX_TOKENS_DEFAULT,
               adapter_path=None, path=None, jit=False, receipts_dir=None, hf_cache=None, memory_policy=None,
               runtime_executable=None, strata_speculation=False,
               strata_expert_memory_gb=None, strata_context_tokens=None, training_receipt=None, thinking="default",
               m5_prefill=False, prompt_lookup=False):
    """Render the start plan and local file identities without side effects."""
    if thinking not in ("default", "off"):
        raise ServeError("invalid_arguments", "thinking must be default or off.", "Pass --thinking off or omit it.")
    if type(m5_prefill) is not bool or (m5_prefill and runtime != "mlx_lm"):
        raise ServeError("invalid_arguments", "M5 prompt acceleration requires MLX-LM.",
                         "Use --runtime auto or mlx_lm, or omit --m5-prefill.")
    if type(prompt_lookup) is not bool or (prompt_lookup and runtime != "mlx_lm"):
        raise ServeError("invalid_arguments", "Code-edit acceleration requires MLX-LM.",
                         "Use --runtime auto or mlx_lm, or omit --prompt-lookup.")
    binding = None
    if training_receipt is not None:
        if repo is not None or path is not None or adapter_path is not None:
            raise ServeError("invalid_arguments", "A training receipt selects its own base model and adapter.",
                             "Use --training-receipt without --repo, --path or --adapter-path.")
        if runtime != "mlx_lm":
            raise ServeError("unsupported_runtime", "Text training receipts require the MLX-LM serving runtime.",
                             "Use --runtime auto or mlx_lm.")
        try:
            from .training_artifacts import read_trained_adapter
            _, binding = read_trained_adapter(training_receipt)
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ServeError("invalid_training_receipt", str(error), "Select a completed receipt whose base and adapter files are unchanged.") from error
        path, adapter_path = binding["base_path"], binding["adapter_path"]
    if memory_policy is not None:
        from .jit_serve import validate_memory_policy
        try:
            memory_policy = validate_memory_policy(memory_policy)
        except ValueError as error:
            raise ServeError("invalid_arguments", str(error), "Correct the memory policy.") from error
        if not jit:
            raise ServeError("jit_required", "Memory policy requires a JIT endpoint.", "Use --jit.")
    has_repo = isinstance(repo, str) and bool(repo.strip())
    has_path = isinstance(path, str) and bool(path.strip())
    if has_repo == has_path:
        raise ServeError(
            "invalid_arguments",
            "serve requires exactly one of a repository identifier or a local path.",
            "Pass --repo as publisher/model from the Hugging Face cache, or --path to a local model directory.",
        )
    if has_repo and "/" not in repo:
        raise ServeError(
            "invalid_repo",
            "serve requires a publisher/model repository identifier.",
            "Pass --repo as publisher/model exactly as it appears in the Hugging Face cache.",
        )
    local_path = None
    if has_path:
        local_path = os.path.abspath(os.path.expanduser(path.strip()))
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
    adapter_identity = None
    if adapter_path is not None:
        adapter_path, adapter_identity = _adapter_evidence(adapter_path)
    model_value = repo.strip() if has_repo else local_path
    values = {
        "executable": runtime_executable or recipe["executable"],
        "repo": model_value,
        "port": str(selected_port),
        "max_tokens": str(max_tokens),
        "runtime_adapter": str(Path(__file__).with_name("strata_runtime.py").resolve()),
    }
    strata_model = strata_runtime = None
    if runtime != "strata" and (strata_expert_memory_gb is not None or strata_context_tokens is not None):
        raise ServeError("invalid_arguments", "Strata memory controls require the Strata runtime.",
                         "Select a prepared local Strata profile.")
    if runtime == "strata":
        from .strata_runtime import StrataError, installed_runtime, read_model, memory_estimate
        try:
            if not has_path or adapter_path is not None or type(strata_speculation) is not bool:
                raise StrataError("strata_model_required", "Strata requires a local GGUF profile without a LoRA adapter.")
            strata_model = read_model(local_path)
            strata_context_tokens = 131072 if strata_context_tokens is None else strata_context_tokens
            strata_memory_bytes = memory_estimate(strata_model, strata_expert_memory_gb, strata_context_tokens)
            strata_runtime = installed_runtime()
            if runtime_executable is not None and runtime_executable != strata_runtime["python"]:
                raise StrataError("strata_not_installed", "The reviewed Strata executable changed.")
            values["executable"] = strata_runtime["python"]
        except StrataError as error:
            raise ServeError(error.code, str(error), error.remediation) from error
    argv = [part.format(**values) for part in recipe["argv"]]
    if strata_model:
        argv += ["--runtime-root", strata_runtime["root"], "--speculation", "on" if strata_speculation else "off",
                 "--context-tokens", str(strata_context_tokens),
                 "--expected-model-fingerprint", strata_model["fingerprint"],
                 "--expected-runtime-fingerprint", strata_runtime["fingerprint"]]
        if strata_expert_memory_gb is not None:
            argv += ["--expert-memory-gb", str(strata_expert_memory_gb)]
    if adapter_path is not None:
        argv.extend(
            part.format(adapter_path=str(adapter_path)) for part in recipe["adapter_argv"]
        )
    if thinking == "off" and recipe.get("thinking_off_argv"):
        # Literal arguments: the chat-template JSON is not a format template.
        argv.extend(recipe["thinking_off_argv"])
    native_argv = argv[:]
    prefill_kernel = None
    if m5_prefill:
        from .prefill_runtime import can_wrap_runtime, identity
        prefill_kernel = identity()
        prefill_kernel.update(adapter_enabled=can_wrap_runtime(values["executable"]),
                              runtime_executable=values["executable"])
        if prefill_kernel["adapter_enabled"]:
            argv = [sys.executable, str(Path(__file__).with_name("prefill_runtime.py").resolve()),
                    "--expected-kernel-fingerprint", prefill_kernel["fingerprint"]] + argv[1:]
    decode_profile = None
    if prompt_lookup:
        from .prompt_lookup_runtime import can_wrap_runtime, identity
        decode_profile = identity()
        decode_profile.update(adapter_enabled=can_wrap_runtime(values["executable"]),
                              runtime_executable=values["executable"])
        if decode_profile["adapter_enabled"]:
            argv = [sys.executable, str(Path(__file__).with_name("prompt_lookup_runtime.py").resolve()),
                    "--expected-decode-fingerprint", decode_profile["fingerprint"]]
            if prefill_kernel is not None and prefill_kernel["adapter_enabled"]:
                argv += ["--m5-prefill-fingerprint", prefill_kernel["fingerprint"]]
            argv += native_argv[1:]
    plan = {
        "repo": repo.strip() if has_repo else None,
        "path": local_path,
        "runtime": runtime,
        "port": selected_port,
        "max_tokens": max_tokens,
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "argv": argv,
        "readiness": recipe["readiness"].format(port=selected_port),
        "bind": "127.0.0.1",
    }
    if strata_model:
        plan.update(strata_model_fingerprint=strata_model["fingerprint"],
                    strata_runtime_fingerprint=strata_runtime["fingerprint"],
                    weight_format="gguf", weight_quantization=strata_model["quantization"], weights_preserved=True,
                    strata_speculation=strata_speculation, strata_expert_memory_gb=strata_expert_memory_gb,
                    strata_context_tokens=strata_context_tokens, strata_memory_estimate_bytes=strata_memory_bytes,
                    readiness_deadline=900)
    elif has_path:
        plan.update(_local_weight_evidence(local_path))
    if prefill_kernel is not None:
        plan["prefill_kernel"] = prefill_kernel
    if decode_profile is not None:
        plan["decode_profile"] = decode_profile
    if adapter_path is not None:
        plan["adapter_identity"] = adapter_identity
    if thinking == "off":
        plan["thinking"] = "off" if recipe.get("thinking_off_argv") else "unsupported"
    if binding is not None:
        plan["training_binding"] = binding
    if jit:
        from .convert import cached_snapshot
        from .jit_serve import model_fingerprint
        local = Path(local_path) if local_path else cached_snapshot(repo.strip(), hf_cache)
        if local is None or not local.is_dir():
            raise ServeError("model_not_local", "JIT requires existing local model files.",
                             "Select an existing local model directory; serving never downloads weights.")
        local = local.resolve()
        worker = list(argv)
        worker[worker.index("--model") + 1] = str(local)
        worker[worker.index("--port") + 1] = "{worker_port}"
        # Both supported runtimes expose --host. The VLM default binds all
        # interfaces, so explicitly fence the private worker to loopback.
        if "--host" not in worker:
            worker.extend(["--host", "127.0.0.1"])
        root = receipts_root(receipts_dir).absolute()
        plan.update(jit=True, local_path=str(local), fingerprint=model_fingerprint(local),
                    worker_argv=worker, control_config=str(root / "{}.jit-config.json".format(selected_port)))
        plan["argv"] = [sys.executable, str(Path(__file__).with_name("jit_serve.py").resolve()),
                        "--model", model_value, "--port", str(selected_port),
                        "--config", plan["control_config"]]
        if memory_policy is not None:
            plan["memory_policy"] = memory_policy
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
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    return process.pid


def _default_readiness(url, deadline_seconds, clock=time.monotonic, sleep=time.sleep, child_pid=None):
    validate_health_endpoint(url)
    deadline = clock() + deadline_seconds
    while clock() < deadline:
        if child_pid is not None:
            try:
                if os.waitpid(child_pid, os.WNOHANG)[0] == child_pid:
                    return False
            except ChildProcessError:
                if not _pid_alive(child_pid):
                    return False
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


def _default_path_present(path):
    return Path(path).is_dir()


def start_serve(plan, receipts_dir=None, confirm=False, preview_hash=None,
                which=None, model_present=None, path_present=None, port_free=None,
                wired_claims=None, spawn=None, readiness=None,
                readiness_deadline=READINESS_DEADLINE_DEFAULT, now=_utc_now,
                pid_alive=None):
    """Execute a reviewed start plan; the only mutating serve entry point."""
    which = which or _default_which
    port_free = port_free or _default_port_free
    spawn = spawn or _default_spawn
    readiness = readiness or (lambda url, deadline: _default_readiness(url, deadline, child_pid=pid if spawn is _default_spawn else None))
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
    _require_training_binding(plan)
    _require_adapter_identity(plan)
    _require_prefill_identity(plan)
    _require_decode_identity(plan)

    if plan["runtime"] == "strata" and not plan.get("jit"):
        from .jit_serve import available_memory_bytes
        available = available_memory_bytes()
        required = plan["strata_memory_estimate_bytes"] + 2_000_000_000
        if type(available) is not int or available < 0:
            raise ServeError("memory_unknown", "Available memory could not be measured.",
                             "Retry after memory reporting is available.")
        if available < required:
            raise ServeError("insufficient_headroom", "Insufficient memory for the reviewed Strata settings and 2 GB reserve.",
                             "Reduce the expert budget or context capacity, or free memory.")

    if which(plan["argv"][0]) is None:
        recipe_hint = load_recipes()[plan["runtime"]]["install_hint"]
        raise ServeError(
            "runtime_not_installed",
            "The {0} executable is not installed.".format(plan["argv"][0]),
            "Install it yourself ({0}); serve never installs runtimes.".format(recipe_hint),
        )
    if plan.get("jit"):
        from .jit_serve import model_fingerprint
        if which(plan["worker_argv"][0]) is None:
            raise ServeError("runtime_not_installed", "The model worker executable is not installed.",
                             load_recipes()[plan["runtime"]]["install_hint"])
        if model_fingerprint(plan["local_path"]) != plan["fingerprint"]:
            raise ServeError("preview_stale", "Local model files changed after preview.",
                             "Review a fresh JIT serve plan.")
    if plan["repo"] is not None and model_present is not None and not model_present(plan["repo"]):
        raise ServeError(
            "model_not_local",
            "The model is not present in a local inventory.",
            "Download it with the runtime's own pull command first; serve never downloads models.",
        )
    if plan["path"] is not None:
        present = path_present or _default_path_present
        if not present(plan["path"]):
            raise ServeError(
                "model_not_local",
                "The model directory is not present: {0}".format(plan["path"]),
                "Point --path at an existing local model directory; serve never downloads models.",
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
    receipt_path = root / "{0}.json".format(plan["port"])
    if receipt_path.exists():
        existing = _read_receipt(receipt_path)
        if existing is not None and (pid_alive(existing.get("pid", -1)) or _owned_jit_worker(existing, root)):
            raise ServeError(
                "port_in_use",
                "A serve receipt for port {0} is still live.".format(plan["port"]),
                "Run serve stop --port {0} first.".format(plan["port"]),
            )

    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "{0}.log".format(plan["port"])
    if plan.get("jit"):
        config_path = Path(plan["control_config"])
        if config_path.parent != root.absolute() or config_path.is_symlink():
            raise ServeError("invalid_arguments", "Unsafe JIT configuration path.", "Review a fresh serve plan.")
        config = {key: plan[key] for key in ("local_path", "fingerprint", "worker_argv", "max_tokens")}
        if plan.get("adapter_path") is not None:
            config.update(adapter_path=plan["adapter_path"], adapter_identity=plan["adapter_identity"])
        if plan.get("training_binding") is not None:
            config["training_binding"] = plan["training_binding"]
        if "readiness_deadline" in plan:
            config["worker_readiness_seconds"] = plan["readiness_deadline"]
        if plan["runtime"] == "strata":
            config["runtime"] = "strata"
            config["strata_expert_memory_gb"] = plan.get("strata_expert_memory_gb")
            config["strata_context_tokens"] = plan.get("strata_context_tokens", 131072)
        config.update(model=plan["repo"] or plan["path"], control_token=secrets.token_hex(32),
                      memory_policy=plan.get("memory_policy"), config_path=str(config_path),
                      state_path=str(root.absolute() / "{}.jit-state.json".format(plan["port"])),
                      log_path=str(root.absolute() / "{}.worker.log".format(plan["port"])))
        _atomic_in_directory(root, config_path.name, json.dumps(config).encode(), 0o600)
    pid = spawn(plan["argv"], str(log_path))
    if not readiness(plan["readiness"], plan.get("readiness_deadline", readiness_deadline)):
        _terminate_pid(pid)
        raise ServeError(
            "readiness_timeout",
            "The server did not answer its readiness endpoint in time.",
            "Inspect the serve log at {0}.".format(log_path),
        )
    try:
        _require_training_binding(plan)
        _require_adapter_identity(plan)
        _require_prefill_identity(plan)
        _require_decode_identity(plan)
    except ServeError:
        _terminate_pid(pid)
        raise

    receipt = {
        "schema_version": SERVE_RECEIPT_SCHEMA_VERSION,
        "kind": SERVE_RECEIPT_KIND,
        "repo": plan["repo"],
        "path": plan["path"],
        "runtime": plan["runtime"],
        "port": plan["port"],
        "argv": list(plan["argv"]),
        "pid": pid,
        "log_path": str(log_path),
        "readiness": plan["readiness"],
        "started_at": now(),
        "preview_hash": plan["preview_hash"],
    }
    if plan.get("jit"):
        receipt["jit"] = True
    if plan.get("adapter_path") is not None:
        receipt.update(adapter_path=plan["adapter_path"], adapter_identity=plan["adapter_identity"])
    if plan.get("training_binding") is not None:
        receipt["training_binding"] = plan["training_binding"]
    if "prefill_kernel" in plan:
        receipt["prefill_kernel"] = plan["prefill_kernel"]
    if "decode_profile" in plan:
        receipt["decode_profile"] = plan["decode_profile"]
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
            "path": receipt.get("path"),
            "runtime": receipt["runtime"],
            "port": receipt["port"],
            "pid": receipt["pid"],
            "alive": alive,
            "argv_match": _argv_matches(receipt, command) if alive else False,
            "log_path": receipt.get("log_path"),
            "started_at": receipt.get("started_at"),
        })
        entries[-1].update(_adapter_metadata(receipt, root))
        if receipt.get("training_binding") is not None:
            entries[-1]["training_binding"] = receipt["training_binding"]
        if "prefill_kernel" in receipt:
            entries[-1]["prefill_kernel"] = receipt["prefill_kernel"]
        if "decode_profile" in receipt:
            entries[-1]["decode_profile"] = receipt["decode_profile"]
        if receipt.get("jit"):
            entries[-1]["jit"] = True
            if alive and entries[-1]["argv_match"]:
                try:
                    entries[-1].update(_jit_control(receipt, root, "status"))
                except ServeError:
                    entries[-1]["model_state"] = "unknown"
            else:
                entries[-1]["model_state"] = "unknown"
    return entries


def _adapter_metadata(receipt, root):
    if receipt.get("adapter_path") is not None:
        return {"adapter_path": receipt["adapter_path"], "adapter_identity": receipt.get("adapter_identity")}
    argv = receipt.get("argv", [])
    if receipt.get("jit"):
        try:
            argv = _jit_config(receipt, root)["worker_argv"]
        except (ServeError, KeyError):
            return {"adapter_status": "unknown"}
    if not isinstance(argv, list) or any(not isinstance(value, str) for value in argv):
        return {"adapter_status": "unknown"}
    if "--adapter-path" in argv:
        index = argv.index("--adapter-path") + 1
        if index >= len(argv) or not argv[index]:
            return {"adapter_status": "unknown"}
        return {"adapter_path": argv[index], "adapter_identity": None}
    return {}


def _argv_matches(receipt, command, require_port=True):
    if not command:
        return False
    argv = receipt.get("argv") or []
    if not argv:
        return False
    executable = Path(str(argv[0])).name
    # macOS framework Python reports its bundle binary in ps, not the
    # interpreter alias used to launch it. Bind a script recipe to its
    # absolute entry point instead of accepting any Python process.
    if len(argv) > 1 and str(argv[1]).endswith(".py") and Path(str(argv[1])).is_absolute():
        executable = str(argv[1])
    model = receipt.get("repo") or receipt.get("path")
    if not model or executable not in command or str(model) not in command:
        return False
    if receipt.get("kind") == SERVE_RECEIPT_KIND and not receipt.get("jit") and "--adapter-path" in argv:
        index = argv.index("--adapter-path") + 1
        if index >= len(argv) or "--adapter-path" not in command or str(argv[index]) not in command:
            return False
    if receipt.get("jit"):
        if "--config" not in argv or str(argv[argv.index("--config") + 1]) not in command:
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


def _jit_config(receipt, root):
    path = root / "{}.jit-config.json".format(receipt["port"])
    if path.is_symlink():
        raise ServeError("invalid_receipt", "JIT configuration is a symbolic link.", "Inspect the owned receipt directory.")
    try:
        config = json.loads(path.read_text())
        if not isinstance(config, dict) or not isinstance(config.get("control_token"), str) or len(config["control_token"]) < 16:
            raise ValueError("Invalid control token.")
        return config
    except (OSError, ValueError) as error:
        raise ServeError("invalid_receipt", "JIT control configuration is unavailable.",
                         "Stop and review a fresh endpoint.") from error


def _jit_control(receipt, root, action, policy=None):
    config = _jit_config(receipt, root)
    request = urllib.request.Request("http://127.0.0.1:{}/_mlx/{}".format(receipt["port"], action),
        data=json.dumps(policy or {}).encode() if action in ("unload", "policy") else None,
        headers={"Authorization": "Bearer " + config["control_token"], "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=10 if action == "unload" else 2) as response:
            state = json.load(response)
            if not isinstance(state, dict) or state.get("model_state") not in ("loaded", "unloaded", "loading", "unloading", "failed"):
                raise ValueError("Invalid model residency status.")
            return state
    except urllib.error.HTTPError as error:
        try:
            detail = json.load(error)["error"]
        except (ValueError, KeyError, TypeError):
            detail = {"code": "control_failed", "message": "JIT control request failed."}
        raise ServeError(detail["code"], detail["message"], "Finish active requests, refresh status and retry.") from error
    except (OSError, ValueError) as error:
        raise ServeError("control_unavailable", "JIT control status is unavailable.",
                         "Refresh endpoint status before retrying.") from error


def _owned_jit_worker(receipt, root):
    if not receipt.get("jit"):
        return None
    path = root / "{}.jit-state.json".format(receipt["port"])
    if path.is_symlink():
        return None
    try:
        state = json.loads(path.read_text())
        pid, argv = state["worker_pid"], state["worker_argv"]
        if not isinstance(pid, int) or pid <= 0 or state["gateway_pid"] != receipt["pid"]:
            return None
        config = _jit_config(receipt, root)
        expected = {"repo": None, "path": config["local_path"], "argv": argv,
                    "port": int(argv[argv.index("--port") + 1])}
        if _pid_alive(pid) and os.getpgid(pid) == receipt["pid"] and _argv_matches(expected, _pid_command(pid)):
            return pid
    except (OSError, ValueError, TypeError, KeyError, ServeError):
        pass
    return None


def unload_serve(port, receipts_dir=None, expected_pid=None):
    return _configure_owned_jit(port, receipts_dir, expected_pid, "unload")


def configure_serve_memory(port, policy, receipts_dir=None, expected_pid=None):
    from .jit_serve import validate_memory_policy
    try:
        policy = validate_memory_policy(policy)
    except ValueError as error:
        raise ServeError("invalid_arguments", str(error), "Correct the memory policy.") from error
    return _configure_owned_jit(port, receipts_dir, expected_pid, "policy", policy)


def _configure_owned_jit(port, receipts_dir, expected_pid, action, policy=None):
    root = receipts_root(receipts_dir)
    receipt = _read_receipt(root / "{}.json".format(port))
    if not receipt or not receipt.get("jit"):
        raise ServeError("jit_required", "This endpoint does not support model unload.",
                         "Enable Load on request for this endpoint first.")
    if expected_pid is not None and receipt["pid"] != expected_pid:
        raise ServeError("pid_argv_mismatch", "Endpoint identity changed.", "Refresh serving status before unloading.")
    if not _pid_alive(receipt["pid"]) or not _argv_matches(receipt, _pid_command(receipt["pid"])):
        raise ServeError("pid_argv_mismatch", "Endpoint process no longer matches its receipt.",
                         "Refresh serving status before unloading.")
    state = _jit_control(receipt, root, action, policy)
    return dict(state, status="unloaded" if action == "unload" else "configured", port=port, pid=receipt["pid"])


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
    orphan = _owned_jit_worker(receipt, root) if not pid_alive(pid) else None
    if orphan:
        pid = orphan
    if not pid_alive(pid):
        receipt_path.unlink()
        return {"status": "already_stopped", "port": port, "pid": pid}
    command = pid_command(pid)
    if not orphan and not _argv_matches(receipt, command):
        raise ServeError(
            "pid_argv_mismatch",
            "The live pid {0} does not match the serve receipt; refusing to signal it.".format(pid),
            "Inspect the process yourself; the receipt is retained at {0}.".format(receipt_path),
        )
    if (receipt.get("jit") or receipt.get("runtime") == "strata") and terminate is _terminate_pid and os.getpgid(pid) == receipt["pid"]:
        terminate = lambda _pid, sig: os.killpg(receipt["pid"], sig)
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

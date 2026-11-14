"""The identity pass: every bound file and runtime still matches the serving plan.

serve and jit_serve import this module. The checks import those modules back,
so those imports stay inside the checks.
"""
from pathlib import Path


def _rejected(code, message, remediation):
    from .serve import ServeError

    raise ServeError(code, message, remediation)


def _require_auth_identity(plan):
    if plan.get("api_key_sha256") is None:
        return
    from .inference_auth import runtime_fingerprint, validate_digest

    try:
        validate_digest(plan["api_key_sha256"])
        if not plan.get("jit") or plan.get("auth_runtime_sha256") != runtime_fingerprint():
            raise ValueError("Authentication configuration changed after preview.")
    except ValueError as error:
        _rejected("preview_stale", str(error), "Review a fresh authenticated serving plan.")


def _require_task_identity(plan):
    if plan.get("task_identity") is None:
        return
    from .task_contracts import TaskError
    from .task_identity import check_identity
    from .task_serving import runtime_fingerprint

    try:
        check_identity(plan["task_identity"])
        if runtime_fingerprint(plan["task_identity"]["backend"]) != plan.get("task_runtime_fingerprint"):
            raise TaskError("runtime_changed", "The accepted inference adapter changed.")
    except TaskError as error:
        code = "runtime_changed" if error.code == "runtime_changed" else "preview_stale"
        _rejected(code, str(error), "Review a fresh task serving plan.")
    except OSError as error:
        _rejected("preview_stale", str(error), "Review a fresh task serving plan.")


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
    _rejected("training_receipt_changed", "The training receipt or its pinned artifacts changed.",
              "Review a fresh serving preview for the completed training receipt.")


def _argv_adapter(plan):
    argv = plan.get("worker_argv")
    if not isinstance(argv, list) or "--adapter-path" not in argv:
        return None
    index = argv.index("--adapter-path") + 1
    if index >= len(argv):
        return False
    return argv[index]


def _require_adapter_identity(plan):
    named = _argv_adapter(plan)
    path = plan.get("adapter_path")
    if path is None and named is None:
        return
    if (named is False or (named is not None and (
            not isinstance(path, str) or named != path or not plan.get("adapter_identity")))):
        _rejected("adapter_changed", "Adapter files changed after the serving preview.",
                  "Review a fresh serve plan before starting this adapter.")
    if not isinstance(path, str):
        _rejected("adapter_changed", "Adapter files changed after the serving preview.",
                  "Review a fresh serve plan before starting this adapter.")
    try:
        from .lora import adapter_identity

        local = Path(path).expanduser().absolute()
        identity = adapter_identity(local)
        if str(local) == path and identity == plan.get("adapter_identity"):
            return
    except (TypeError, ValueError, OSError):
        pass
    _rejected("adapter_changed", "Adapter files changed after the serving preview.",
              "Review a fresh serve plan before starting this adapter.")


def _require_prefill_identity(plan):
    if "prefill_kernel" not in plan:
        return
    from .prefill_runtime import can_wrap_runtime, identity

    configured = plan["prefill_kernel"]
    facts = {key: value for key, value in configured.items() if key not in ("adapter_enabled", "runtime_executable")}
    if facts != identity() or (configured.get("adapter_enabled") and not can_wrap_runtime(configured["runtime_executable"])):
        _rejected("prefill_kernel_changed", "Prompt kernel source or dependencies changed.",
                  "Review a fresh serving preview.")


def _require_decode_identity(plan):
    if "decode_profile" not in plan:
        return
    from .prompt_lookup_runtime import can_wrap_runtime, identity

    configured = plan["decode_profile"]
    facts = {key: value for key, value in configured.items() if key not in ("adapter_enabled", "runtime_executable")}
    if facts != identity() or (configured.get("adapter_enabled") and not can_wrap_runtime(configured["runtime_executable"])):
        _rejected("decode_profile_changed", "Code-edit runtime source or dependencies changed.",
                  "Review a fresh serving preview.")


def _require_vision_identity(plan):
    if "vision_profile" not in plan:
        return
    from .vision_runtime import identity

    configured = plan["vision_profile"]
    facts = {key: value for key, value in configured.items() if key != "adapter_enabled"}
    if facts != identity(configured["runtime_executable"]):
        _rejected("vision_profile_changed", "Vision runtime source or dependencies changed.",
                  "Review a fresh serving preview.")


def _require_splash_identity(plan):
    if plan.get("runtime") != "splash":
        return
    from .splash_runtime import SplashError, installed_runtime, read_draft, read_target

    try:
        current = (read_target(plan["splash_model_dir"])["fingerprint"],
                   read_draft(plan["splash_draft_model"])["fingerprint"] if plan.get("splash_draft_model") else None,
                   installed_runtime(plan["splash_launcher"])["fingerprint"])
    except (SplashError, KeyError) as error:
        _rejected("preview_stale", str(error), "Review a fresh Splash serve plan.")
    if current != (plan["splash_model_fingerprint"], plan["splash_draft_fingerprint"], plan["splash_runtime_fingerprint"]):
        _rejected("preview_stale", "The Splash target, draft or launcher changed after preview.",
                  "Review a fresh Splash serve plan.")


def _require_access_identity(plan):
    if "private_worker" not in plan and "access_key_file" not in plan:
        return
    from .private_serve import identity
    from .serving_access import read_access_key

    try:
        if "private_worker" in plan and plan.get("runtime") != "mlx_lm":
            raise ValueError("Private worker source changed.")
        if "access_key_file" in plan:
            read_access_key(plan["access_key_file"], plan["access_key_sha256"])
        if "private_worker" in plan and plan["private_worker"] != identity():
            raise ValueError("Private worker source changed.")
    except (OSError, ValueError):
        _rejected("preview_stale", "Protected serving inputs changed after preview.",
                  "Review a fresh serve plan.")


def _require_model_identity(plan):
    if "fingerprint" not in plan or "local_path" not in plan:
        return
    from .jit_serve import model_fingerprint

    try:
        current = model_fingerprint(plan["local_path"])
    except ValueError as error:
        _rejected("model_changed", str(error), "Review a fresh serve plan.")
    if current != plan["fingerprint"]:
        _rejected("model_changed", "Model files changed. Stop and review a fresh serve plan.",
                  "Review a fresh serve plan.")


def recheck_bound_identity(plan):
    """One pass over every identity bound into a reviewed serving plan."""
    for check in (
        _require_auth_identity,
        _require_task_identity,
        _require_training_binding,
        _require_adapter_identity,
        _require_prefill_identity,
        _require_decode_identity,
        _require_vision_identity,
        _require_splash_identity,
        _require_access_identity,
        _require_model_identity,
    ):
        check(plan)

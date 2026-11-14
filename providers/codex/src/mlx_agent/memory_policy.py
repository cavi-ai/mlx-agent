"""One memory policy: reviewed limits in, the residency outcome for one endpoint out.

Callers use this module. They do not translate field names or restate which
runtimes may carry a budget, a KV context, or an allocator cache. Splash and
Strata sizes are read from the serving plan; those modules still compute them.
"""
import json
import math
from pathlib import Path

from .allocator_cache import FIELDS as CACHE_FIELDS
from .allocator_cache import from_memory, requested
from .contextfit import CONTEXT_MAX, CONTEXT_MIN, extract_architecture, kv_cache_bytes
from .serving_plan import memory_basis

STRATA_CONTEXT = "Strata JIT admission already includes the KV cache at --strata-context-tokens."
TASK_CONTEXT = "Task endpoints do not use text KV context settings."

_DEFAULTS = {"idle_timeout_seconds": 0, "keep_loaded": False, "minimum_headroom_gb": None}
_EXTRA = CACHE_FIELDS | {
    "context_tokens", "unload_under_pressure", "model_memory_budget_gb",
    "release_after_request", "execution_wait_seconds",
}


class PolicyRejected(ValueError):
    """A policy the runtime cannot carry. code is the serving error code."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def validate(policy=None):
    """Return a normalized policy. None is the idle-off, keep-off default."""
    if policy is None:
        return dict(_DEFAULTS)
    if not isinstance(policy, dict) or set(policy) - set(_DEFAULTS) - _EXTRA:
        raise ValueError("Unknown memory policy fields.")
    result = dict(_DEFAULTS, **policy)
    cache = from_memory(result)
    if cache["cache_reclaim"] == "pressure" and result["minimum_headroom_gb"] is None:
        raise ValueError("Pressure cache reclaim requires a headroom reserve.")
    wait = result.get("execution_wait_seconds", 0)
    if type(wait) is not int or not 0 <= wait <= 3600:
        raise ValueError("Execution wait must be an integer between 0 and 3600 seconds.")
    release = result.pop("release_after_request", False)
    if type(release) is not bool or (release and result["keep_loaded"]):
        raise ValueError("Release after each request cannot be combined with keep loaded.")
    if "release_after_request" in policy:
        result["release_after_request"] = release
    context = result.pop("context_tokens", None)
    pressure = result.pop("unload_under_pressure", False)
    budget = result.pop("model_memory_budget_gb", None)
    if budget is not None:
        if type(budget) not in (int, float) or not math.isfinite(budget) or not 1 <= budget <= 1024:
            raise ValueError("Model memory budget must be a finite value from 1 to 1024 GB.")
        result["model_memory_budget_gb"] = budget
        if result["minimum_headroom_gb"] is None:
            result["minimum_headroom_gb"] = 2
    idle, keep, reserve = (result[key] for key in _DEFAULTS)
    if type(idle) is not int or not 0 <= idle <= 86400:
        raise ValueError("Idle timeout must be an integer between 0 and 86400 seconds.")
    if type(keep) is not bool:
        raise ValueError("Keep loaded must be a boolean.")
    if reserve is not None and (type(reserve) not in (int, float) or not math.isfinite(reserve) or not 0 <= reserve <= 1024):
        raise ValueError("Headroom reserve must be a finite value between 0 and 1024 GB.")
    if type(pressure) is not bool:
        raise ValueError("Unload under pressure must be a boolean.")
    if pressure:
        if reserve is None or keep:
            raise ValueError("Unload under pressure requires a headroom reserve and cannot be combined with keep loaded.")
        result["unload_under_pressure"] = True
    if context is not None:
        if type(context) is not int or not CONTEXT_MIN <= context <= CONTEXT_MAX:
            raise ValueError("Context tokens must be an integer between {} and {}.".format(CONTEXT_MIN, CONTEXT_MAX))
        if reserve is None:
            raise ValueError("Context tokens require a headroom reserve.")
        result["context_tokens"] = context
    return result


def from_cli(arguments):
    """Translate CLI names into a policy dict. Does not validate."""
    policy = {"idle_timeout_seconds": arguments.idle_timeout or 0, "keep_loaded": bool(arguments.keep_loaded),
              "minimum_headroom_gb": arguments.min_headroom_gb}
    for key in ("allocator_cache_limit_mib", "cache_reclaim"):
        if getattr(arguments, key, None) is not None:
            policy[key] = getattr(arguments, key)
    if getattr(arguments, "clear_allocator_cache_limit", False):
        policy["allocator_cache_limit_mib"] = None
    if getattr(arguments, "execution_wait_seconds", None) is not None:
        policy["execution_wait_seconds"] = arguments.execution_wait_seconds
    if getattr(arguments, "release_after_request", None) is not None:
        policy["release_after_request"] = arguments.release_after_request
    if arguments.memory_budget_gb is not None:
        policy["model_memory_budget_gb"] = arguments.memory_budget_gb
    if getattr(arguments, "clear_memory_budget", False):
        if arguments.memory_budget_gb is not None:
            raise ValueError("Choose --memory-budget-gb or --clear-memory-budget.")
        policy["model_memory_budget_gb"] = None
    if arguments.context_tokens is not None:
        policy["context_tokens"] = arguments.context_tokens
    if arguments.unload_under_pressure is not None:
        policy["unload_under_pressure"] = arguments.unload_under_pressure
    return policy


def cli_supplied(arguments):
    """True when serve start should attach a policy. Same fields from_cli reads."""
    if (getattr(arguments, "release_after_request", None) is not None
            or getattr(arguments, "idle_timeout", None) is not None
            or arguments.keep_loaded
            or getattr(arguments, "min_headroom_gb", None) is not None
            or getattr(arguments, "context_tokens", None) is not None
            or arguments.unload_under_pressure
            or getattr(arguments, "memory_budget_gb", None) is not None
            or getattr(arguments, "execution_wait_seconds", None) is not None
            or getattr(arguments, "allocator_cache_limit_mib", None) is not None
            or getattr(arguments, "cache_reclaim", None) is not None
            or getattr(arguments, "clear_allocator_cache_limit", False)
            or getattr(arguments, "clear_memory_budget", False)):
        return True
    return False


def _residency_chosen(arguments):
    return (getattr(arguments, "idle_timeout", None) is not None
            or bool(arguments.keep_loaded)
            or getattr(arguments, "min_headroom_gb", None) is not None
            or getattr(arguments, "release_after_request", None) is not None
            or bool(arguments.unload_under_pressure)
            or getattr(arguments, "context_tokens", None) is not None
            or getattr(arguments, "memory_budget_gb", None) is not None)


def for_start(arguments, runtime):
    """Policy for serve start, or None when the caller set no memory flag.

    A task endpoint that only sets execution wait or allocator cache keeps the
    task default of releasing weights after each job.
    """
    if not cli_supplied(arguments):
        return None
    policy = from_cli(arguments)
    if runtime == "tasks" and not _residency_chosen(arguments) and "release_after_request" not in policy:
        policy["release_after_request"] = True
    return policy


def policy_from_profile(settings):
    """Profile names to policy names, for validation. The profile stores CLI names."""
    policy = {"minimum_headroom_gb": settings["min_headroom_gb"], "idle_timeout_seconds": settings["idle_timeout"],
              "keep_loaded": settings["keep_loaded"], "unload_under_pressure": settings["unload_under_pressure"]}
    if "memory_budget_gb" in settings:
        policy["model_memory_budget_gb"] = settings["memory_budget_gb"]
    for key in ("execution_wait_seconds", "release_after_request", *CACHE_FIELDS):
        if key in settings:
            policy[key] = settings[key]
    return policy


def _reject_cache(policy, runtime):
    if runtime != "tasks" and requested(policy):
        raise PolicyRejected("cache_control_unavailable",
                             "Allocator cache controls require a resident task backend.",
                             "Select a supported task model; inspect allocator_cache in endpoint status.")


def enforce_launch(policy, runtime, jit):
    """Gates for a non-task start plan. Task plans do not call this."""
    _reject_cache(policy, runtime)
    if not jit:
        raise PolicyRejected("jit_required", "Memory policy requires a JIT endpoint.", "Use --jit.")
    if runtime != "strata" and "model_memory_budget_gb" in policy:
        raise PolicyRejected("invalid_arguments", "Model memory budgets currently require Strata.",
                             "Select a prepared Strata profile.")
    if runtime == "strata" and "context_tokens" in policy:
        raise PolicyRejected("invalid_arguments", STRATA_CONTEXT, "Remove --context-tokens.")


def reject_task_context(changes):
    if changes.get("context_tokens") is not None:
        raise PolicyRejected("invalid_policy", TASK_CONTEXT, "Review the memory policy.")


def enforce_configure(policy, runtime, changes):
    """Gates when a live endpoint replaces its policy."""
    if runtime == "tasks":
        reject_task_context(changes)
    if runtime == "strata" and "context_tokens" in policy:
        raise PolicyRejected("invalid_policy", STRATA_CONTEXT, "Remove --context-tokens.")
    if runtime not in ("strata", "tasks") and "model_memory_budget_gb" in policy:
        raise PolicyRejected("invalid_policy", "Model memory budgets require Strata or a task endpoint.",
                             "Select a prepared Strata profile or a task model.")
    _reject_cache(policy, runtime)


def reject_cache_outside_tasks(policy, runtime):
    """Worker construction. Same rule as configure, raised as ValueError for the existing caller."""
    try:
        _reject_cache(policy, runtime)
    except PolicyRejected as error:
        raise ValueError(str(error)) from error


def execution_wait_seconds(policy):
    """How long one endpoint may wait for execution."""
    return policy.get("execution_wait_seconds", 0)


def releases_after_request(policy):
    """Whether weights are released when the endpoint has no active request."""
    return bool(policy.get("release_after_request", False))


_UNMEASURED = object()


def release_reason(policy, *, busy, idle_seconds, budget_reload, available=_UNMEASURED):
    """Why resident weights should be released, or None to keep them.

    Omit ``available`` before a headroom probe. ``probe`` means the caller
    should measure headroom and ask again. A measured ``None`` is unknown
    headroom and does not release weights.
    """
    if busy:
        return None
    if budget_reload:
        return "memory_budget"
    if policy["keep_loaded"] or idle_seconds is None:
        return None
    if releases_after_request(policy) or (
            policy["idle_timeout_seconds"] and idle_seconds >= policy["idle_timeout_seconds"]):
        return "idle"
    if not policy.get("unload_under_pressure"):
        return None
    if available is _UNMEASURED:
        return "probe"
    reserve = policy["minimum_headroom_gb"]
    if available is None or reserve is None or type(available) is not int or available < 0:
        return None
    if available < math.ceil(reserve * 1e9):
        return "memory_pressure"
    return None


def _kv_cache_bytes(local_path, context_tokens):
    """KV cache bytes at the context from the local config.json, or None when its architecture is unknown."""
    try:
        location = Path(local_path) / "config.json"
        if location.stat().st_size > 1024 * 1024:
            return None
        config = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    architecture = extract_architecture(config) or extract_architecture(config.get("text_config"))
    return kv_cache_bytes(architecture, context_tokens)


def _weight_bytes(local_path):
    return sum(item.stat().st_size for item in Path(local_path).rglob("*")
               if item.is_file() and item.suffix in (".safetensors", ".npz", ".bin", ".gguf"))


def reservation(policy, config):
    """Bytes to reserve for one endpoint, from the policy and the serving plan's sizes.

    Returns None when the estimate is unavailable. Splash weight bytes and Strata
    estimate, allocation, and growth are the values already stored on the plan.
    """
    context = policy.get("context_tokens")
    basis = memory_basis(config)
    runtime_bytes = 1_500_000_000
    weights = 0
    kv = 0
    if basis.kind == "splash":
        weights = basis.weights
    elif basis.kind != "strata":
        weights = _weight_bytes(config["local_path"])
    if basis.kind != "strata" and weights and config.get("adapter_path") is not None:
        weights += config["adapter_identity"]["adapters.safetensors"]["bytes"]
    if basis.kind != "strata":
        kv = 0 if context is None else _kv_cache_bytes(config["local_path"], context)
    dynamic_bytes = runtime_bytes + (kv or 0)
    size = math.ceil(weights * 1.10 + kv + 1.5e9) if weights and kv is not None else None
    if basis.kind == "splash" and basis.cap_gb is not None:
        size = math.ceil(basis.cap_gb * 1e9)
    if basis.kind == "strata":
        runtime_bytes = 5_000_000_000
        if policy.get("model_memory_budget_gb") is not None:
            size = basis.allocation_bytes
        else:
            size = basis.estimate_bytes
        dynamic_bytes = basis.growth_bytes
        if size is None or dynamic_bytes is None:
            size = None
    if size is None:
        return None
    size = math.ceil(size)
    workspace = min(size, runtime_bytes)
    growth = min(size - workspace, max(0, math.ceil(dynamic_bytes) - workspace))
    return {"size": size, "workspace_bytes": workspace, "growth_bytes": growth,
            "reserve_bytes": size - workspace - growth}

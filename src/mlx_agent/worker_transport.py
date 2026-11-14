"""How a reviewed serving plan reaches its model worker.

The interface is `choose`: one of inherited Unix, loopback key, resident task,
or an open worker. `gateway_config` is the one config both the supervised
start and the foreground protected endpoint build.
"""
from pathlib import Path
from typing import Protocol


class WorkerTransport(Protocol):
    name: str

    def mark(self, plan) -> None:
        """Record this transport on the plan. Must not copy the endpoint key."""


class _InheritedUnix:
    name = "inherited-unix"

    def mark(self, plan):
        from .private_serve import identity
        plan["private_worker"] = identity()


class _LoopbackKey:
    name = "loopback-key"

    def mark(self, plan):
        """The endpoint key stays off the worker. The worker gets its own digest at launch."""
        return plan


class _ResidentTask:
    name = "resident"

    def mark(self, plan):
        """A task worker has no private HTTP port. Gateway auth is the whole transport."""
        return plan


class _OpenWorker:
    name = "open"

    def mark(self, plan):
        return plan


def choose(plan):
    """Return the transport for a plan that already carries its endpoint digest, if any."""
    if plan.get("api_key_sha256") is None:
        return _OpenWorker()
    if plan.get("runtime") == "tasks":
        return _ResidentTask()
    if (plan.get("runtime") == "mlx_lm" and Path(plan["worker_argv"][0]).name == "mlx_lm.server"
            and not any(plan.get(key) for key in ("prefill_kernel", "decode_profile", "vision_profile"))):
        return _InheritedUnix()
    return _LoopbackKey()


def gateway_config(plan, *, control_token, state_path, log_path=None, config_path=None, task_state_dir=None):
    """Config for the JIT gateway. Callers supply paths; the key set lives here."""
    config = {key: plan[key] for key in ("local_path", "fingerprint", "worker_argv", "max_tokens")}
    config["runtime"] = plan["runtime"]
    if "request_defaults" in plan:
        config["request_defaults"] = plan["request_defaults"]
    config.update({key: plan[key] for key in (
        "private_worker", "access_key_file", "access_key_sha256", "api_key_sha256", "auth_runtime_sha256",
    ) if key in plan})
    if plan["runtime"] == "tasks":
        config.update(task_identity=plan["task_identity"], task_kinds=plan["task_kinds"], port=plan["port"],
                      task_runtime_fingerprint=plan["task_runtime_fingerprint"], task_state_dir=task_state_dir)
    if plan.get("adapter_path") is not None:
        config.update(adapter_path=plan["adapter_path"], adapter_identity=plan["adapter_identity"])
    if plan.get("training_binding") is not None:
        config["training_binding"] = plan["training_binding"]
    if "readiness_deadline" in plan:
        config["worker_readiness_seconds"] = plan["readiness_deadline"]
    if plan["runtime"] == "strata":
        config["strata_expert_memory_gb"] = plan.get("strata_expert_memory_gb")
        config["strata_context_tokens"] = plan.get("strata_context_tokens", 131072)
        config["strata_speculation"] = plan.get("strata_speculation", False)
        config["strata_decoding"] = plan["strata_decoding"]
        if "strata_memory_allocation" in plan:
            config["strata_memory_allocation"] = plan["strata_memory_allocation"]
    if plan["runtime"] == "splash":
        config.update({key: plan[key] for key in ("splash_served_model", "splash_weight_bytes", "splash_max_memory_gb")})
    for key in (
        "jit", "prefill_kernel", "decode_profile", "vision_profile",
        "splash_model_dir", "splash_model_fingerprint", "splash_draft_model", "splash_draft_fingerprint",
        "splash_launcher", "splash_runtime_fingerprint",
        "strata_memory_estimate_bytes", "strata_memory_growth_bytes",
    ):
        if key in plan:
            config[key] = plan[key]
    config.update(model=plan["repo"] or plan["path"], control_token=control_token,
                  memory_policy=plan.get("memory_policy"), state_path=state_path)
    if config_path is not None:
        config["config_path"] = str(config_path)
    if log_path is not None:
        config["log_path"] = str(log_path)
    return config

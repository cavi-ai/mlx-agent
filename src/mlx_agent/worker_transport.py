"""How a reviewed serving plan reaches its model worker.

The interface is `choose`: one of inherited Unix, loopback key, resident task,
or an open worker. `gateway_config` is that plan's worker config, shared by
the supervised start and the foreground protected endpoint.
"""
from pathlib import Path
from typing import Protocol

from .serving_plan import worker_config


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
    """Worker config for a reviewed plan. The key set lives on the serving plan."""
    return worker_config(plan, control_token=control_token, state_path=state_path, log_path=log_path,
                         config_path=config_path, task_state_dir=task_state_dir)

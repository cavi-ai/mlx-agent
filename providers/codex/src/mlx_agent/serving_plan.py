"""The reviewed serving plan: one description, one preview hash, and the copies callers used to repeat.

Splash and Strata each return one contribution. Task endpoints return earlier
through task_serving.plan_start. Chat runtimes add nothing here.
"""
import hashlib
import json


class Contribution:
    """What one runtime adds to a serving plan."""

    def __init__(self, argv, values, fields, worker_entry=None):
        self.argv = list(argv)
        self.values = dict(values)
        self.fields = dict(fields)
        self.worker_entry = None if worker_entry is None else list(worker_entry)

    def apply_values(self, values):
        values.update(self.values)

    def extend_argv(self, argv):
        argv.extend(self.argv)

    def apply_fields(self, plan):
        plan.update(self.fields)

    def retarget_worker(self, worker):
        """Replace the worker program when this runtime does not use the recipe executable."""
        if self.worker_entry is None:
            return False
        worker[:1] = self.worker_entry
        return True


class MemoryBasis:
    """Sizes on a reviewed plan that reservation consumes without naming a runtime's fields."""

    def __init__(self, kind, weights=None, cap_gb=None, allocation_bytes=None, estimate_bytes=None, growth_bytes=None):
        self.kind = kind
        self.weights = weights
        self.cap_gb = cap_gb
        self.allocation_bytes = allocation_bytes
        self.estimate_bytes = estimate_bytes
        self.growth_bytes = growth_bytes


def preview_hash(plan):
    canonical = json.dumps(
        {key: plan[key] for key in sorted(plan) if key != "preview_hash"},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def memory_basis(plan):
    """Runtime sizes stored on the plan. Missing Splash weights still raise KeyError."""
    runtime = plan.get("runtime")
    if runtime == "splash":
        return MemoryBasis("splash", weights=plan["splash_weight_bytes"], cap_gb=plan.get("splash_max_memory_gb"))
    if runtime == "strata":
        allocation = plan.get("strata_memory_allocation") or {}
        return MemoryBasis(
            "strata",
            allocation_bytes=allocation.get("estimated_resident_bytes"),
            estimate_bytes=plan.get("strata_memory_estimate_bytes"),
            growth_bytes=plan.get("strata_memory_growth_bytes"),
        )
    return MemoryBasis("local")


def worker_config(plan, *, control_token, state_path, log_path=None, config_path=None, task_state_dir=None):
    """Config for the supervised worker. Callers supply paths; the key set lives here."""
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


def receipt_fields(plan):
    """Plan fields copied onto a serve receipt. The caller adds pid, log, and time."""
    fields = {}
    if plan.get("jit"):
        fields["jit"] = True
    if plan.get("api_key_sha256") is not None:
        fields["api_key_sha256"] = plan["api_key_sha256"]
    if "private_worker" in plan:
        fields["protection"] = {
            "access": "api_key",
            "worker_transport": plan["private_worker"]["protocol"],
            "worker_source_sha256": plan["private_worker"]["source_sha256"],
        }
    if plan.get("task_kinds"):
        fields["task_kinds"] = plan["task_kinds"]
    if plan.get("adapter_path") is not None:
        fields.update(adapter_path=plan["adapter_path"], adapter_identity=plan["adapter_identity"])
    if plan.get("training_binding") is not None:
        fields["training_binding"] = plan["training_binding"]
    if "prefill_kernel" in plan:
        fields["prefill_kernel"] = plan["prefill_kernel"]
    if "decode_profile" in plan:
        fields["decode_profile"] = plan["decode_profile"]
    if "vision_profile" in plan:
        fields["vision_profile"] = plan["vision_profile"]
    if "strata_decoding" in plan:
        fields["strata_decoding"] = plan["strata_decoding"]
    if "strata_memory_allocation" in plan:
        fields["strata_memory_allocation"] = plan["strata_memory_allocation"]
    if plan["runtime"] == "splash":
        fields.update(splash_served_model=plan["splash_served_model"], splash_version=plan["splash_version"])
    return fields

"""Scoped, receipt-owned installer built on the Transaction safety boundary."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .providers import detect_providers
from .transactions import (
    Receipt, Transaction, _assert_safe_directory, _assert_safe_target,
    _atomic_in_directory, _read_regular, _read_target, _walk_directory, rollback,
)
from .wiring import redact_secrets


class InstallerConflictError(ValueError):
    """A requested operation would touch an unowned or modified artifact."""


class _ArtifactAdapter:
    version = "installer-1.0"

    def validate(self, content):
        if not isinstance(content, str):
            raise TypeError("installer artifacts must be UTF-8 text")
        if redact_secrets(content) != content:
            raise ValueError("installer artifacts must not contain persisted secrets")
        return True


@dataclass(frozen=True)
class _PlannedTransaction:
    provider_id: str
    changes: tuple
    preview_hash: str
    destinations: tuple
    transaction_id: str
    expected_receipt_path: str


@dataclass
class InstallerReceipt:
    status: str
    receipts: list = field(default_factory=list)
    receipt_path: str = ""
    targets: list = field(default_factory=list)
    batch_path: str = ""

    def to_dict(self):
        return {
            "status": self.status,
            "receipt_path": self.receipt_path or None,
            "batch_path": self.batch_path or None,
            "targets": list(self.targets),
            "receipts": [_receipt_data(item) for item in self.receipts],
        }


@dataclass
class InstallPlan:
    action: str
    provider_ids: tuple
    scope: str
    project_root: Path
    preview: dict
    transactions: list = field(default_factory=list)
    rollback_receipts: list = field(default_factory=list)
    uninstall_hashes: dict = field(default_factory=dict)
    restore_changes: tuple = field(default_factory=tuple)
    binding: dict = field(default_factory=dict)
    noop: bool = False

    def to_dict(self):
        return {
            "action": self.action,
            "providers": list(self.provider_ids),
            "scope": self.scope,
            "project_root": str(self.project_root),
            "preview": self.preview,
            "noop": self.noop,
        }


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _receipt_data(receipt):
    value = receipt.to_dict()
    value["receipt_path"] = receipt.receipt_path
    return value


def _digest(value):
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


class Installer:
    """Plan first; execute only the exact reviewed transaction previews."""

    def __init__(self, registry, project_root=None, executable_lookup=None, env=None,
                 transaction_factory=None, rollback_func=None):
        self.registry = registry
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
        self.executable_lookup = executable_lookup
        self.env = env
        self.transaction_factory = transaction_factory or Transaction
        self.rollback_func = rollback_func or rollback

    def detected(self):
        return detect_providers(
            self.registry.definitions().values(), self.env, executable_lookup=self.executable_lookup
        )

    def plan(self, action, providers, scope="user", project_root=None):
        if action not in {"install", "update", "uninstall", "doctor"}:
            raise ValueError("unsupported installer action: {0}".format(action))
        if scope not in {"user", "project"}:
            raise ValueError("scope must be 'user' or 'project'")
        project = Path(project_root or self.project_root).resolve()
        if scope == "project" and not project.is_dir():
            raise ValueError("project root must be an existing directory")
        definitions = self.registry.definitions()
        selected = tuple(providers or ())
        if not selected:
            raise ValueError("provider selection is required; inspect detected providers first")
        if len(selected) != len(set(selected)) or any(item not in definitions for item in selected):
            raise ValueError("providers must be unique supported provider IDs")
        if action == "doctor":
            return self._doctor_plan(selected, definitions, scope, project)
        if action == "uninstall":
            return self._uninstall_plan(selected, definitions, scope, project)
        return self._install_plan(action, selected, definitions, scope, project)

    def execute(self, plan, confirmed=False):
        if plan.action == "doctor":
            return self._doctor(plan)
        if plan.noop:
            return InstallerReceipt("noop")
        if confirmed is not True and confirmed != plan.preview["preview_hash"]:
            raise PermissionError("explicit confirmation for this installer preview is required")
        self._verify_plan_binding(plan)
        batch_path = self._create_batch(plan)
        if plan.action == "uninstall":
            return self._execute_uninstall(plan, batch_path)
        return self._execute_install(plan, batch_path)

    def _execute_install(self, plan, batch_path):
        successful, all_receipts = [], []
        try:
            prepared = [(operation, *self._repreview(operation, plan.scope, plan.project_root)) for operation in plan.transactions]
        except PermissionError as error:
            self._batch_update(batch_path, "rolled_back", successful, error=str(error))
            raise
        try:
            for operation, transaction, preview in prepared:
                self._batch_update(batch_path, "pending", all_receipts, active=self._active_operation(operation))
                receipt = transaction.apply(operation.preview_hash)
                all_receipts.append(receipt)
                self._batch_update(batch_path, "pending", all_receipts)
                if receipt.status != "applied":
                    raise InstallerConflictError("provider transaction did not apply: {0}".format(receipt.status))
                successful.append(receipt)
            self._batch_update(batch_path, "complete", successful)
            return InstallerReceipt("applied", all_receipts, successful[-1].receipt_path, [target for item in successful for target in item.targets], str(batch_path))
        except Exception as error:
            recovered, unresolved = self._compensate_install(all_receipts)
            child_requires_recovery = any(item.status in {"recovery_required", "rollback_failed"} for item in all_receipts)
            status = "rolled_back" if recovered and not child_requires_recovery else "recovery_required"
            remediation = unresolved or ([{"paths": list(all_receipts[-1].targets), "message": "Child receipt requires manual recovery: {0}".format(all_receipts[-1].receipt_path)}] if child_requires_recovery and all_receipts else [])
            self._batch_update(batch_path, status, all_receipts, error=str(error), remediation=remediation)
            return InstallerReceipt(status, all_receipts, all_receipts[-1].receipt_path if all_receipts else "", [target for item in all_receipts for target in item.targets], str(batch_path))

    def _execute_uninstall(self, plan, batch_path):
        completed = []
        try:
            for receipt in plan.rollback_receipts:
                self._batch_update(batch_path, "pending", completed, active={"provider": "uninstall", "transaction_id": receipt.transaction_id, "expected_receipt_path": receipt.receipt_path})
                result = self.rollback_func(receipt.receipt_path, expected_after_hashes=receipt.after_hashes)
                if result.status != "rolled_back":
                    raise InstallerConflictError("uninstall rollback did not complete: {0}".format(result.status))
                completed.append(result)
            self._batch_update(batch_path, "complete", completed)
            return InstallerReceipt("rolled_back", completed, completed[-1].receipt_path if completed else "", [target for item in completed for target in item.targets], str(batch_path))
        except Exception as error:
            if not completed:
                self._batch_update(batch_path, "rolled_back", completed, error=str(error))
                raise InstallerConflictError("uninstall aborted before mutation: {0}".format(error))
            recovered, unresolved = self._compensate_uninstall(plan.restore_changes, completed, plan.scope, plan.project_root)
            status = "rolled_back" if recovered else "recovery_required"
            self._batch_update(batch_path, status, completed, error=str(error), remediation=unresolved)
            if not recovered:
                raise InstallerConflictError("uninstall recovery is required: {0}".format(error))
            return InstallerReceipt(status, completed, completed[-1].receipt_path if completed else "", [target for item in completed for target in item.targets], str(batch_path))

    def _install_plan(self, action, selected, definitions, scope, project):
        transactions, summary = [], []
        for provider_id in selected:
            definition = definitions[provider_id]
            root = definition.destination(scope, project)
            changes = []
            for artifact in definition.artifacts:
                target = root / artifact.destination
                desired = artifact.source.read_text(encoding="utf-8")
                current = target.read_bytes() if target.is_file() else None
                if current is not None:
                    history = self._artifact_history(definition, target, scope, project)
                    if not history or _sha256(current) != history[-1].after_hashes[str(target)]:
                        raise InstallerConflictError("refusing to overwrite unowned or modified artifact: {0}".format(target))
                    if current == desired.encode("utf-8"):
                        continue
                changes.append({"path": str(target), "content": desired, "adapter": _ArtifactAdapter()})
            if changes:
                transaction = self._new_transaction(scope, project)
                preview = transaction.preview(changes)
                transaction_id = self._operation_transaction_id(action, scope, project, provider_id, preview["preview_hash"])
                expected = self._receipts_dir(scope, project) / transaction_id / "receipt.json"
                transactions.append(_PlannedTransaction(provider_id, tuple(changes), preview["preview_hash"], tuple(str(item["path"]) for item in changes), transaction_id, str(expected)))
                summary.append({"provider": provider_id, "changes": preview["changes"], "diff": preview["diff"]})
        return self._make_plan(action, selected, scope, project, transactions=transactions, summary=summary, definitions=definitions)

    def _uninstall_plan(self, selected, definitions, scope, project):
        steps, summary, uninstall_hashes, restore_changes = [], [], {}, {}
        seen = set()
        for provider_id in selected:
            definition = definitions[provider_id]
            provider_steps = []
            for artifact in definition.artifacts:
                target = definition.destination(scope, project) / artifact.destination
                history = self._artifact_history(definition, target, scope, project)
                if not history:
                    continue
                oldest, latest = history[0], history[-1]
                target_name = str(target)
                if oldest.backup_paths.get(target_name) is not None:
                    raise InstallerConflictError("artifact was not receipt-owned at installation: {0}".format(target))
                location = _assert_safe_target(target)
                content = _read_regular(location)
                if _sha256(content) != latest.after_hashes[target_name]:
                    raise InstallerConflictError("refusing to remove user-modified artifact: {0}".format(target))
                uninstall_hashes[target_name] = latest.after_hashes[target_name]
                restore_changes.setdefault(provider_id, []).append({"path": target_name, "content": content.decode("utf-8"), "adapter": _ArtifactAdapter()})
                provider_steps.extend(history)
            for receipt in sorted({item.receipt_path: item for item in provider_steps}.values(), key=lambda item: item.timestamp, reverse=True):
                if receipt.receipt_path not in seen:
                    seen.add(receipt.receipt_path)
                    steps.append(receipt)
            summary.append({"provider": provider_id, "changes": len(provider_steps), "diff": "remove receipt-owned artifacts"})
        return self._make_plan("uninstall", selected, scope, project, rollback_receipts=steps,
                               uninstall_hashes=uninstall_hashes, restore_changes=tuple((provider_id, tuple(changes)) for provider_id, changes in restore_changes.items()),
                               summary=summary, definitions=definitions)

    def _doctor_plan(self, selected, definitions, scope, project):
        return self._make_plan("doctor", selected, scope, project, definitions=definitions)

    def _doctor(self, plan):
        problems, checked = [], []
        definitions = self.registry.definitions()
        for provider_id in plan.provider_ids:
            definition = definitions[provider_id]
            root = definition.destination(plan.scope, plan.project_root)
            for artifact in definition.artifacts:
                target = root / artifact.destination
                checked.append(str(target))
                try:
                    safe_root = _assert_safe_directory(root)
                    location = _assert_safe_target(target)
                    location.relative_to(safe_root)
                    history = self._artifact_history(definition, target, plan.scope, plan.project_root)
                    if not history or _sha256(_read_regular(location)) != history[-1].after_hashes[str(target)]:
                        raise ValueError("artifact hash is not receipt-owned")
                except (OSError, ValueError):
                    problems.append({"provider": provider_id, "code": "artifact_invalid", "path": str(target)})
        problems.extend(self._batch_problems(plan.scope, plan.project_root))
        return {"healthy": not problems, "problems": problems, "checked": checked,
                "cooperative_concurrency": "Transaction advisory locks protect cooperative writers."}

    def _batch_problems(self, scope, project):
        batches = self._receipts_dir(scope, project) / "batches"
        try:
            safe_batches = _assert_safe_directory(batches)
        except ValueError:
            return []
        problems = []
        for path in safe_batches.glob("*/batch.json"):
            try:
                batch = self._read_batch(path)
            except (OSError, ValueError, json.JSONDecodeError):
                problems.append({"code": "batch_recovery_required", "path": str(path)})
                continue
            if batch["status"] in {"pending", "recovery_required"}:
                active = batch.get("active")
                if isinstance(active, dict) and active.get("expected_receipt_path"):
                    expected = active["expected_receipt_path"]
                    try:
                        child = _assert_safe_target(expected)
                        child_value = json.loads(_read_regular(child).decode("utf-8"))
                        if child_value.get("transaction_id") != active.get("transaction_id"):
                            raise ValueError("active child receipt does not match batch transaction")
                    except (OSError, ValueError, json.JSONDecodeError):
                        expected = "missing or untrusted active receipt: {0}".format(expected)
                    problems.append({"code": "batch_recovery_required", "path": str(path), "active_receipt": expected})
                else:
                    problems.append({"code": "batch_recovery_required", "path": str(path)})
        return problems

    def _make_plan(self, action, selected, scope, project, transactions=(), rollback_receipts=(),
                   uninstall_hashes=None, restore_changes=(), summary=(), definitions=None):
        definitions = definitions or self.registry.definitions()
        destinations = [{"provider": provider_id, "targets": [str(definitions[provider_id].destination(scope, project) / artifact.destination) for artifact in definitions[provider_id].artifacts]} for provider_id in selected]
        binding = {
            "action": action, "providers": list(selected), "scope": scope, "project": str(project),
            "destinations": destinations,
            "inner_previews": [{"provider": item.provider_id, "hash": item.preview_hash, "targets": list(item.destinations), "transaction_id": item.transaction_id, "expected_receipt_path": item.expected_receipt_path} for item in transactions],
            "rollback_chain": [self._receipt_identity(item) for item in rollback_receipts],
        }
        preview = {"changes": sum(item["changes"] for item in summary), "providers": list(selected),
                   "diff": "\n".join(item["diff"] for item in summary), "preview_hash": _digest(binding),
                   "requires_confirmation": action != "doctor"}
        return InstallPlan(action, selected, scope, project, preview, list(transactions), list(rollback_receipts),
                           dict(uninstall_hashes or {}), tuple(restore_changes), binding,
                           noop=not transactions and not rollback_receipts)

    def _verify_plan_binding(self, plan):
        definitions = self.registry.definitions()
        binding = {
            "action": plan.action, "providers": list(plan.provider_ids), "scope": plan.scope,
            "project": str(plan.project_root),
            "destinations": [{"provider": provider_id, "targets": [str(definitions[provider_id].destination(plan.scope, plan.project_root) / artifact.destination) for artifact in definitions[provider_id].artifacts]} for provider_id in plan.provider_ids],
            "inner_previews": [{"provider": item.provider_id, "hash": item.preview_hash, "targets": list(item.destinations), "transaction_id": item.transaction_id, "expected_receipt_path": item.expected_receipt_path} for item in plan.transactions],
            "rollback_chain": [self._receipt_identity(self._load_receipt(item.receipt_path, plan.scope, plan.project_root)) for item in plan.rollback_receipts],
        }
        if binding != plan.binding or _digest(binding) != plan.preview["preview_hash"]:
            raise PermissionError("installer preview is stale; regenerate and review it before confirmation")

    def _repreview(self, operation, scope, project):
        transaction = self._new_transaction(scope, project, operation.transaction_id)
        preview = transaction.preview(list(operation.changes))
        if preview["preview_hash"] != operation.preview_hash:
            raise PermissionError("installer preview is stale; an inner preview changed")
        return transaction, preview

    def _new_transaction(self, scope, project, transaction_id=None):
        return self.transaction_factory(receipts_dir=self._receipts_dir(scope, project), create_target_parents=True, transaction_id=transaction_id)

    @staticmethod
    def _operation_transaction_id(action, scope, project, provider_id, preview_hash):
        key = "mlx-agent:{0}:{1}:{2}:{3}:{4}".format(action, scope, project, provider_id, preview_hash)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    def _artifact_history(self, definition, target, scope, project):
        target_name = str(target)
        allowed = {str(definition.destination(scope, project) / artifact.destination) for artifact in definition.artifacts}
        return [item for item in self._all_receipts(scope, project) if target_name in item.targets and set(item.targets).issubset(allowed) and item.status == "applied"]

    def _all_receipts(self, scope, project):
        receipts_dir = self._receipts_dir(scope, project)
        try:
            safe_dir = _assert_safe_directory(receipts_dir)
        except ValueError:
            return []
        result = []
        for candidate in safe_dir.glob("*/receipt.json"):
            try:
                result.append(self._load_receipt(candidate, scope, project))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return sorted(result, key=lambda item: item.timestamp)

    def _load_receipt(self, path, scope, project):
        receipts_dir = _assert_safe_directory(self._receipts_dir(scope, project))
        location = _assert_safe_target(path)
        if location.parent.parent != receipts_dir:
            raise ValueError("installer receipt escapes its scoped receipt directory")
        return Receipt.from_dict(json.loads(_read_regular(location).decode("utf-8")), str(location))

    @staticmethod
    def _receipt_identity(receipt):
        return {"transaction_id": receipt.transaction_id, "receipt_path": receipt.receipt_path,
                "targets": list(receipt.targets), "before_hashes": receipt.before_hashes,
                "after_hashes": receipt.after_hashes, "backup_paths": receipt.backup_paths}

    def _compensate_install(self, receipts):
        unresolved = []
        for receipt in reversed(receipts):
            try:
                expected = {"expected_after_hashes": receipt.after_hashes} if receipt.status == "applied" else {}
                result = self.rollback_func(receipt.receipt_path, **expected)
                if result.status != "rolled_back":
                    unresolved.append({"paths": list(receipt.targets), "message": "Could not recover child receipt: {0}".format(receipt.receipt_path)})
            except Exception as error:
                unresolved.append({"paths": list(receipt.targets), "message": redact_secrets(str(error))})
        return not unresolved, unresolved

    def _compensate_uninstall(self, changes, completed, scope, project):
        if not changes:
            return True, []
        expected = {}
        for receipt in completed:
            for target in receipt.targets:
                expected[target] = (receipt.before_hashes[target], receipt.backup_paths[target] is not None, receipt.target_modes[target])
        unresolved = []
        try:
            affected = {target for receipt in completed for target in receipt.targets}
            for _provider_id, provider_changes in changes:
                provider_changes = [change for change in provider_changes if change["path"] in affected]
                if not provider_changes:
                    continue
                for change in provider_changes:
                    content, exists, mode = _read_target(change["path"])
                    expected_hash, expected_exists, expected_mode = expected[change["path"]]
                    if _sha256(content) != expected_hash or exists != expected_exists or mode != expected_mode:
                        unresolved.append({"paths": [change["path"]], "message": "Target changed after rollback; compensation will not overwrite it."})
                if unresolved:
                    continue
                transaction = self._new_transaction(scope, project)
                preview = transaction.preview(provider_changes)
                if transaction.apply(preview["preview_hash"]).status != "applied":
                    unresolved.append({"paths": [change["path"] for change in provider_changes], "message": "Compensation transaction did not apply."})
            return not unresolved, unresolved
        except Exception as error:
            return False, [{"paths": sorted(expected), "message": redact_secrets(str(error))}]

    def _receipts_dir(self, scope, project):
        return (self.registry.config_root / "mlx-agent" / "installer-receipts") if scope == "user" else (project / ".mlx-agent" / "installer-receipts")

    def _create_batch(self, plan):
        batches = self._receipts_dir(plan.scope, plan.project_root) / "batches"
        directory, descriptor = _walk_directory(batches, create=True)
        batch_id = str(uuid.uuid4())
        try:
            os.mkdir(batch_id, 0o700, dir_fd=descriptor)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        path = directory / batch_id / "batch.json"
        immutable = {
            "schema_version": "1.0", "action": plan.action, "scope": plan.scope,
            "project_root": str(plan.project_root), "preview_hash": plan.preview["preview_hash"],
            "binding": plan.binding,
            "operations": [self._active_operation(item) for item in plan.transactions],
        }
        self._write_batch(path, dict(immutable, status="pending", children=[], active=None, remediation=[]))
        return path

    def _batch_update(self, path, status, receipts, active=None, error=None, remediation=None):
        value = self._read_batch(path)
        children = {item["receipt_path"]: item for item in value.get("children", [])}
        for receipt in receipts:
            children[receipt.receipt_path] = dict(self._receipt_identity(receipt), status=receipt.status)
        value["children"] = list(children.values())
        value["status"] = status
        value["active"] = active
        if error:
            value.setdefault("remediation", []).append({"paths": [], "message": redact_secrets(str(error))})
        if remediation:
            value.setdefault("remediation", []).extend(remediation)
        self._write_batch(path, value)

    def _read_batch(self, path):
        location = _assert_safe_target(path)
        value = json.loads(_read_regular(location).decode("utf-8"))
        required = {"schema_version", "action", "scope", "project_root", "preview_hash", "binding", "operations", "status", "children", "active", "remediation"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("installer batch journal is malformed")
        return value

    def _write_batch(self, path, value):
        root = _assert_safe_directory(Path(path).parent)
        _atomic_in_directory(root, "batch.json", (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"), 0o600)

    @staticmethod
    def _active_operation(operation):
        return {"provider": operation.provider_id, "transaction_id": operation.transaction_id,
                "expected_receipt_path": operation.expected_receipt_path,
                "preview_hash": operation.preview_hash, "targets": list(operation.destinations)}

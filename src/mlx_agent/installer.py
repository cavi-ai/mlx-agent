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
    _atomic_in_directory, _read_regular, _walk_directory, rollback,
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
                self._batch_update(batch_path, "pending", successful, operation.provider_id)
                receipt = transaction.apply(operation.preview_hash)
                all_receipts.append(receipt)
                if receipt.status != "applied":
                    raise InstallerConflictError("provider transaction did not apply: {0}".format(receipt.status))
                successful.append(receipt)
            self._batch_update(batch_path, "complete", successful)
            return InstallerReceipt("applied", all_receipts, successful[-1].receipt_path, [target for item in successful for target in item.targets], str(batch_path))
        except Exception as error:
            recovered = self._compensate_install(successful)
            status = "rolled_back" if recovered else "recovery_required"
            self._batch_update(batch_path, status, successful, error=str(error))
            return InstallerReceipt(status, all_receipts, all_receipts[-1].receipt_path if all_receipts else "", [target for item in all_receipts for target in item.targets], str(batch_path))

    def _execute_uninstall(self, plan, batch_path):
        completed = []
        try:
            for receipt in plan.rollback_receipts:
                self._batch_update(batch_path, "pending", completed, receipt.transaction_id)
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
            recovered = self._compensate_uninstall(plan.restore_changes, completed, plan.scope, plan.project_root)
            status = "rolled_back" if recovered else "recovery_required"
            self._batch_update(batch_path, status, completed, error=str(error))
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
                transactions.append(_PlannedTransaction(provider_id, tuple(changes), preview["preview_hash"], tuple(str(item["path"]) for item in changes)))
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
        return {"healthy": not problems, "problems": problems, "checked": checked,
                "cooperative_concurrency": "Transaction advisory locks protect cooperative writers."}

    def _make_plan(self, action, selected, scope, project, transactions=(), rollback_receipts=(),
                   uninstall_hashes=None, restore_changes=(), summary=(), definitions=None):
        definitions = definitions or self.registry.definitions()
        destinations = [{"provider": provider_id, "targets": [str(definitions[provider_id].destination(scope, project) / artifact.destination) for artifact in definitions[provider_id].artifacts]} for provider_id in selected]
        binding = {
            "action": action, "providers": list(selected), "scope": scope, "project": str(project),
            "destinations": destinations,
            "inner_previews": [{"provider": item.provider_id, "hash": item.preview_hash, "targets": list(item.destinations)} for item in transactions],
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
            "inner_previews": [{"provider": item.provider_id, "hash": item.preview_hash, "targets": list(item.destinations)} for item in plan.transactions],
            "rollback_chain": [self._receipt_identity(self._load_receipt(item.receipt_path, plan.scope, plan.project_root)) for item in plan.rollback_receipts],
        }
        if binding != plan.binding or _digest(binding) != plan.preview["preview_hash"]:
            raise PermissionError("installer preview is stale; regenerate and review it before confirmation")

    def _repreview(self, operation, scope, project):
        transaction = self._new_transaction(scope, project)
        preview = transaction.preview(list(operation.changes))
        if preview["preview_hash"] != operation.preview_hash:
            raise PermissionError("installer preview is stale; an inner preview changed")
        return transaction, preview

    def _new_transaction(self, scope, project):
        return self.transaction_factory(receipts_dir=self._receipts_dir(scope, project), create_target_parents=True)

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
        try:
            for receipt in reversed(receipts):
                result = self.rollback_func(receipt.receipt_path, expected_after_hashes=receipt.after_hashes)
                if result.status != "rolled_back":
                    return False
            return True
        except Exception:
            return False

    def _compensate_uninstall(self, changes, completed, scope, project):
        if not changes:
            return True
        try:
            affected = {target for receipt in completed for target in receipt.targets}
            for _provider_id, provider_changes in changes:
                provider_changes = [change for change in provider_changes if change["path"] in affected]
                if not provider_changes:
                    continue
                transaction = self._new_transaction(scope, project)
                preview = transaction.preview(provider_changes)
                if transaction.apply(preview["preview_hash"]).status != "applied":
                    return False
            return True
        except Exception:
            return False

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
        self._batch_update(path, "pending", [], plan=plan)
        return path

    def _batch_update(self, path, status, receipts, active=None, error=None, plan=None):
        root = _assert_safe_directory(Path(path).parent)
        value = {
            "schema_version": "1.0", "status": status,
            "receipts": [self._receipt_identity(item) for item in receipts],
            "active": active, "error": redact_secrets(error) if error else None,
        }
        if plan is not None:
            value["action"] = plan.action
            value["preview_hash"] = plan.preview["preview_hash"]
            value["binding"] = plan.binding
        _atomic_in_directory(root, "batch.json", (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"), 0o600)

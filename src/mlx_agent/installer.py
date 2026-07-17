"""Scoped, receipt-owned installer built on the Transaction safety boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .providers import detect_providers
from .transactions import Receipt, Transaction, _assert_safe_target, _read_regular, rollback
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


@dataclass
class InstallerReceipt:
    status: str
    receipts: list = field(default_factory=list)
    receipt_path: str = ""
    targets: list = field(default_factory=list)

    def to_dict(self):
        return {
            "status": self.status,
            "receipt_path": self.receipt_path or None,
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


class Installer:
    """Plan first; execute only a confirmed preview through Transactions."""

    def __init__(self, registry, project_root=None, executable_lookup=None, env=None):
        self.registry = registry
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
        self.executable_lookup = executable_lookup
        self.env = env

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
        if plan.action == "uninstall":
            for target, expected in plan.uninstall_hashes.items():
                location = _assert_safe_target(target)
                if not location.is_file() or _sha256(_read_regular(location)) != expected:
                    raise InstallerConflictError("refusing to remove user-modified artifact: {0}".format(location))
            receipts = []
            for receipt in plan.rollback_receipts:
                result = rollback(receipt.receipt_path)
                if result.status != "rolled_back":
                    raise InstallerConflictError("uninstall rollback did not complete: {0}".format(result.status))
                receipts.append(result)
            return InstallerReceipt("rolled_back", receipts, receipts[-1].receipt_path if receipts else "", [target for item in receipts for target in item.targets])
        receipts = []
        for transaction, preview in plan.transactions:
            receipt = transaction.apply(preview["preview_hash"])
            if receipt.status != "applied":
                return InstallerReceipt(receipt.status, receipts + [receipt], receipt.receipt_path, receipt.targets)
            receipts.append(receipt)
        return InstallerReceipt("applied", receipts, receipts[-1].receipt_path if receipts else "", [target for item in receipts for target in item.targets])

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
                    history = self._provider_history(definition, scope, project)
                    if not history or str(target) not in history[-1].after_hashes or _sha256(current) != history[-1].after_hashes[str(target)]:
                        raise InstallerConflictError("refusing to overwrite unowned or modified artifact: {0}".format(target))
                    if current == desired.encode("utf-8"):
                        continue
                changes.append({"path": str(target), "content": desired, "adapter": _ArtifactAdapter()})
            if changes:
                transaction = Transaction(receipts_dir=self._receipts_dir(scope, project), create_target_parents=True)
                preview = transaction.preview(changes)
                transactions.append((transaction, preview))
                summary.append({"provider": provider_id, "changes": preview["changes"], "diff": preview["diff"]})
        preview = self._preview(action, selected, scope, project, summary)
        return InstallPlan(action, selected, scope, project, preview, transactions, noop=not transactions)

    def _uninstall_plan(self, selected, definitions, scope, project):
        rollback_receipts, summary, uninstall_hashes = [], [], {}
        for provider_id in selected:
            definition = definitions[provider_id]
            history = self._provider_history(definition, scope, project)
            if not history:
                continue
            oldest, latest = history[0], history[-1]
            for artifact in definition.artifacts:
                target = str(definition.destination(scope, project) / artifact.destination)
                if oldest.backup_paths.get(target) is not None:
                    raise InstallerConflictError("artifact was not receipt-owned at installation: {0}".format(target))
                location = Path(target)
                if not location.is_file() or _sha256(location.read_bytes()) != latest.after_hashes.get(target):
                    raise InstallerConflictError("refusing to remove user-modified artifact: {0}".format(target))
                uninstall_hashes[target] = latest.after_hashes[target]
            rollback_receipts.extend(reversed(history))
            summary.append({"provider": provider_id, "changes": len(latest.targets), "diff": "remove receipt-owned artifacts"})
        preview = self._preview("uninstall", selected, scope, project, summary)
        return InstallPlan("uninstall", selected, scope, project, preview, rollback_receipts=rollback_receipts, uninstall_hashes=uninstall_hashes, noop=not rollback_receipts)

    def _doctor_plan(self, selected, definitions, scope, project):
        return InstallPlan("doctor", selected, scope, project, self._preview("doctor", selected, scope, project, []))

    def _doctor(self, plan):
        problems, checked = [], []
        definitions = self.registry.definitions()
        for provider_id in plan.provider_ids:
            definition = definitions[provider_id]
            history = self._provider_history(definition, plan.scope, plan.project_root)
            if not history:
                problems.append({"provider": provider_id, "code": "not_installed"})
                continue
            latest = history[-1]
            for artifact in definition.artifacts:
                target = definition.destination(plan.scope, plan.project_root) / artifact.destination
                checked.append(str(target))
                if not target.is_file() or _sha256(target.read_bytes()) != latest.after_hashes.get(str(target)):
                    problems.append({"provider": provider_id, "code": "artifact_invalid", "path": str(target)})
        return {"healthy": not problems, "problems": problems, "checked": checked, "cooperative_concurrency": "Transaction advisory locks protect cooperative writers."}

    def _provider_history(self, definition, scope, project):
        targets = {str(definition.destination(scope, project) / artifact.destination) for artifact in definition.artifacts}
        receipts = []
        receipts_dir = self._receipts_dir(scope, project)
        if not receipts_dir.is_dir():
            return receipts
        for candidate in receipts_dir.glob("*/receipt.json"):
            try:
                location = _assert_safe_target(candidate)
                receipt = Receipt.from_dict(json.loads(_read_regular(location).decode("utf-8")), str(location))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if set(receipt.targets) == targets and receipt.status == "applied":
                receipts.append(receipt)
        return sorted(receipts, key=lambda item: item.timestamp)

    def _receipts_dir(self, scope, project):
        return (self.registry.config_root / "mlx-agent" / "installer-receipts") if scope == "user" else (project / ".mlx-agent" / "installer-receipts")

    @staticmethod
    def _preview(action, providers, scope, project, changes):
        binding = {"action": action, "providers": list(providers), "scope": scope, "project": str(project), "changes": changes}
        preview_hash = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return {"changes": sum(item["changes"] for item in changes), "providers": list(providers), "diff": "\n".join(item["diff"] for item in changes), "preview_hash": preview_hash, "requires_confirmation": action != "doctor"}

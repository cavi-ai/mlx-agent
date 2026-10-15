"""Link a cached-base/adapter comparison to its existing training receipt."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

from mlx_agent.compute import ComputeLease
from mlx_agent.lab import _run_child


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preview(receipt_path, prompt, max_tokens=128):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
        raise ValueError("Evaluation prompt must contain 1..8000 characters.")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
        raise ValueError("Evaluation token limit must be 1..512.")
    from mlx_agent.training_artifacts import read_trained_adapter
    receipt, binding = read_trained_adapter(receipt_path)
    path = Path(binding["training_receipt"])
    model = Path(binding["base_path"])
    identity = binding["adapter_identity"]
    files = [p for p in model.rglob("*") if p.is_file()]
    if not files or len(files) > 256:
        raise ValueError("Base-model inventory is unavailable or exceeds its bound.")
    inventory = {str(p.relative_to(model)): {"sha256": _digest(p), "bytes": p.stat().st_size}
                 for p in sorted(files)}
    plan = {"schema_version": "text-training-evaluation/1", "training_receipt": str(path),
            "training_receipt_sha256": binding["training_receipt_sha256"], "training_run_id": receipt.get("run_id"),
            "base_repo": receipt["repo"], "base_path": str(model), "base_revision": model.name,
            "base_inventory": inventory, "dataset": receipt.get("dataset"), "adapter_path": binding["adapter_path"],
            "adapter_identity": identity, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "seed": 42}
    plan["preview_hash"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return plan


def history(database):
    with database.connect() as connection:
        present = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='text_training_evaluations'").fetchone()
        if not present:
            return []
        return [dict(row) for row in connection.execute(
            "SELECT id, training_receipt, training_run_id, artifact_id, artifact_sha256, created_at FROM text_training_evaluations ORDER BY created_at DESC LIMIT 200")]


def inspect(database, artifacts, evaluation_id):
    with database.connect() as connection:
        present = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='text_training_evaluations'").fetchone()
        row = connection.execute("SELECT * FROM text_training_evaluations WHERE id=?", (evaluation_id,)).fetchone() if present else None
    record = dict(row) if row else None
    if record is None:
        raise ValueError("Evaluation was not found in this workspace.")
    artifact = artifacts.get(record["artifact_id"])
    if artifact.sha256 != record["artifact_sha256"] or artifact.byte_size > 1024 * 1024:
        raise ValueError("Evaluation artifact does not match its retained record.")
    with database.connect() as connection:
        with artifacts.open_verified_with_connection(connection, artifact.id,
                sha256=artifact.sha256, byte_size=artifact.byte_size,
                media_type="application/json") as handle:
            content = os.read(handle.descriptor, artifact.byte_size + 1)
            artifacts.reverify_open_handle(connection, handle)
    report = json.loads(content)
    return {"evaluation_id": evaluation_id, "artifact_sha256": artifact.sha256, "report": report}


def run(database, artifacts, request, generate=None):
    with ComputeLease("text-training-evaluation") as lease:
        plan = preview(request["receipt"], request["prompt"], request.get("max_tokens", 128))
        if request.get("preview_hash") != plan["preview_hash"]:
            raise ValueError("The evaluation plan changed; review a fresh preview.")
        with tempfile.TemporaryDirectory(prefix="mlx-training-evaluation-") as directory:
            folder = Path(directory)
            file = folder / "plan.json"
            file.write_text(json.dumps(plan))
            outputs = {}
            for variant in ("baseline", "adapter"):
                if generate is not None:
                    outputs[variant] = generate(plan, variant)
                    continue
                destination = folder / (variant + ".json")
                environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
                result = _run_child([sys.executable, "-m", "mlx_agent.training_eval_runtime",
                                     str(file), variant, str(destination)], environment, timeout=900,
                                    pass_fds=(lease.descriptor,))
                if result.returncode != 0:
                    raise ValueError("Local evaluation failed: " + result.stderr[-2048:])
                if not destination.is_file() or destination.stat().st_size > 1024 * 1024:
                    raise ValueError("Evaluation produced no bounded output.")
                outputs[variant] = json.loads(destination.read_text())["text"]
            if preview(request["receipt"], request["prompt"], request.get("max_tokens", 128)) != plan:
                raise ValueError("Evaluation inputs changed during generation; no result was accepted.")
        if any(not isinstance(text, str) or not text.strip() for text in outputs.values()):
            raise ValueError("Evaluation returned an empty response.")
        report = {**plan, "outputs": outputs, "judgment": None}
        staged = artifacts.stage_bytes(json.dumps(report, sort_keys=True).encode(), "application/json")
        artifact = artifacts.commit(staged, None)
        identity = "text_eval_" + uuid.uuid4().hex
        with database.connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS text_training_evaluations (
                id TEXT PRIMARY KEY, training_receipt TEXT NOT NULL, training_run_id TEXT,
                artifact_id TEXT NOT NULL REFERENCES artifacts(id), artifact_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))""")
            connection.execute("INSERT INTO text_training_evaluations(id,training_receipt,training_run_id,artifact_id,artifact_sha256) VALUES (?,?,?,?,?)",
                               (identity, plan["training_receipt"], plan["training_run_id"], artifact.id, artifact.sha256))
        return {"evaluation_id": identity, "artifact_id": artifact.id, "artifact_sha256": artifact.sha256,
                "report": report, "path": str(artifacts.resolve(artifact.id))}

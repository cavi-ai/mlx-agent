"""Link a cached-base/adapter comparison to its existing training receipt."""
from __future__ import annotations
import hashlib
import json
import os
import copy
import re
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


def _prompt(text, limit):
    if not isinstance(text, str) or not text.strip() or len(text) > 8000:
        raise ValueError("Evaluation prompt must contain 1..8000 characters.")
    if type(limit) is not int or not 1 <= limit <= 512:
        raise ValueError("Evaluation token limit must be 1..512.")


def validate_suite(value):
    if not isinstance(value, dict) or set(value) != {"id", "name", "prompts"}:
        raise ValueError("An evaluation suite needs id, name and prompts only.")
    if not isinstance(value["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value["id"]):
        raise ValueError("Invalid evaluation suite id.")
    name = value["name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 200 or any(ord(c) < 32 for c in name):
        raise ValueError("Evaluation suite name must contain 1..200 characters without controls.")
    prompts = value["prompts"]
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= 16:
        raise ValueError("An evaluation suite must contain 1..16 prompts.")
    identities = set()
    for entry in prompts:
        if not isinstance(entry, dict) or set(entry) != {"id", "text", "max_tokens"}:
            raise ValueError("Evaluation entries support id, text and max_tokens only.")
        identity = entry["id"]
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity) or identity in identities:
            raise ValueError("Evaluation prompt ids must be valid and unique.")
        identities.add(identity)
        _prompt(entry["text"], entry["max_tokens"])
    return copy.deepcopy(value)


def preview(receipt_path, prompt=None, max_tokens=128, prompt_suite=None):
    if prompt_suite is not None:
        if prompt is not None or max_tokens != 128:
            raise ValueError("Choose either a prompt suite or a single prompt and token limit.")
        prompt_suite = validate_suite(prompt_suite)
    else:
        _prompt(prompt, max_tokens)
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
    if prompt_suite is not None:
        plan.update(schema_version="text-training-evaluation/2", prompt_suite=prompt_suite)
        del plan["prompt"], plan["max_tokens"]
    plan["preview_hash"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return plan


def history(database):
    with database.connect() as connection:
        present = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='text_training_evaluations'").fetchone()
        if not present:
            return []
        return [dict(row) for row in connection.execute(
            "SELECT * FROM text_training_evaluations ORDER BY created_at DESC, rowid DESC LIMIT 200")]


def latest_reviews(database, evaluation_id, digest):
    with database.connect() as connection:
        present = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='text_training_evaluation_reviews'").fetchone()
        if not present:
            return []
        return [dict(row) for row in connection.execute("""SELECT r.* FROM text_training_evaluation_reviews r
            WHERE evaluation_id=? AND artifact_sha256=? AND NOT EXISTS (
                SELECT 1 FROM text_training_evaluation_reviews newer WHERE newer.evaluation_id=r.evaluation_id
                AND newer.prompt_id=r.prompt_id AND newer.artifact_sha256=r.artifact_sha256
                AND newer.rowid>r.rowid) ORDER BY r.rowid""", (evaluation_id, digest))]


def report_samples(report):
    if report.get("schema_version") == "text-training-evaluation/2":
        return report["samples"]
    if report.get("schema_version") == "text-training-evaluation/1":
        return [{"id": "prompt", "prompt": report["prompt"], "max_tokens": report["max_tokens"], "outputs": report["outputs"]}]
    raise ValueError("Unsupported retained evaluation schema.")


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
    return {"evaluation_id": evaluation_id, "artifact_sha256": artifact.sha256, "report": report,
            "reviews": latest_reviews(database, evaluation_id, artifact.sha256)}


def run(database, artifacts, request, generate=None):
    with ComputeLease("text-training-evaluation") as lease:
        plan = preview(request["receipt"], request.get("prompt"), request.get("max_tokens", 128), request.get("prompt_suite"))
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
                output = json.loads(destination.read_text())
                outputs[variant] = output["samples"] if "prompt_suite" in plan else output["text"]
            if preview(request["receipt"], request.get("prompt"), request.get("max_tokens", 128), request.get("prompt_suite")) != plan:
                raise ValueError("Evaluation inputs changed during generation; no result was accepted.")
        if "prompt_suite" in plan:
            prompts = plan["prompt_suite"]["prompts"]
            for variant in outputs:
                values = outputs[variant]
                if not isinstance(values, list) or len(values) != len(prompts) or any(
                    not isinstance(value, dict) or value.get("id") != entry["id"]
                    or not isinstance(value.get("text"), str) or not value["text"].strip()
                    for entry, value in zip(prompts, values)):
                    raise ValueError("Evaluation did not return every paired prompt in order.")
            samples = [{"id": entry["id"], "prompt": entry["text"], "max_tokens": entry["max_tokens"],
                        "outputs": {variant: outputs[variant][index]["text"] for variant in outputs}}
                       for index, entry in enumerate(prompts)]
            report = {**plan, "samples": samples, "judgment": None}
        else:
            if any(not isinstance(text, str) or not text.strip() for text in outputs.values()):
                raise ValueError("Evaluation returned an empty response.")
            report = {**plan, "outputs": outputs, "judgment": None}
        encoded = json.dumps(report, sort_keys=True).encode()
        if len(encoded) > 1024 * 1024:
            raise ValueError("Evaluation report exceeds its retained size bound.")
        staged = artifacts.stage_bytes(encoded, "application/json")
        artifact = artifacts.commit(staged, None)
        identity = "text_eval_" + uuid.uuid4().hex
        with database.connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS text_training_evaluations (
                id TEXT PRIMARY KEY, training_receipt TEXT NOT NULL, training_run_id TEXT,
                artifact_id TEXT NOT NULL REFERENCES artifacts(id), artifact_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))""")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(text_training_evaluations)")}
            for name, kind in (("prompt_count", "INTEGER"), ("suite_name", "TEXT")):
                if name not in columns:
                    connection.execute("ALTER TABLE text_training_evaluations ADD COLUMN " + name + " " + kind)
            connection.execute("INSERT INTO text_training_evaluations(id,training_receipt,training_run_id,artifact_id,artifact_sha256,prompt_count,suite_name) VALUES (?,?,?,?,?,?,?)",
                               (identity, plan["training_receipt"], plan["training_run_id"], artifact.id, artifact.sha256,
                                len(report_samples(report)), plan.get("prompt_suite", {}).get("name")))
        return {"evaluation_id": identity, "artifact_id": artifact.id, "artifact_sha256": artifact.sha256,
                "report": report, "reviews": [], "path": str(artifacts.resolve(artifact.id))}


def review(database, artifacts, reviewer, request):
    identity, digest, prompt_id = request["evaluation_id"], request["artifact_sha256"], request["prompt_id"]
    preference, notes = request["preference"], request.get("notes", "")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("A retained evaluation SHA-256 is required.")
    if preference not in ("baseline", "adapter", "tie", "neither") or not isinstance(notes, str) or len(notes) > 2000:
        raise ValueError("Choose a supported preference and notes of at most 2000 characters.")
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 256:
        raise ValueError("A bounded reviewer identity is required.")
    with database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        if not connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='text_training_evaluations'").fetchone():
            raise ValueError("Evaluation was not found in this workspace.")
        record = connection.execute("SELECT * FROM text_training_evaluations WHERE id=?", (identity,)).fetchone()
        if record is None or record["artifact_sha256"] != digest:
            raise ValueError("The retained evaluation identity changed; inspect it again.")
        artifact = artifacts.get(record["artifact_id"])
        if artifact.sha256 != digest or artifact.byte_size > 1024 * 1024:
            raise ValueError("The retained evaluation artifact changed.")
        with artifacts.open_verified_with_connection(connection, artifact.id, sha256=digest,
                byte_size=artifact.byte_size, media_type="application/json") as handle:
            report = json.loads(os.read(handle.descriptor, artifact.byte_size + 1))
            if prompt_id not in {sample["id"] for sample in report_samples(report)}:
                raise ValueError("The prompt is not part of this retained evaluation.")
            connection.execute("""CREATE TABLE IF NOT EXISTS text_training_evaluation_reviews (
                id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL REFERENCES text_training_evaluations(id),
                artifact_sha256 TEXT NOT NULL, prompt_id TEXT NOT NULL, reviewer_id TEXT NOT NULL,
                preference TEXT NOT NULL, notes TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))""")
            review_id = "text_review_" + uuid.uuid4().hex
            connection.execute("INSERT INTO text_training_evaluation_reviews(id,evaluation_id,artifact_sha256,prompt_id,reviewer_id,preference,notes) VALUES (?,?,?,?,?,?,?)",
                (review_id, identity, digest, prompt_id, reviewer, preference, notes))
            artifacts.reverify_open_handle(connection, handle)
    return {"evaluation_id": identity, "artifact_sha256": digest, "review_id": review_id,
            "reviews": latest_reviews(database, identity, digest)}

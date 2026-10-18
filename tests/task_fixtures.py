"""Valid private task evidence for contract and CLI tests."""

import hashlib

from mlx_agent.task_contracts import encoded


def identity_evidence(model_path):
    manifest = {"id": "mlx-embeddings", "version": "1.0.0"}
    return {
        "schema": "task-identity/1",
        "requested_model_path": model_path,
        "model_path": model_path,
        "model_fingerprint": "a" * 64,
        "backend": manifest["id"],
        "backend_version": manifest["version"],
        "backend_manifest": manifest,
        "backend_manifest_hash": hashlib.sha256(encoded(manifest)).hexdigest(),
        "requested_executable": "/usr/bin/python3",
        "executable": "/usr/bin/python3",
        "executable_identity": [1, 2, 3, 4],
        "engine_version": "1.0.0",
        "python_version": "3.12.0",
    }

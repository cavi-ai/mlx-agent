"""Generic dataset commands run through the real standalone agent entrypoint."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def document():
    return {"schema_version": "training-dataset.v1", "samples": [{
        "sample_id": "sample-a", "source_lineage": "source-a", "caption": "A moving object",
        "split": "train", "media": {"sha256": "sha256:" + "a" * 64,
        "media_type": "image/png", "byte_size": 12, "width": 16, "height": 16}
    }]}


def invoke(tmp_path, action, payload=None, relative="request.json"):
    if payload is not None:
        (tmp_path / "request.json").write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        )
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/mlx-agent"), "lab",
         "--database", str(tmp_path / "unused.sqlite3"),
         "--artifacts", str(tmp_path / "unused-artifacts"),
         "training-data", action, "--request-root", str(tmp_path),
         "--request-file", relative, "--json"],
        check=False, capture_output=True, text=True, timeout=15,
    )


def test_freeze_and_inspect_roundtrip_without_creating_workspace(tmp_path):
    result = invoke(tmp_path, "freeze", document())
    assert result.returncode == 0, result.stdout + result.stderr
    envelope = json.loads(result.stdout)
    assert envelope["status"] == "ok"
    frozen = envelope["data"]
    assert frozen["dataset_digest"].startswith("sha256:")
    assert frozen["samples"][0]["caption"] == "A moving object"
    result = invoke(tmp_path, "inspect", frozen)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["data"]["sample_count"] == 1
    assert not (tmp_path / "unused.sqlite3").exists()
    assert not (tmp_path / "unused-artifacts").exists()


def test_validate_rejects_split_leakage_with_reason_codes(tmp_path):
    payload = document()
    payload["samples"].append({**payload["samples"][0], "sample_id": "sample-b",
        "split": "validation", "media": {**payload["samples"][0]["media"],
        "sha256": "sha256:" + "b" * 64}})
    result = invoke(tmp_path, "validate", payload)
    assert result.returncode != 0
    error = json.loads(result.stdout)["error"]
    assert error["code"] == "dataset_not_ready"
    assert error["reason_codes"] == ["lineage_split_leakage"]


def test_inspect_rejects_modified_snapshot(tmp_path):
    result = invoke(tmp_path, "freeze", document())
    assert result.returncode == 0, result.stdout + result.stderr
    frozen = json.loads(result.stdout)["data"]
    frozen["samples"][0]["caption"] = "Modified"
    result = invoke(tmp_path, "inspect", frozen)
    assert result.returncode != 0
    assert json.loads(result.stdout)["error"]["code"] == "rejected"


@pytest.mark.parametrize("kind", ["symlink", "fifo", "oversize"])
def test_input_boundary_rejects_unsafe_files(tmp_path, kind):
    path = tmp_path / "request.json"
    if kind == "symlink":
        (tmp_path / "target").write_text("{}")
        path.symlink_to(tmp_path / "target")
    elif kind == "fifo":
        import os
        os.mkfifo(path)
    else:
        path.write_bytes(b" " * (1024 * 1024 + 1))
    result = invoke(tmp_path, "validate")
    assert result.returncode != 0
    assert json.loads(result.stdout)["status"] == "error"


def test_freeze_canonicalizes_sample_order(tmp_path):
    payload = document()
    second = {**payload["samples"][0], "sample_id": "sample-b", "source_lineage": "source-b",
              "media": {**payload["samples"][0]["media"], "sha256": "sha256:" + "b" * 64}}
    payload["samples"].insert(0, second)
    first = invoke(tmp_path, "freeze", payload)
    assert first.returncode == 0, first.stdout + first.stderr
    payload["samples"].reverse()
    second = invoke(tmp_path, "freeze", payload)
    assert second.returncode == 0, second.stdout + second.stderr
    assert json.loads(first.stdout)["data"] == json.loads(second.stdout)["data"]


def test_invalid_command_returns_structured_error(tmp_path):
    result = invoke(tmp_path, "unknown", document())
    assert result.returncode != 0
    assert json.loads(result.stdout)["error"]["code"] == "usage_error"


def test_validate_returns_metadata_summary(tmp_path):
    result = invoke(tmp_path, "validate", document())
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)["data"]
    assert data["sample_count"] == 1
    assert data["split_counts"] == {"train": 1}
    assert "samples" not in data

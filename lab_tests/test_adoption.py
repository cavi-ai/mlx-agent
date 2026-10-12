from pathlib import Path
import hashlib
import json
import pytest
from mlx_agent.lab_adoption import preview, adopt
from video_lab.database import Database
from video_lab.review_service import ReviewService
from lab_tests.test_review_service import _experiment, _service

def test_adoption_preserves_database_backup_artifacts_and_sealed_assignments(database, tmp_path):
    root = tmp_path / "artifacts"
    experiment, jobs, artifacts = _experiment(database, root)
    reviewer = _service(database, root)
    session = reviewer.start(experiment.id, "owner")
    original = database.path.read_bytes()
    target = tmp_path / "adopted"
    plan = preview(database.path, root, target)
    receipt = adopt(database.path, root, target, plan["preview_hash"])
    assert receipt["store_path"] == str(target)
    assert database.path.read_bytes() == original
    assert (target / "before-adoption.sqlite3").read_bytes() == original
    assert (target / "experiments.sqlite3").read_bytes() == original
    migrated = Database(target / "experiments.sqlite3")
    assert _service(migrated, target / "artifacts").start(experiment.id, "owner") == session
    for artifact in artifacts:
        output = target / "artifacts" / artifact.relative_path
        assert hashlib.sha256(output.read_bytes()).hexdigest() == artifact.sha256
    assert json.loads((target / "adoption.json").read_text())["source_database_digest"] == plan["source_database_digest"]

def test_adoption_refuses_active_jobs_and_never_changes_source(database, tmp_path):
    root = tmp_path / "artifacts"
    _, jobs, _ = _experiment(database, root)
    with database.connect() as connection:
        connection.execute("UPDATE jobs SET state='generating' WHERE id=?", (jobs[0].id,))
    original = database.path.read_bytes()
    with pytest.raises(ValueError, match="active"):
        preview(database.path, root, tmp_path / "adopted")
    assert database.path.read_bytes() == original
    assert not (tmp_path / "adopted").exists()

def test_adoption_refuses_stale_preview_and_existing_destination(database, tmp_path):
    root = tmp_path / "artifacts"
    _, jobs, _ = _experiment(database, root)
    target = tmp_path / "adopted"
    plan = preview(database.path, root, target)
    with database.connect() as connection:
        connection.execute("UPDATE jobs SET state='queued' WHERE id=?", (jobs[0].id,))
    original = database.path.read_bytes()
    with pytest.raises(ValueError, match="preview changed"):
        adopt(database.path, root, target, plan["preview_hash"])
    assert database.path.read_bytes() == original
    assert not target.exists()
    target.mkdir()
    marker = target / "owned"
    marker.write_text("retain")
    with pytest.raises(ValueError, match="never replaced"):
        preview(database.path, root, target)
    assert marker.read_text() == "retain"

def test_adoption_failed_publication_rolls_back_only_staging(database, tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    _experiment(database, root)
    target = tmp_path / "adopted"
    plan = preview(database.path, root, target)
    original = database.path.read_bytes()
    def fail(*_): raise OSError("injected publication failure")
    monkeypatch.setattr("mlx_agent.lab_adoption._publish", fail)
    with pytest.raises(OSError, match="injected"):
        adopt(database.path, root, target, plan["preview_hash"])
    assert database.path.read_bytes() == original
    assert not target.exists()
    assert not list(tmp_path.glob(".lab-adoption-*"))

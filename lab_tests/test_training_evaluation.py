import copy

import pytest

from mlx_agent import training_evaluation as evaluation
from video_lab.artifacts import ArtifactStore


def test_evaluation_retains_verified_outputs_and_history(database, tmp_path, monkeypatch):
    from mlx_agent.compute import ComputeLease
    monkeypatch.setattr(evaluation, "ComputeLease", lambda operation: ComputeLease(operation, root=tmp_path / "compute"))
    plan = {"schema_version": "text-training-evaluation/1", "preview_hash": "reviewed",
            "training_receipt": "/fixture/receipt.json", "training_run_id": "fixture-run",
            "prompt": "fixture", "max_tokens": 128}
    monkeypatch.setattr(evaluation, "preview", lambda *args: copy.deepcopy(plan))
    artifacts = ArtifactStore(database, tmp_path / "artifacts")
    result = evaluation.run(database, artifacts,
        {"receipt": "/fixture/receipt.json", "prompt": "fixture", "preview_hash": "reviewed"},
        generate=lambda plan, variant: "fixture " + variant)
    record = evaluation.history(database)[0]
    assert record["artifact_sha256"] == result["artifact_sha256"]
    retained = evaluation.inspect(database, artifacts, result["evaluation_id"])
    assert retained["report"]["outputs"] == {"baseline": "fixture baseline", "adapter": "fixture adapter"}
    assert retained["report"]["judgment"] is None
    artifacts.resolve(result["artifact_id"]).write_bytes(b"tampered")
    with pytest.raises(Exception, match="(integrity|digest|size|content|mismatch)"):
        evaluation.inspect(database, artifacts, result["evaluation_id"])


def test_changed_inputs_accept_no_evaluation(database, tmp_path, monkeypatch):
    from mlx_agent.compute import ComputeLease
    monkeypatch.setattr(evaluation, "ComputeLease", lambda operation: ComputeLease(operation, root=tmp_path / "compute"))
    plan = {"preview_hash": "reviewed", "training_receipt": "/fixture/receipt.json",
            "training_run_id": "fixture-run"}
    monkeypatch.setattr(evaluation, "preview", lambda *args: copy.deepcopy(plan))
    def generate(plan_arg, variant):
        if variant == "adapter":
            plan["preview_hash"] = "changed"
        return "fixture"
    artifacts = ArtifactStore(database, tmp_path / "artifacts")
    with pytest.raises(ValueError, match="inputs changed"):
        evaluation.run(database, artifacts,
            {"receipt": "/fixture/receipt.json", "prompt": "fixture", "preview_hash": "reviewed"}, generate=generate)
    assert evaluation.history(database) == []


def test_unreviewed_evaluation_does_not_generate(database, tmp_path, monkeypatch):
    from mlx_agent.compute import ComputeLease
    monkeypatch.setattr(evaluation, "ComputeLease", lambda operation: ComputeLease(operation, root=tmp_path / "compute"))
    monkeypatch.setattr(evaluation, "preview", lambda *args: {"preview_hash": "current"})
    with pytest.raises(ValueError, match="fresh preview"):
        evaluation.run(database, ArtifactStore(database, tmp_path / "artifacts"),
            {"receipt": "/fixture/receipt.json", "prompt": "fixture", "preview_hash": "old"},
            generate=lambda *args: pytest.fail("must not generate"))

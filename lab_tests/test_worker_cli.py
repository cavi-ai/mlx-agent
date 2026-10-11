import json
from dataclasses import asdict
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from lab_tests.test_capabilities import manifest as base_manifest
from lab_tests.test_cli import common
from video_lab.adapter_protocol import ProbeReport, ProbeRequest
from video_lab.adapters.fake import FakeVideoAdapter
from video_lab.cli import app
from video_lab.database import Database
from video_lab.domain import ExperimentDraft
from video_lab.repositories import JobRepository
from video_lab.services import ExperimentService

runner = CliRunner()


def assert_compact(result: Result) -> dict[str, object]:
    assert result.exit_code == 0
    payload: dict[str, object] = json.loads(result.stdout)
    assert result.stdout == json.dumps(payload, separators=(",", ":")) + "\n"
    assert payload["schema_version"] == "cli.v1"
    assert payload["ok"] is True
    return payload


def fake_draft() -> ExperimentDraft:
    manifest = base_manifest().model_copy(
        update={
            "model_id": "video-lab/fake",
            "adapter_id": "video-lab.fake.v1",
            "width": 64,
            "height": 48,
            "scheduler": "fake",
            "dtype": "uint8",
            "backend": "cpu",
            "device": "cpu",
        }
    )
    return ExperimentDraft(baseline=manifest, variable_path="inference_steps", alternate_value=8)


def create_jobs(tmp_path: Path) -> tuple[str, str]:
    view = ExperimentService.from_paths(tmp_path / "lab.sqlite3", tmp_path / "data").create(
        fake_draft(), FakeVideoAdapter().capabilities()
    )
    return view.jobs[0].id, view.jobs[1].id


def test_run_once_requires_explicit_nonblank_worker_id(tmp_path: Path) -> None:
    missing = runner.invoke(app, [*common(tmp_path), "worker", "run-once", "--json"])
    blank = runner.invoke(
        app, [*common(tmp_path), "worker", "run-once", "--worker-id", "   ", "--json"]
    )

    for result in (missing, blank):
        assert result.exit_code == 2
        payload = json.loads(result.stdout)
        assert payload["error"]["code"] == "invalid_input"
        assert "worker_id" in payload["error"]["message"]


def test_run_once_returns_exact_idle_outcome(tmp_path: Path) -> None:
    result = runner.invoke(
        app, [*common(tmp_path), "worker", "run-once", "--worker-id", "stable-a", "--json"]
    )

    assert assert_compact(result)["data"] == {
        "schema_version": "worker.v1",
        "status": "idle",
        "job_id": None,
        "attempt_id": None,
        "error_code": None,
    }


def test_run_once_completes_one_fake_job_and_status_uses_repositories(tmp_path: Path) -> None:
    first_job, second_job = create_jobs(tmp_path)

    ran = runner.invoke(
        app, [*common(tmp_path), "worker", "run-once", "--worker-id", "stable-a", "--json"]
    )
    outcome = assert_compact(ran)["data"]
    assert isinstance(outcome, dict)
    assert outcome["schema_version"] == "worker.v1"
    assert outcome["status"] == "completed"
    assert outcome["job_id"] in {first_job, second_job}
    assert outcome["error_code"] is None
    completed_job = outcome["job_id"]
    untouched_job = second_job if completed_job == first_job else first_job
    assert isinstance(completed_job, str)

    status = assert_compact(
        runner.invoke(app, [*common(tmp_path), "worker", "status", completed_job, "--json"])
    )["data"]
    assert isinstance(status, dict)
    assert status["schema_version"] == "worker.v1"
    assert status["job"]["state"] == "completed"
    assert status["current_attempt"] == status["attempts"][-1]
    assert [attempt["number"] for attempt in status["attempts"]] == [1]
    assert [event["id"] for event in status["events"]] == sorted(
        event["id"] for event in status["events"]
    )
    assert status["telemetry_count"] > 0
    assert len(status["artifacts"]) == 1
    assert status["artifacts"][0]["job_id"] == completed_job

    untouched = assert_compact(
        runner.invoke(app, [*common(tmp_path), "worker", "status", untouched_job, "--json"])
    )["data"]
    assert isinstance(untouched, dict)
    assert untouched["job"]["state"] == "queued"


def test_cancel_uses_repository_cancellation_semantics(tmp_path: Path) -> None:
    first_job, _ = create_jobs(tmp_path)

    cancelled = runner.invoke(app, [*common(tmp_path), "worker", "cancel", first_job, "--json"])

    assert assert_compact(cancelled)["data"] == {
        "schema_version": "worker.v1",
        "job": {
            **JobRepository(Database(tmp_path / "lab.sqlite3"))
            .get(first_job)
            .model_dump(mode="json")
        },
    }


def test_recover_requires_worker_id_and_returns_ordered_jobs(tmp_path: Path) -> None:
    create_jobs(tmp_path)
    database = Database(tmp_path / "lab.sqlite3")
    repository = JobRepository(database)
    claimed = repository.claim_next("stable-a")
    assert claimed is not None

    missing = runner.invoke(app, [*common(tmp_path), "worker", "recover", "--json"])
    assert missing.exit_code == 2
    assert "worker_id" in json.loads(missing.stdout)["error"]["message"]

    result = runner.invoke(
        app, [*common(tmp_path), "worker", "recover", "--worker-id", "stable-a", "--json"]
    )

    data = assert_compact(result)["data"]
    assert isinstance(data, dict)
    assert data == {
        "schema_version": "worker.v1",
        "recovered_count": 1,
        "jobs": [repository.get(claimed.job.id).model_dump(mode="json")],
    }


def test_adapter_probe_resolves_exact_default_without_creating_database(tmp_path: Path) -> None:
    database = tmp_path / "lab.sqlite3"
    result = runner.invoke(
        app,
        [
            "--database",
            str(database),
            "adapter",
            "probe",
            "video-lab.fake.v1",
            "--json",
        ],
    )

    assert assert_compact(result)["data"] == {
        "schema_version": "worker.v1",
        "adapter_id": "video-lab.fake.v1",
        "available": True,
    }
    assert not database.exists()


def test_adapter_probe_unknown_adapter_is_typed_error_without_database(tmp_path: Path) -> None:
    database = tmp_path / "lab.sqlite3"
    result = runner.invoke(
        app, ["--database", str(database), "adapter", "probe", "missing", "--json"]
    )

    assert result.exit_code == 3
    payload = json.loads(result.stdout)
    assert payload["error"] == {
        "code": "adapter_not_found",
        "message": "unknown adapter: missing",
    }
    assert not database.exists()


class StagedProbeAdapter(FakeVideoAdapter):
    adapter_id = "test.staged.v1"

    def __init__(self, reports: dict[str, ProbeReport]) -> None:
        super().__init__()
        self.reports = reports
        self.requests: list[tuple[str, ProbeRequest | None]] = []

    def probe_phase(self, phase: str, request: ProbeRequest | None = None) -> ProbeReport:
        self.requests.append((phase, request))
        return self.reports[phase]


def install_staged_adapter(monkeypatch, adapter: StagedProbeAdapter) -> None:
    class Registry:
        def resolve(self, adapter_id: str) -> StagedProbeAdapter:
            assert adapter_id == adapter.adapter_id
            return adapter

    monkeypatch.setattr("video_lab.cli.AdapterRegistry.with_defaults", lambda: Registry())


@pytest.mark.parametrize("phase", ["metadata", "cache", "load"])
def test_adapter_probe_exposes_explicit_non_generating_phases_with_full_report(
    tmp_path: Path, monkeypatch, phase: str
) -> None:
    report = ProbeReport(phase=phase, ok=True, diagnostics={"resolved_revision": "abc123"})
    adapter = StagedProbeAdapter({phase: report})
    install_staged_adapter(monkeypatch, adapter)

    result = runner.invoke(
        app, ["adapter", "probe", adapter.adapter_id, "--phase", phase, "--json"]
    )

    assert assert_compact(result)["data"] == {
        "schema_version": "worker.v1",
        "adapter_id": adapter.adapter_id,
        "available": True,
        "probe_report": asdict(report),
    }
    assert adapter.requests == [(phase, None)]
    assert not (tmp_path / "lab.sqlite3").exists()


def test_adapter_probe_preserves_failed_report_diagnostics_fail_closed(monkeypatch) -> None:
    report = ProbeReport(
        phase="load",
        ok=False,
        diagnostics={"error_type": "RuntimeError", "error": "MPS unavailable"},
    )
    adapter = StagedProbeAdapter({"load": report})
    install_staged_adapter(monkeypatch, adapter)

    result = runner.invoke(
        app, ["adapter", "probe", adapter.adapter_id, "--phase", "load", "--json"]
    )

    assert assert_compact(result)["data"] == {
        "schema_version": "worker.v1",
        "adapter_id": adapter.adapter_id,
        "available": False,
        "probe_report": asdict(report),
    }


def test_adapter_probe_rejects_invalid_phase_before_adapter_resolution(monkeypatch) -> None:
    monkeypatch.setattr(
        "video_lab.cli.AdapterRegistry.with_defaults",
        lambda: pytest.fail("invalid phase must fail before adapter resolution"),
    )

    result = runner.invoke(
        app, ["adapter", "probe", "test.staged.v1", "--phase", "download", "--json"]
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "invalid_input"
    assert "phase" in payload["error"]["message"]


def test_adapter_generate_probe_requires_explicit_confirmation_and_safe_scratch(
    tmp_path: Path, monkeypatch
) -> None:
    report = ProbeReport("generate", True, {"video_path": str(tmp_path / "probe/output.mp4")})
    adapter = StagedProbeAdapter({"generate": report})
    install_staged_adapter(monkeypatch, adapter)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(base_manifest().model_dump_json(), encoding="utf-8")
    scratch = tmp_path / "probe"

    missing_confirmation = runner.invoke(
        app,
        [
            "adapter",
            "probe",
            adapter.adapter_id,
            "--phase",
            "generate",
            "--manifest",
            str(manifest_path),
            "--scratch-root",
            str(scratch),
            "--json",
        ],
    )
    assert missing_confirmation.exit_code == 2
    assert adapter.requests == []

    result = runner.invoke(
        app,
        [
            "adapter",
            "probe",
            adapter.adapter_id,
            "--phase",
            "generate",
            "--manifest",
            str(manifest_path),
            "--scratch-root",
            str(scratch),
            "--allow-generate",
            "--json",
        ],
    )

    data = assert_compact(result)["data"]
    assert data == {
        "schema_version": "worker.v1",
        "adapter_id": adapter.adapter_id,
        "available": True,
        "probe_report": asdict(report),
    }
    assert len(adapter.requests) == 1
    phase, request = adapter.requests[0]
    assert phase == "generate"
    assert request is not None
    assert request.generation.scratch_root == scratch.resolve()
    assert request.generation.job_id == "cli-probe"
    assert request.generation.attempt_id == "cli-probe"
    assert scratch.is_dir()


def test_adapter_generate_probe_rejects_existing_scratch_without_invoking_adapter(
    tmp_path: Path, monkeypatch
) -> None:
    adapter = StagedProbeAdapter({"generate": ProbeReport("generate", True, {})})
    install_staged_adapter(monkeypatch, adapter)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(base_manifest().model_dump_json(), encoding="utf-8")
    scratch = tmp_path / "existing"
    scratch.mkdir()

    result = runner.invoke(
        app,
        [
            "adapter",
            "probe",
            adapter.adapter_id,
            "--phase",
            "generate",
            "--manifest",
            str(manifest_path),
            "--scratch-root",
            str(scratch),
            "--allow-generate",
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "invalid_input"
    assert adapter.requests == []


def test_adapter_generate_probe_validates_manifest_before_creating_scratch(
    tmp_path: Path, monkeypatch
) -> None:
    adapter = StagedProbeAdapter({"generate": ProbeReport("generate", True, {})})
    install_staged_adapter(monkeypatch, adapter)
    manifest_path = tmp_path / "invalid.json"
    manifest_path.write_text("{}", encoding="utf-8")
    scratch = tmp_path / "probe"

    result = runner.invoke(
        app,
        [
            "adapter",
            "probe",
            adapter.adapter_id,
            "--phase",
            "generate",
            "--manifest",
            str(manifest_path),
            "--scratch-root",
            str(scratch),
            "--allow-generate",
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert not scratch.exists()
    assert adapter.requests == []


def test_worker_status_human_mode_and_parser_errors_keep_cli_contract(tmp_path: Path) -> None:
    missing_id = runner.invoke(app, [*common(tmp_path), "worker", "status", "--json"])
    bad_option = runner.invoke(app, ["worker", "status", "job", "--unknown", "--json"])
    missing_human = runner.invoke(app, [*common(tmp_path), "worker", "status"])

    for result in (missing_id, bad_option):
        assert result.exit_code == 2
        assert json.loads(result.stdout)["error"]["code"] == "invalid_input"
    assert missing_human.exit_code == 2
    assert missing_human.stdout == ""
    assert "job_id" in missing_human.stderr


def test_run_once_documents_ffprobe_preflight_failure_in_typed_outcome(
    tmp_path: Path, monkeypatch
) -> None:
    create_jobs(tmp_path)
    monkeypatch.setattr("video_lab.worker.shutil.which", lambda _: None)

    result = runner.invoke(
        app, [*common(tmp_path), "worker", "run-once", "--worker-id", "stable-a", "--json"]
    )

    data = assert_compact(result)["data"]
    assert isinstance(data, dict)
    assert data["status"] == "failed"
    assert data["error_code"] == "worker_configuration_error"
    assert data["error_message"] == "ffprobe is required on PATH"

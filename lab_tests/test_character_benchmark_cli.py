from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest
from typer.testing import CliRunner

from video_lab.character_benchmark_models import (
    CharacterBenchmarkPanelInput,
    IdentityChoice,
    SecondaryChoice,
    SecondaryDimension,
)
from video_lab.character_benchmark_protocol import DEFAULT_CHARACTER_CONTEXTS
from video_lab.cli import app

runner = CliRunner()


def _args(tmp_path: Path, *args: str) -> list[str]:
    return [
        "--database",
        str(tmp_path / "lab.sqlite3"),
        "--artifacts",
        str(tmp_path / "artifacts"),
        "--reviewer-id",
        "owner",
        "character-benchmark",
        *args,
    ]


def _payload(result):
    assert result.stdout.count("\n") == 1
    return json.loads(result.stdout)


class FakeBenchmark:
    def __init__(self, tmp_path: Path) -> None:
        self.reference = tmp_path / "reference.png"
        self.left = tmp_path / "left.mp4"
        self.right = tmp_path / "right.mp4"
        for path in (self.reference, self.left, self.right):
            path.write_bytes(path.name.encode())
        self.identity = None
        self.secondary = None

    def create(self, **kwargs):
        return {"campaign_id": "character_benchmark_" + "a" * 64, **kwargs}

    def preview(self, campaign_id):
        return {"campaign_id": campaign_id, "primary_count": 36, "repeat_count": 4}

    def preregister(self, campaign_id):
        return {"campaign_id": campaign_id, "state": "generating"}

    def show(self, campaign_id):
        return {"campaign_id": campaign_id, "state": "judging"}

    def resume(self, campaign_id):
        return SimpleNamespace(
            campaign_id=campaign_id,
            trial_id="character_trial_" + "b" * 64,
            review_id="opaque-review",
            state=SimpleNamespace(value="judging"),
            next_action="record_identity",
            progress=SimpleNamespace(
                model_dump=lambda mode: {
                    "completed_identity": 0,
                    "total_identity": 40,
                    "generated_primary": 36,
                    "total_primary": 36,
                }
            ),
            media=(SimpleNamespace(position="left"), SimpleNamespace(position="right")),
            result=None,
        )

    def cli_resume_projection(self, campaign_id, view):
        return {
            "campaign_id": campaign_id,
            "phase": view.next_action,
            "prompt": "frozen prompt",
            "references": [self.reference.resolve().as_uri()],
            "media": [
                {"position": "left", "media_ref": self.left.resolve().as_uri()},
                {"position": "right", "media_ref": self.right.resolve().as_uri()},
            ],
            "review_id": view.review_id,
            "trial_id": view.trial_id,
            "state": view.state.value,
            "progress": view.progress.model_dump(mode="json"),
        }

    def record_identity(self, campaign_id, review_id, observation):
        self.identity = (campaign_id, review_id, observation)
        return {"campaign_id": campaign_id, "next_action": "record_identity"}

    def record_secondary(self, campaign_id, review_id, observation):
        self.secondary = (campaign_id, review_id, observation)
        return {"campaign_id": campaign_id, "next_action": "record_secondary"}

    def analyze(self, campaign_id):
        return {"campaign_id": campaign_id, "state": "completed"}

    def export_evidence(self, campaign_id):
        return {"schema_version": "character-benchmark.v1", "campaign_id": campaign_id}


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeBenchmark:
    value = FakeBenchmark(tmp_path)
    from video_lab import cli

    monkeypatch.setattr(cli, "_character_benchmark_service_for_id", lambda *_: value)
    monkeypatch.setattr(cli, "_character_benchmark_service", lambda *_: value)
    return value


def test_character_benchmark_create_and_lifecycle_are_json_first(
    tmp_path: Path, fake: FakeBenchmark
) -> None:
    panel = tmp_path / "panel.json"
    panel.write_text(
        CharacterBenchmarkPanelInput(
            contexts=DEFAULT_CHARACTER_CONTEXTS, seeds=(1, 2, 3), schedule_seed=4
        ).model_dump_json()
    )
    commands = [
        (
            "create",
            "--pack",
            "pack",
            "--stage-2-evidence",
            "stage2",
            "--identity-prompt",
            "hero",
            "--panel",
            str(panel),
        ),
        ("preview", "character_benchmark_" + "a" * 64),
        ("preregister", "character_benchmark_" + "a" * 64),
        ("status", "character_benchmark_" + "a" * 64),
        ("analyze", "character_benchmark_" + "a" * 64),
    ]
    for command in commands:
        result = runner.invoke(app, _args(tmp_path, *command))
        assert result.exit_code == 0, result.stdout
        assert _payload(result)["ok"] is True


def test_resume_projects_only_dereferenceable_file_uris(
    tmp_path: Path, fake: FakeBenchmark
) -> None:
    result = runner.invoke(app, _args(tmp_path, "resume", "character_benchmark_" + "a" * 64))
    data = _payload(result)["data"]
    assert result.exit_code == 0
    assert data["prompt"] == "frozen prompt"
    assert data["review_id"] == "opaque-review"
    assert data["phase"] == "record_identity"
    for uri in [*data["references"], *(item["media_ref"] for item in data["media"])]:
        parsed = urlparse(uri)
        assert parsed.scheme == "file"
        assert Path(unquote(parsed.path)).is_file()


def test_identity_neither_and_secondary_are_typed(tmp_path: Path, fake: FakeBenchmark) -> None:
    campaign = "character_benchmark_" + "a" * 64
    trial = "character_trial_" + "b" * 64
    identity = runner.invoke(
        app,
        _args(
            tmp_path,
            "submit-identity",
            campaign,
            "--review-id",
            "opaque-review",
            "--trial-id",
            trial,
            "--choice",
            "neither",
        ),
    )
    assert identity.exit_code == 0
    assert fake.identity[2].choice is IdentityChoice.NEITHER
    secondary = runner.invoke(
        app,
        _args(
            tmp_path,
            "submit-secondary",
            campaign,
            "--review-id",
            "opaque-review",
            "--trial-id",
            trial,
            "--dimension",
            "visual_quality",
            "--choice",
            "tie",
        ),
    )
    assert secondary.exit_code == 0
    assert fake.secondary[2].dimension is SecondaryDimension.VISUAL_QUALITY
    assert fake.secondary[2].choice is SecondaryChoice.TIE


def test_terminal_export_is_write_once(tmp_path: Path, fake: FakeBenchmark) -> None:
    output = tmp_path / "evidence.json"
    args = _args(
        tmp_path, "export-evidence", "character_benchmark_" + "a" * 64, "--output", str(output)
    )
    first = runner.invoke(app, args)
    second = runner.invoke(app, args)
    assert first.exit_code == 0 and output.is_file()
    assert second.exit_code == 2
    assert _payload(second)["error"]["code"] == "invalid_character_benchmark_input"

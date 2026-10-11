from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer
from pydantic import ValidationError
from typer import _click as click
from typer.core import TyperGroup

from video_lab.adapter_protocol import (
    GenerationRequest,
    ProbePhase,
    ProbeRequest,
    StagedProbeAdapter,
)
from video_lab.adapter_registry import AdapterRegistry, UnknownAdapterError
from video_lab.adaptive_study_models import (
    AdaptiveStudyDefinition,
    adaptive_study_input_fingerprint,
)
from video_lab.adaptive_study_repository import (
    AdaptiveStudyConflict,
    AdaptiveStudyNotFound,
    AdaptiveStudyRepository,
)
from video_lab.adaptive_study_service import AdaptiveStudyService
from video_lab.artifacts import ArtifactError, ArtifactStore
from video_lab.capabilities import AdapterCapabilities, validate_manifest_support
from video_lab.character_benchmark_models import (
    CharacterBenchmarkPanelInput,
    IdentityChoice,
    IdentityObservation,
    SecondaryChoice,
    SecondaryDimension,
    SecondaryObservation,
)
from video_lab.character_benchmark_repository import (
    CharacterBenchmarkConflict,
    CharacterBenchmarkNotFound,
    CharacterBenchmarkRepository,
)
from video_lab.character_benchmark_service import CharacterBenchmarkService
from video_lab.character_models import (
    BodyVisibility,
    CharacterAnnotation,
    CuratorDisposition,
    Expression,
    Framing,
    HeadAngle,
    Lighting,
    Occlusion,
    WarningResolution,
)
from video_lab.character_pack import CharacterPackNotReady
from video_lab.character_repository import (
    CharacterConflict,
    CharacterNotFound,
    CharacterRepository,
)
from video_lab.character_service import CharacterImportError, CharacterService
from video_lab.character_web_grant_repository import (
    CHARACTER_CURATION_GRANT_TTL,
    CharacterCurationGrantError,
    CharacterCurationGrantRepository,
)
from video_lab.comparison import ComparisonError, build_controlled_pair
from video_lab.database import Database
from video_lab.domain import (
    DisplayPreference,
    ExperimentDraft,
    FrozenModel,
    Manifest,
    RubricScore,
    utc_now,
)
from video_lab.preference_optimizer import OptimizerConfig
from video_lab.repositories import ExperimentRepository, JobRepository, RecordNotFound
from video_lab.review_service import BlindMedia, BlindReviewView, ReviewService
from video_lab.reviews import ReviewRepository
from video_lab.services import ExperimentService, ExperimentView, ValidationFailure
from video_lab.states import InvalidTransition
from video_lab.validation_models import (
    ValidationChoice,
    ValidationIneligible,
    ValidationPanelInput,
)
from video_lab.validation_repository import (
    ValidationConflict,
    ValidationNotFound,
    ValidationRepository,
)
from video_lab.validation_service import ValidationCampaignService
from video_lab.worker import Worker, WorkerOutcome


class JsonEnvelope(FrozenModel):
    schema_version: str = "cli.v1"
    ok: bool
    data: object | None = None
    error: dict[str, object] | None = None


class JsonTyperGroup(TyperGroup):
    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        raw_args = list(args) if args is not None else sys.argv[1:]
        root_command = self._root_command(raw_args)
        study_command = root_command == "study"
        validation_command = root_command == "validation"
        character_command = root_command == "character"
        character_benchmark_command = root_command == "character-benchmark"
        json_command = (
            study_command or validation_command or character_command or character_benchmark_command
        )
        if (validation_command or character_command or character_benchmark_command) and any(
            argument in {"--help", "-h"} for argument in raw_args
        ):
            command_name = (
                "character-benchmark"
                if character_benchmark_command
                else "character"
                if character_command
                else "validation"
            )
            help_map = (
                _CHARACTER_BENCHMARK_HELP
                if character_benchmark_command
                else _CHARACTER_HELP
                if character_command
                else _VALIDATION_HELP
            )
            subcommand = _json_help_subcommand(raw_args, command_name)
            if subcommand is not None and subcommand not in help_map:
                _emit(
                    JsonEnvelope(
                        ok=False,
                        error={
                            "code": "invalid_input",
                            "message": f"invalid {command_name} command input",
                        },
                    )
                )
                if standalone_mode:
                    raise SystemExit(2)
                raise typer.Exit(2)
            help_data = (
                _character_benchmark_help_data(raw_args)
                if character_benchmark_command
                else _character_help_data(raw_args)
                if character_command
                else _validation_help_data(raw_args)
            )
            _emit(JsonEnvelope(ok=True, data=help_data))
            return None
        if json_command and raw_args[-1] == root_command:
            _emit(
                JsonEnvelope(
                    ok=False,
                    error={
                        "code": "invalid_input",
                        "message": (
                            "adaptive study command is required"
                            if study_command
                            else "character benchmark command is required"
                            if character_benchmark_command
                            else "character command is required"
                            if character_command
                            else "validation command is required"
                        ),
                    },
                )
            )
            if standalone_mode:
                raise SystemExit(2)
            raise typer.Exit(2)
        if "--json" not in raw_args and not json_command:
            return super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        try:
            result = super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                windows_expand_args=windows_expand_args,
                **extra,
            )
            if isinstance(result, int) and result != 0:
                if standalone_mode:
                    raise SystemExit(result)
                raise typer.Exit(result)
            return result
        except click.ClickException as error:
            _emit(
                JsonEnvelope(
                    ok=False,
                    error={
                        "code": (
                            "invalid_validation_input"
                            if validation_command
                            else "invalid_character_benchmark_input"
                            if character_benchmark_command
                            else "invalid_character_input"
                            if character_command
                            else "invalid_input"
                        ),
                        "message": (
                            "invalid adaptive study command input"
                            if study_command
                            else "character benchmark input is invalid"
                            if character_benchmark_command
                            else "character input is invalid"
                            if character_command
                            else "validation input is invalid"
                            if validation_command
                            else error.format_message()
                        ),
                    },
                )
            )
            if standalone_mode:
                raise SystemExit(2) from error
            raise typer.Exit(2) from error
        except typer.Exit as error:
            if standalone_mode:
                raise SystemExit(error.exit_code) from error
            raise

    @staticmethod
    def _root_command(raw_args: Sequence[str]) -> str | None:
        value_options = ("--database", "--artifacts", "--reviewer-id")
        index = 0
        while index < len(raw_args):
            argument = raw_args[index]
            if argument in value_options:
                index += 2
                continue
            if any(argument.startswith(f"{option}=") for option in value_options):
                index += 1
                continue
            if argument.startswith("-"):
                return None
            return argument
        return None


class CliContext(FrozenModel):
    database: Path | None
    artifacts: Path | None
    reviewer_id: str | None


class FeasibilityCommandEnvelope(FrozenModel):
    schema_version: Literal["feasibility-command.v1"] = "feasibility-command.v1"
    ok: bool
    data: object | None = None
    error: dict[str, object] | None = None


class ReviewInput(FrozenModel):
    schema_version: Literal["review.v1"] = "review.v1"
    display_preference: DisplayPreference
    scores: tuple[RubricScore, ...]
    notes: str | None = None
    supersedes_review_id: str | None = None


class InvalidInput(ValueError):
    pass


class InvalidReviewInput(ValueError):
    pass


class InvalidStudyInput(ValueError):
    pass


class InvalidValidationInput(ValueError):
    pass


app = typer.Typer(no_args_is_help=True, cls=JsonTyperGroup)
db_app = typer.Typer(no_args_is_help=True)
experiment_app = typer.Typer(no_args_is_help=True)
worker_app = typer.Typer(no_args_is_help=True)
adapter_app = typer.Typer(no_args_is_help=True)
feasibility_app = typer.Typer(no_args_is_help=True)
review_app = typer.Typer(no_args_is_help=True)
study_app = typer.Typer(no_args_is_help=True)
validation_app = typer.Typer(no_args_is_help=True)
character_app = typer.Typer(no_args_is_help=True)
character_benchmark_app = typer.Typer(no_args_is_help=True)
dataset_app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Strictly local curated dataset operations. Stable exits: 0 accepted; "
        "64 usage; 65 rejected; 66 needs-review; 67 conflict; 68 in-progress; "
        "69 cancelled; 70 integrity/security; 71 internal."
    ),
)
app.add_typer(db_app, name="db")
app.add_typer(experiment_app, name="experiment")
app.add_typer(worker_app, name="worker")
app.add_typer(adapter_app, name="adapter")
adapter_app.add_typer(feasibility_app, name="feasibility")
app.add_typer(review_app, name="review")
app.add_typer(study_app, name="study")
app.add_typer(validation_app, name="validation")
app.add_typer(character_app, name="character")
app.add_typer(character_benchmark_app, name="character-benchmark")
app.add_typer(dataset_app, name="dataset")


def _dataset_roots(context: typer.Context, json_output: bool) -> tuple[Path, Path]:
    from video_lab.dataset_cli import emit_usage_error

    options = _context(context)
    if options.database is None or options.artifacts is None:
        emit_usage_error(
            json_output=json_output, message="database and artifact roots are required"
        )
        raise AssertionError("unreachable")
    return options.database, options.artifacts


@dataset_app.command("validate")
def dataset_validate(
    context: typer.Context,
    request_root: Annotated[Path, typer.Option("--request-root")],
    request_file: Annotated[str, typer.Option("--request-file")],
    source_root: Annotated[Path | None, typer.Option("--source-root")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    from video_lab.dataset_cli import run_command

    database, artifacts = _dataset_roots(context, json_output)
    run_command(
        command="validate",
        database_path=database,
        artifact_root=artifacts,
        request_root=request_root,
        request_file=request_file,
        source_root=source_root,
        json_output=json_output,
    )


def _run_dataset_compilation_command(
    command: Literal["compile", "reproduce"],
    context: typer.Context,
    request_root: Path,
    request_file: str,
    json_output: bool,
    tokenizer_root: Path | None,
    text_encoder_root: Path | None,
    vae_root: Path | None,
    scheduler_root: Path | None,
) -> None:
    from video_lab.dataset_cli import run_command

    database, artifacts = _dataset_roots(context, json_output)
    values = (tokenizer_root, text_encoder_root, vae_root, scheduler_root)
    run_command(
        command=command,
        database_path=database,
        artifact_root=artifacts,
        request_root=request_root,
        request_file=request_file,
        json_output=json_output,
        component_roots=values if any(value is not None for value in values) else None,
    )


@dataset_app.command("compile")
def dataset_compile(
    context: typer.Context,
    request_root: Annotated[Path, typer.Option("--request-root")],
    request_file: Annotated[str, typer.Option("--request-file")],
    tokenizer_root: Annotated[Path | None, typer.Option("--tokenizer-root")] = None,
    text_encoder_root: Annotated[Path | None, typer.Option("--text-encoder-root")] = None,
    vae_root: Annotated[Path | None, typer.Option("--vae-root")] = None,
    scheduler_root: Annotated[Path | None, typer.Option("--scheduler-root")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    _run_dataset_compilation_command(
        "compile",
        context,
        request_root,
        request_file,
        json_output,
        tokenizer_root,
        text_encoder_root,
        vae_root,
        scheduler_root,
    )


@dataset_app.command("inspect")
def dataset_inspect(
    context: typer.Context,
    request_root: Annotated[Path, typer.Option("--request-root")],
    request_file: Annotated[str, typer.Option("--request-file")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    from video_lab.dataset_cli import run_command

    database, artifacts = _dataset_roots(context, json_output)
    run_command(
        command="inspect",
        database_path=database,
        artifact_root=artifacts,
        request_root=request_root,
        request_file=request_file,
        json_output=json_output,
    )


@dataset_app.command("reproduce")
def dataset_reproduce(
    context: typer.Context,
    request_root: Annotated[Path, typer.Option("--request-root")],
    request_file: Annotated[str, typer.Option("--request-file")],
    tokenizer_root: Annotated[Path | None, typer.Option("--tokenizer-root")] = None,
    text_encoder_root: Annotated[Path | None, typer.Option("--text-encoder-root")] = None,
    vae_root: Annotated[Path | None, typer.Option("--vae-root")] = None,
    scheduler_root: Annotated[Path | None, typer.Option("--scheduler-root")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    _run_dataset_compilation_command(
        "reproduce",
        context,
        request_root,
        request_file,
        json_output,
        tokenizer_root,
        text_encoder_root,
        vae_root,
        scheduler_root,
    )


_CHARACTER_BENCHMARK_HELP: Mapping[str, dict[str, object]] = {
    "create": {
        "usage": (
            "create --pack PACK_ID --stage-2-evidence CAMPAIGN_ID "
            "--identity-prompt TEXT --panel PATH"
        ),
        "arguments": [],
        "options": ["--pack", "--stage-2-evidence", "--identity-prompt", "--panel"],
        "choices": {},
    },
    "preview": {
        "usage": "preview CAMPAIGN_ID",
        "arguments": ["campaign_id"],
        "options": [],
        "choices": {},
    },
    "preregister": {
        "usage": "preregister CAMPAIGN_ID",
        "arguments": ["campaign_id"],
        "options": [],
        "choices": {},
    },
    "status": {
        "usage": "status CAMPAIGN_ID",
        "arguments": ["campaign_id"],
        "options": [],
        "choices": {},
    },
    "resume": {
        "usage": "resume CAMPAIGN_ID",
        "arguments": ["campaign_id"],
        "options": [],
        "choices": {},
    },
    "submit-identity": {
        "usage": (
            "submit-identity CAMPAIGN_ID --review-id REVIEW_ID "
            "--trial-id TRIAL_ID --choice {a|b|both|neither}"
        ),
        "arguments": ["campaign_id"],
        "options": ["--review-id", "--trial-id", "--choice"],
        "choices": {"--choice": ["a", "b", "both", "neither"]},
    },
    "submit-secondary": {
        "usage": (
            "submit-secondary CAMPAIGN_ID --review-id REVIEW_ID "
            "--trial-id TRIAL_ID --dimension DIMENSION --choice {a|b|tie}"
        ),
        "arguments": ["campaign_id"],
        "options": ["--review-id", "--trial-id", "--dimension", "--choice"],
        "choices": {
            "--dimension": [item.value for item in SecondaryDimension],
            "--choice": ["a", "b", "tie"],
        },
    },
    "analyze": {
        "usage": "analyze CAMPAIGN_ID",
        "arguments": ["campaign_id"],
        "options": [],
        "choices": {},
    },
    "export-evidence": {
        "usage": "export-evidence CAMPAIGN_ID --output PATH",
        "arguments": ["campaign_id"],
        "options": ["--output"],
        "choices": {},
    },
}
_CHARACTER_BENCHMARK_GLOBAL_USAGE = (
    "video-lab --database PATH --artifacts PATH --reviewer-id TEXT character-benchmark"
)


_CHARACTER_HELP: Mapping[str, dict[str, object]] = {
    "create": {
        "usage": "create NAME",
        "arguments": ["name"],
        "options": [],
        "choices": {},
    },
    "import": {
        "usage": "import CHARACTER_ID --source SOURCE",
        "arguments": ["character_id"],
        "options": ["--source"],
        "choices": {},
    },
    "status": {
        "usage": "status PACK_ID",
        "arguments": ["pack_id"],
        "options": [],
        "choices": {},
    },
    "curate": {
        "usage": (
            "curate PACK_ID REFERENCE_ID --framing FRAMING --head-angle HEAD_ANGLE "
            "--body-visibility BODY_VISIBILITY --expression EXPRESSION --lighting LIGHTING "
            "--occlusion OCCLUSION --identity-confirmed|--identity-not-confirmed "
            "--accept|--reject"
        ),
        "arguments": ["pack_id", "reference_id"],
        "options": [
            "--framing",
            "--head-angle",
            "--body-visibility",
            "--expression",
            "--lighting",
            "--occlusion",
            "--identity-confirmed|--identity-not-confirmed",
            "--accept|--reject",
        ],
        "choices": {
            "--framing": [item.value for item in Framing],
            "--head-angle": [item.value for item in HeadAngle],
            "--body-visibility": [item.value for item in BodyVisibility],
            "--expression": [item.value for item in Expression],
            "--lighting": [item.value for item in Lighting],
            "--occlusion": [item.value for item in Occlusion],
        },
    },
    "resolve-warning": {
        "usage": (
            "resolve-warning PACK_ID LEFT_REFERENCE_ID RIGHT_REFERENCE_ID "
            "--resolution {accepted|dismissed}"
        ),
        "arguments": ["pack_id", "left_reference_id", "right_reference_id"],
        "options": ["--resolution"],
        "choices": {"--resolution": ["accepted", "dismissed"]},
    },
    "preregister": {
        "usage": "preregister PACK_ID",
        "arguments": ["pack_id"],
        "options": [],
        "choices": {},
    },
    "export-pack": {
        "usage": "export-pack PACK_ID --output OUTPUT_PATH",
        "arguments": ["pack_id"],
        "options": ["--output"],
        "choices": {},
    },
    "open": {
        "usage": "open CHARACTER_ID --pack PACK_ID",
        "arguments": ["character_id"],
        "options": ["--pack"],
        "choices": {},
    },
}
_CHARACTER_GLOBAL_OPTIONS: list[dict[str, object]] = [
    {"option": "--database", "value": "PATH", "required": True},
    {"option": "--artifacts", "value": "PATH", "required": True},
    {"option": "--reviewer-id", "value": "TEXT", "required": True},
]
_CHARACTER_GLOBAL_USAGE = "video-lab --database PATH --artifacts PATH --reviewer-id TEXT character"


_VALIDATION_HELP: Mapping[str, dict[str, object]] = {
    "create": {
        "usage": "create --study STUDY_ID --panel PANEL_PATH",
        "arguments": [],
        "options": ["--study", "--panel"],
        "choices": {},
    },
    "preview": {
        "usage": "preview CAMPAIGN_ID",
        "arguments": ["campaign_id"],
        "options": [],
        "choices": {},
    },
    "preregister": {
        "usage": "preregister CAMPAIGN_ID",
        "arguments": ["campaign_id"],
        "options": [],
        "choices": {},
    },
    "status": {
        "usage": "status CAMPAIGN_ID",
        "arguments": ["campaign_id"],
        "options": [],
        "choices": {},
    },
    "resume": {
        "usage": "resume CAMPAIGN_ID",
        "arguments": ["campaign_id"],
        "options": [],
        "choices": {},
    },
    "submit": {
        "usage": "submit CAMPAIGN_ID --review-id REVIEW_ID --choice {a|b|tie}",
        "arguments": ["campaign_id"],
        "options": ["--review-id", "--choice"],
        "choices": {"--choice": ["a", "b", "tie"]},
    },
    "analyze": {
        "usage": "analyze CAMPAIGN_ID",
        "arguments": ["campaign_id"],
        "options": [],
        "choices": {},
    },
    "export-evidence": {
        "usage": "export-evidence CAMPAIGN_ID --output OUTPUT_PATH",
        "arguments": ["campaign_id"],
        "options": ["--output"],
        "choices": {},
    },
}
_VALIDATION_GLOBAL_OPTIONS: list[dict[str, object]] = [
    {"option": "--database", "value": "PATH", "required": True},
    {"option": "--artifacts", "value": "PATH", "required": True},
    {"option": "--reviewer-id", "value": "TEXT", "required": True},
]
_VALIDATION_GLOBAL_USAGE = (
    "video-lab --database PATH --artifacts PATH --reviewer-id TEXT validation"
)


def _json_help_subcommand(raw_args: Sequence[str], command_name: str) -> str | None:
    command_index = raw_args.index(command_name)
    return next(
        (argument for argument in raw_args[command_index + 1 :] if not argument.startswith("-")),
        None,
    )


def _validation_help_data(raw_args: Sequence[str]) -> dict[str, object]:
    command = _json_help_subcommand(raw_args, "validation")
    if command is None:
        return {
            "schema_version": "validation-cli-help.v1",
            "command": "validation",
            "usage": f"{_VALIDATION_GLOBAL_USAGE} <command>",
            "global_options": _VALIDATION_GLOBAL_OPTIONS,
            "required_arguments": [],
            "required_options": [],
            "choices": {},
            "commands": list(_VALIDATION_HELP),
        }
    details = _VALIDATION_HELP[command]
    return {
        "schema_version": "validation-cli-help.v1",
        "command": f"validation {command}",
        "usage": f"{_VALIDATION_GLOBAL_USAGE} {details['usage']}",
        "global_options": _VALIDATION_GLOBAL_OPTIONS,
        "required_arguments": details["arguments"],
        "required_options": details["options"],
        "choices": details["choices"],
    }


def _character_help_data(raw_args: Sequence[str]) -> dict[str, object]:
    command = _json_help_subcommand(raw_args, "character")
    if command is None:
        return {
            "schema_version": "character-cli-help.v1",
            "command": "character",
            "usage": f"{_CHARACTER_GLOBAL_USAGE} <command>",
            "global_options": _CHARACTER_GLOBAL_OPTIONS,
            "required_arguments": [],
            "required_options": [],
            "choices": {},
            "commands": list(_CHARACTER_HELP),
        }
    details = _CHARACTER_HELP[command]
    return {
        "schema_version": "character-cli-help.v1",
        "command": f"character {command}",
        "usage": f"{_CHARACTER_GLOBAL_USAGE} {details['usage']}",
        "global_options": _CHARACTER_GLOBAL_OPTIONS,
        "required_arguments": details["arguments"],
        "required_options": details["options"],
        "choices": details["choices"],
    }


def _character_benchmark_help_data(raw_args: Sequence[str]) -> dict[str, object]:
    command = _json_help_subcommand(raw_args, "character-benchmark")
    if command is None:
        return {
            "schema_version": "character-benchmark-cli-help.v1",
            "command": "character-benchmark",
            "usage": f"{_CHARACTER_BENCHMARK_GLOBAL_USAGE} <command>",
            "global_options": _CHARACTER_GLOBAL_OPTIONS,
            "required_arguments": [],
            "required_options": [],
            "choices": {},
            "commands": list(_CHARACTER_BENCHMARK_HELP),
        }
    details = _CHARACTER_BENCHMARK_HELP[command]
    return {
        "schema_version": "character-benchmark-cli-help.v1",
        "command": f"character-benchmark {command}",
        "usage": f"{_CHARACTER_BENCHMARK_GLOBAL_USAGE} {details['usage']}",
        "global_options": _CHARACTER_GLOBAL_OPTIONS,
        "required_arguments": details["arguments"],
        "required_options": details["options"],
        "choices": details["choices"],
    }


@app.callback()
def configure(
    context: typer.Context,
    database: Annotated[Path | None, typer.Option("--database")] = None,
    artifacts: Annotated[Path | None, typer.Option("--artifacts")] = None,
    reviewer_id: Annotated[
        str | None, typer.Option("--reviewer-id", envvar="VIDEO_LAB_REVIEWER_ID")
    ] = None,
) -> None:
    context.obj = CliContext(database=database, artifacts=artifacts, reviewer_id=reviewer_id)


def _context(context: typer.Context) -> CliContext:
    value = context.find_root().obj
    if not isinstance(value, CliContext):
        raise RuntimeError("CLI paths were not configured")
    return value


def _service(context: typer.Context) -> ExperimentService:
    options = _context(context)
    database, artifacts = _required_paths(
        (options.database, "database"), (options.artifacts, "artifacts")
    )
    return ExperimentService.from_paths(database, artifacts)


def _review_service(context: typer.Context) -> ReviewService:
    options = _context(context)
    database_path, artifact_root = _required_paths(
        (options.database, "database"), (options.artifacts, "artifacts")
    )
    database = Database(database_path)
    database.migrate()
    return ReviewService(
        ExperimentRepository(database),
        JobRepository(database),
        ArtifactStore(database, artifact_root),
        ReviewRepository(database),
    )


def _reviewer_id(context: typer.Context) -> str:
    try:
        return _required_nonblank_text(_context(context).reviewer_id, "reviewer_id").strip()
    except InvalidInput as error:
        raise InvalidReviewInput(str(error)) from error


def _study_reviewer_id(context: typer.Context) -> str:
    try:
        return _required_nonblank_text(_context(context).reviewer_id, "reviewer_id").strip()
    except InvalidInput as error:
        raise InvalidStudyInput(str(error)) from error


def _adaptive_study_service(
    context: typer.Context,
    capabilities: AdapterCapabilities,
) -> AdaptiveStudyService:
    options = _context(context)
    database_path, artifact_root = _required_paths(
        (options.database, "database"), (options.artifacts, "artifacts")
    )
    database = Database(database_path)
    database.migrate()
    artifacts = ArtifactStore(database, artifact_root)
    return AdaptiveStudyService(
        AdaptiveStudyRepository(database),
        ExperimentService(
            ExperimentRepository(database),
            JobRepository(database),
            artifacts,
        ),
        ReviewService(
            ExperimentRepository(database),
            JobRepository(database),
            artifacts,
            ReviewRepository(database),
        ),
        capabilities,
        reviewer_id=_study_reviewer_id(context),
    )


def _adaptive_study_service_for_id(context: typer.Context, study_id: str) -> AdaptiveStudyService:
    database_path = _required_paths((_context(context).database, "database"))[0]
    database = Database(database_path)
    database.migrate()
    definition = AdaptiveStudyRepository(database).get(study_id)
    capabilities = AdapterRegistry.with_defaults().capabilities(definition.base_manifest.adapter_id)
    return _adaptive_study_service(context, capabilities)


def _validation_reviewer_id(context: typer.Context) -> str:
    try:
        return _required_nonblank_text(_context(context).reviewer_id, "reviewer_id").strip()
    except InvalidInput as error:
        raise InvalidValidationInput(str(error)) from error


def _character_service(context: typer.Context) -> CharacterService:
    options = _context(context)
    database_path, artifact_root = _required_paths(
        (options.database, "database"), (options.artifacts, "artifacts")
    )
    try:
        reviewer = _required_nonblank_text(options.reviewer_id, "reviewer_id").strip()
    except InvalidInput as error:
        raise ValueError("character reviewer is required") from error
    database = Database(database_path)
    database.migrate()
    return CharacterService(
        CharacterRepository(database, reviewer_id=reviewer),
        ArtifactStore(database, artifact_root),
    )


def _character_grants(
    context: typer.Context,
) -> tuple[CharacterCurationGrantRepository, str]:
    options = _context(context)
    database_path, _artifact_root = _required_paths(
        (options.database, "database"), (options.artifacts, "artifacts")
    )
    reviewer = _required_nonblank_text(options.reviewer_id, "reviewer_id").strip()
    database = Database(database_path)
    database.migrate()
    return CharacterCurationGrantRepository(database), reviewer


def _validation_campaign_service_for_study(
    context: typer.Context, study_id: str
) -> ValidationCampaignService:
    options = _context(context)
    database_path, artifact_root = _required_paths(
        (options.database, "database"), (options.artifacts, "artifacts")
    )
    database = Database(database_path)
    database.migrate()
    definition = AdaptiveStudyRepository(database).get(study_id)
    capabilities = AdapterRegistry.with_defaults().capabilities(definition.base_manifest.adapter_id)
    stage_1 = _adaptive_study_service(context, capabilities)
    artifacts = ArtifactStore(database, artifact_root)
    experiments = ExperimentService(
        ExperimentRepository(database), JobRepository(database), artifacts
    )
    reviews = ReviewService(
        ExperimentRepository(database),
        JobRepository(database),
        artifacts,
        ReviewRepository(database),
    )
    return ValidationCampaignService(
        ValidationRepository(database),
        stage_1,
        experiments,
        reviews,
        AdapterRegistry.with_defaults(),
        reviewer_id=_validation_reviewer_id(context),
    )


def _validation_campaign_service_for_id(
    context: typer.Context, campaign_id: str
) -> ValidationCampaignService:
    database_path = _required_paths((_context(context).database, "database"))[0]
    database = Database(database_path)
    database.migrate()
    definition = ValidationRepository(database).get(campaign_id)
    return _validation_campaign_service_for_study(context, definition.stage_1_study_id)


def _character_benchmark_service(
    context: typer.Context, stage_2_campaign_id: str
) -> CharacterBenchmarkService:
    options = _context(context)
    database_path, artifact_root = _required_paths(
        (options.database, "database"), (options.artifacts, "artifacts")
    )
    reviewer = _required_nonblank_text(options.reviewer_id, "reviewer_id").strip()
    database = Database(database_path)
    database.migrate()
    artifacts = ArtifactStore(database, artifact_root)
    experiments = ExperimentService(
        ExperimentRepository(database), JobRepository(database), artifacts
    )
    reviews = ReviewService(
        ExperimentRepository(database),
        JobRepository(database),
        artifacts,
        ReviewRepository(database),
    )
    return CharacterBenchmarkService(
        CharacterBenchmarkRepository(database, reviewer_id=reviewer),
        CharacterService(CharacterRepository(database, reviewer_id=reviewer), artifacts),
        _validation_campaign_service_for_id(context, stage_2_campaign_id),
        experiments,
        reviews,
        AdapterRegistry.with_defaults(),
        reviewer_id=reviewer,
    )


def _character_benchmark_service_for_id(
    context: typer.Context, campaign_id: str
) -> CharacterBenchmarkService:
    database_path = _required_paths((_context(context).database, "database"))[0]
    reviewer = _required_nonblank_text(_context(context).reviewer_id, "reviewer_id").strip()
    database = Database(database_path)
    database.migrate()
    campaign = CharacterBenchmarkRepository(database, reviewer_id=reviewer).get(campaign_id)
    return _character_benchmark_service(context, campaign.stage_2_campaign_id)


def _projection_reviewer_id(context: typer.Context) -> str | None:
    from video_lab.feasibility_projection import normalize_reviewer_id

    return normalize_reviewer_id(_context(context).reviewer_id)


def _required_paths(*values: tuple[Path | None, str]) -> tuple[Path, ...]:
    missing = [name for value, name in values if value is None]
    if missing:
        raise InvalidInput(f"missing required value: {', '.join(missing)}")
    return tuple(value for value, _ in values if value is not None)


def _required_text(value: str | None, name: str) -> str:
    if value is None:
        raise InvalidInput(f"missing required value: {name}")
    return value


def _required_nonblank_text(value: str | None, name: str) -> str:
    text = _required_text(value, name)
    if not text.strip():
        raise InvalidInput(f"{name} must not be blank")
    return text


def _json_data(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_json_data(item) for item in value]
    if isinstance(value, list):
        return [_json_data(item) for item in value]
    return value


def _emit(envelope: JsonEnvelope) -> None:
    typer.echo(envelope.model_dump_json())


def _emit_feasibility(envelope: FeasibilityCommandEnvelope) -> None:
    typer.echo(envelope.model_dump_json())


def _fail(code: str, message: str, exit_code: int, details: object | None = None) -> None:
    error: dict[str, object] = {"code": code, "message": message}
    if details is not None:
        error["details"] = _json_data(details)
    _emit(JsonEnvelope(ok=False, error=error))
    raise typer.Exit(exit_code)


def _run_json(action: Callable[[], object]) -> None:
    try:
        _emit(JsonEnvelope(ok=True, data=_json_data(action())))
    except ValidationFailure as error:
        _fail("manifest_unsupported", str(error), 2, error.violations)
    except (
        InvalidInput,
        InvalidTransition,
        json.JSONDecodeError,
        ValidationError,
        ComparisonError,
    ) as error:
        _fail("invalid_input", str(error), 2)
    except UnknownAdapterError as error:
        _fail("adapter_not_found", str(error), 3)
    except RecordNotFound as error:
        _fail("record_not_found", str(error), 3)
    except (ArtifactError, sqlite3.Error, OSError) as error:
        _fail("persistence_failure", str(error), 4)
    except Exception as error:
        typer.echo(f"internal error: {error}", err=True)
        _fail("internal_error", "unexpected internal error", 1)


def _run_human(action: Callable[[], object], render: Callable[[object], str]) -> None:
    try:
        typer.echo(render(action()))
    except (
        ValidationFailure,
        InvalidInput,
        InvalidTransition,
        json.JSONDecodeError,
        ValidationError,
        ComparisonError,
    ) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    except RecordNotFound as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(3) from error
    except UnknownAdapterError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(3) from error
    except (ArtifactError, sqlite3.Error, OSError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(4) from error
    except Exception as error:
        typer.echo(f"internal error: {error}", err=True)
        raise typer.Exit(1) from error


def _run_character(action: Callable[[], object]) -> None:
    try:
        _emit(JsonEnvelope(ok=True, data=_json_data(action())))
    except CharacterNotFound:
        _fail("character_not_found", "character evidence was not found", 2)
    except CharacterCurationGrantError:
        _fail("character_not_found", "character evidence was not found", 2)
    except CharacterConflict:
        _fail("character_not_found", "character evidence was not found", 2)
    except CharacterPackNotReady as error:
        _fail(
            "character_not_ready",
            "character pack coverage is incomplete",
            2,
            {"missing": list(error.reason_codes)},
        )
    except PermissionError:
        _fail("character_conflict", "character action conflicts with frozen evidence", 2)
    except (CharacterImportError, InvalidInput, ValidationError, ValueError):
        _fail("invalid_character_input", "character input is invalid", 2)
    except (ArtifactError, sqlite3.Error, OSError):
        _fail("character_persistence_failure", "character persistence failed", 4)
    except Exception:
        _fail("internal_error", "unexpected internal error", 1)


def _run_review_json(action: Callable[[], object]) -> None:
    try:
        _emit(JsonEnvelope(ok=True, data=_json_data(action())))
    except InvalidReviewInput:
        _fail("invalid_review_input", "invalid review input", 2)
    except PermissionError as error:
        code = (
            "review_reveal_forbidden"
            if "submitted before reveal" in str(error)
            else "review_not_owned"
        )
        _fail(code, "review operation is not permitted", 2)
    except (RecordNotFound, LookupError):
        _fail("review_not_found", "review record not found", 3)
    except ValueError:
        _fail("invalid_review_state", "review operation is invalid", 2)
    except (ArtifactError, sqlite3.Error, OSError):
        _fail("review_persistence_failure", "review persistence failed", 4)
    except Exception as error:
        typer.echo(f"internal error: {type(error).__name__}", err=True)
        _fail("internal_error", "unexpected internal error", 1)


def _run_review_human(action: Callable[[], object]) -> None:
    try:
        typer.echo(json.dumps(_json_data(action()), indent=2))
    except InvalidReviewInput as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    except PermissionError as error:
        typer.echo("review operation is not permitted", err=True)
        raise typer.Exit(2) from error
    except (RecordNotFound, LookupError) as error:
        typer.echo("review record not found", err=True)
        raise typer.Exit(3) from error
    except ValueError as error:
        typer.echo("review operation is invalid", err=True)
        raise typer.Exit(2) from error
    except (ArtifactError, sqlite3.Error, OSError) as error:
        typer.echo("review persistence failed", err=True)
        raise typer.Exit(4) from error


def _run_study_json(action: Callable[[], object]) -> None:
    try:
        _emit(JsonEnvelope(ok=True, data=_json_data(action())))
    except InvalidStudyInput:
        _fail("invalid_study_input", "adaptive study input is invalid", 2)
    except InvalidInput:
        _fail("invalid_study_input", "adaptive study input is invalid", 2)
    except ValidationFailure:
        _fail("manifest_unsupported", "manifest is unsupported by the selected adapter", 2)
    except ValidationError:
        _fail("invalid_study_input", "adaptive study input is invalid", 2)
    except UnknownAdapterError:
        _fail("adapter_not_found", "adaptive study adapter was not found", 3)
    except AdaptiveStudyNotFound:
        _fail("study_not_found", "adaptive study was not found", 3)
    except AdaptiveStudyConflict:
        _fail("study_conflict", "adaptive study evidence conflicts with the requested action", 2)
    except PermissionError:
        _fail("study_review_not_owned", "adaptive review operation is not permitted", 2)
    except LookupError:
        _fail("study_review_not_found", "adaptive review was not found", 3)
    except ValueError:
        _fail("invalid_study_state", "adaptive study action is invalid in its current state", 2)
    except (ArtifactError, sqlite3.Error, OSError):
        _fail("study_persistence_failure", "adaptive study persistence failed", 4)
    except Exception:
        _fail("internal_error", "unexpected internal error", 1)


def _run_validation_json(action: Callable[[], object]) -> None:
    try:
        _emit(JsonEnvelope(ok=True, data=_json_data(action())))
    except (InvalidValidationInput, InvalidInput, ValidationError, json.JSONDecodeError):
        _fail("invalid_validation_input", "validation input is invalid", 2)
    except ValidationIneligible:
        _fail("validation_ineligible", "validation evidence is not eligible", 2)
    except ValidationNotFound:
        _fail("validation_not_found", "validation campaign was not found", 2)
    except ValidationConflict:
        _fail("validation_conflict", "validation action conflicts with sealed evidence", 2)
    except PermissionError as error:
        if "sealed after preregistration" in str(error):
            _fail(
                "validation_preview_sealed",
                "validation preview is sealed after preregistration",
                2,
            )
        _fail("validation_not_owned", "validation operation is not permitted", 2)
    except (ArtifactError, sqlite3.Error, OSError):
        _fail("validation_persistence_failure", "validation persistence failed", 4)
    except (UnknownAdapterError, AdaptiveStudyNotFound):
        _fail("validation_ineligible", "validation evidence is not eligible", 2)
    except ValueError:
        _fail("invalid_validation_state", "validation action is invalid in its current state", 2)
    except Exception:
        _fail("internal_error", "unexpected internal error", 1)


def _run_character_benchmark_json(action: Callable[[], object]) -> None:
    try:
        _emit(JsonEnvelope(ok=True, data=_json_data(action())))
    except (InvalidInput, ValidationError, json.JSONDecodeError):
        _fail("invalid_character_benchmark_input", "character benchmark input is invalid", 2)
    except CharacterBenchmarkNotFound:
        _fail("character_benchmark_not_found", "character benchmark was not found", 2)
    except CharacterBenchmarkConflict as error:
        message = str(error).lower()
        if "another reviewer" in message or "belongs to another" in message:
            _fail("character_benchmark_not_found", "character benchmark was not found", 2)
        elif "character pack" in message:
            code = "character_benchmark_incomplete_curation"
        elif "stage 2" in message:
            code = "character_benchmark_stage_2_ineligible"
        elif "cannot enter analysis" in message or "not eligible" in message:
            code = "character_benchmark_invalid_phase"
        elif any(
            marker in message
            for marker in (
                "inconsistent",
                "canonical",
                "corrupt",
                "digest",
                "immutable",
                "ownership",
            )
        ):
            code = "character_benchmark_corrupt_provenance"
        elif "review" in message:
            code = "character_benchmark_stale_review"
        else:
            code = "character_benchmark_conflict"
        _fail(code, "character benchmark action conflicts with sealed evidence", 2)
    except PermissionError as error:
        code = (
            "character_benchmark_preview_sealed"
            if "sealed after preregistration" in str(error)
            else "character_benchmark_not_owned"
        )
        _fail(code, "character benchmark operation is not permitted", 2)
    except ArtifactError:
        _fail(
            "character_benchmark_corrupt_provenance",
            "character benchmark action conflicts with sealed evidence",
            2,
        )
    except (sqlite3.Error, OSError):
        _fail(
            "character_benchmark_persistence_failure", "character benchmark persistence failed", 4
        )
    except (UnknownAdapterError, ValidationNotFound, CharacterNotFound):
        _fail("character_benchmark_ineligible", "character benchmark evidence is not eligible", 2)
    except ValueError:
        _fail("invalid_character_benchmark_state", "character benchmark action is invalid", 2)
    except Exception:
        _fail("internal_error", "unexpected internal error", 1)


def _feasibility_service(
    context: typer.Context,
    cache: Path | None = None,
    *,
    migrate: bool = True,
    run_id: str | None = None,
    inventory_evidence: Mapping[str, object] | None = None,
) -> object:
    from video_lab.animegen_feasibility import (
        AnimeGenFeasibilityExecutor,
        production_component_options,
        production_generation_runtime,
        production_inventory_client,
    )
    from video_lab.artifacts import ArtifactStore
    from video_lab.database import Database
    from video_lab.feasibility import FeasibilityService, build_generation_feasibility_service
    from video_lab.feasibility_repository import FeasibilityRepository

    database_path = _required_paths((_context(context).database, "database"))[0]
    database = Database(database_path, read_only=not migrate)
    if migrate:
        database.migrate()

    safe_cache = None if cache is None else cache.expanduser().resolve()
    inventory_client = None if safe_cache is None else production_inventory_client(safe_cache)
    executor = AnimeGenFeasibilityExecutor(
        safe_cache,
        inventory_client,
        component_options=production_component_options(),
        inventory_evidence=inventory_evidence,
    )
    options = _context(context)
    artifact_store = (
        None if options.artifacts is None else ArtifactStore(database, options.artifacts)
    )
    repository = FeasibilityRepository(database, artifact_store)
    delegate = FeasibilityService(repository, executor)
    if options.artifacts is None:
        return delegate
    if run_id is None or inventory_client is None:
        return delegate
    assert safe_cache is not None
    from video_lab.adapters.animegen_mps import (
        ADAPTER_ID,
        AnimeGenGenerationRuntime,
        AnimeGenMpsAdapter,
    )
    from video_lab.feasibility import StageExecutionContext
    from video_lab.feasibility_models import FeasibilityStage

    run = repository.get_run(run_id)
    attempts = repository.list_attempts(run_id)
    generation_stages = {
        FeasibilityStage.GENERATE_MINIMUM_1.value,
        FeasibilityStage.GENERATE_MINIMUM_2.value,
        FeasibilityStage.QUALIFY_STANDARD.value,
        FeasibilityStage.VALIDATE_CONTROLLED_PAIR.value,
    }
    if _next_feasibility_stage(delegate.status(run_id)) not in generation_stages:
        return delegate
    inventory = executor.validated_download_inventory(StageExecutionContext(run, attempts))
    sources = {source.component: source for source in inventory.sources}
    runtime = production_generation_runtime(safe_cache, sources)
    candidate = AnimeGenMpsAdapter(
        runtime=cast(AnimeGenGenerationRuntime, runtime),
        authorized_profile=(512, 288, 17),
        exact_profile_only=False,
    )
    feasibility_registry = AdapterRegistry()
    feasibility_registry.register(
        ADAPTER_ID, lambda: candidate, capabilities_factory=candidate.capabilities
    )
    service = ExperimentService.from_paths(database_path, options.artifacts)
    worker = Worker(
        repository=service.jobs,
        artifact_store=service.artifacts,
        adapter_registry=feasibility_registry,
        scratch_root=options.artifacts / ".scratch",
        worker_id="feasibility-cli",
    )
    return build_generation_feasibility_service(
        repository=repository,
        delegate=executor,
        experiments=service.experiments,
        jobs=service.jobs,
        artifacts=service.artifacts,
        worker=worker,
    )


def _next_feasibility_stage(status: Any) -> str | None:
    from video_lab.feasibility_projection import next_feasibility_stage

    return next_feasibility_stage(status.attempts)


def _feasibility_data(
    status: Any,
    repository: Any,
    *,
    reviewer_id: str | None = None,
    completed: Any | None = None,
) -> dict[str, object]:
    from video_lab.feasibility_projection import FeasibilityEvidenceProjector

    return FeasibilityEvidenceProjector(repository).project(
        status,
        reviewer_id=reviewer_id,
        completed_attempt=completed,
    )


def _run_feasibility(action: Callable[[], object], *, json_output: bool) -> None:
    from video_lab.feasibility_repository import FeasibilityRunNotFound
    from video_lab.source_inventory import SourceInventoryError

    try:
        value = action()
        if json_output:
            _emit_feasibility(FeasibilityCommandEnvelope(ok=True, data=value))
        else:
            data = value if isinstance(value, dict) else {}
            run = data.get("run", {})
            typer.echo(
                f"{run.get('adapter_id', 'feasibility')} {run.get('id', '')}: "
                f"{data.get('verdict', 'pending')}"
            )
    except UnknownAdapterError as error:
        _fail_feasibility("adapter_not_found", str(error), 3, json_output)
    except FeasibilityRunNotFound as error:
        _fail_feasibility("feasibility_run_not_found", str(error), 3, json_output)
    except (
        InvalidInput,
        ValidationError,
        json.JSONDecodeError,
        SourceInventoryError,
        ValueError,
    ) as error:
        code = "cache_required" if "cache is required" in str(error) else "invalid_input"
        _fail_feasibility(code, str(error), 2, json_output)
    except (sqlite3.Error, OSError) as error:
        _fail_feasibility("persistence_failure", str(error), 4, json_output)
    except Exception as error:
        typer.echo(f"internal error: {type(error).__name__}", err=True)
        _fail_feasibility("internal_error", "unexpected internal error", 1, json_output)


def _fail_feasibility(code: str, message: str, exit_code: int, json_output: bool) -> None:
    from video_lab.feasibility import sanitize_diagnostic

    safe_message = str(sanitize_diagnostic(message))
    if json_output:
        _emit_feasibility(
            FeasibilityCommandEnvelope(ok=False, error={"code": code, "message": safe_message})
        )
    else:
        typer.echo(safe_message, err=True)
    raise typer.Exit(exit_code)


def _review_input(path: Path) -> ReviewInput:
    try:
        return _load(path, ReviewInput)
    except (InvalidInput, ValidationError, json.JSONDecodeError) as error:
        raise InvalidReviewInput("invalid review input") from error


def _review_data(key: str, value: object) -> dict[str, object]:
    return {"schema_version": "review.v1", key: _json_data(value)}


def _load[ModelT: FrozenModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        document = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise InvalidInput(f"cannot read input document {path}: {error}") from error
    return model.model_validate_json(document)


def _view_data(view: ExperimentView) -> dict[str, object]:
    return {
        "experiment": view.experiment.model_dump(mode="json"),
        "jobs": [job.model_dump(mode="json") for job in view.jobs],
    }


def _validate(draft_path: Path, capabilities_path: Path) -> dict[str, list[str]]:
    draft = _load(draft_path, ExperimentDraft)
    capabilities = _load(capabilities_path, AdapterCapabilities)
    violations = validate_manifest_support(draft.baseline, capabilities)
    if violations:
        raise ValidationFailure(violations)
    pair = build_controlled_pair(draft)
    violations = validate_manifest_support(pair.variant_b, capabilities)
    if violations:
        raise ValidationFailure(violations)
    return {"changed_paths": list(pair.changed_paths)}


def _worker(context: typer.Context, worker_id: str) -> Worker:
    service = _service(context)
    options = _context(context)
    artifact_root = _required_paths((options.artifacts, "artifacts"))[0]
    return Worker(
        repository=service.jobs,
        artifact_store=service.artifacts,
        adapter_registry=AdapterRegistry.with_defaults(),
        scratch_root=artifact_root / ".scratch",
        worker_id=worker_id,
    )


def _outcome_data(outcome: WorkerOutcome, service: ExperimentService) -> dict[str, object]:
    data: dict[str, object] = {"schema_version": "worker.v1", **asdict(outcome)}
    if outcome.error_code is not None and outcome.attempt_id is not None:
        attempts = service.jobs.attempts_for_job(outcome.job_id or "")
        attempt = next((item for item in attempts if item.id == outcome.attempt_id), None)
        data["error_message"] = None if attempt is None else attempt.error_message
    return data


def _status_data(service: ExperimentService, job_id: str) -> dict[str, object]:
    job = service.jobs.get(job_id)
    attempts = service.jobs.attempts_for_job(job_id)
    events = service.jobs.events_for_job(job_id)
    telemetry_count = sum(
        len(service.jobs.telemetry_for_attempt(attempt.id)) for attempt in attempts
    )
    return {
        "schema_version": "worker.v1",
        "job": job.model_dump(mode="json"),
        "current_attempt": None if not attempts else attempts[-1].model_dump(mode="json"),
        "attempts": [attempt.model_dump(mode="json") for attempt in attempts],
        "events": [event.model_dump(mode="json") for event in events],
        "telemetry_count": telemetry_count,
        "artifacts": [asdict(artifact) for artifact in service.artifacts.list_for_job(job_id)],
    }


@db_app.command("init")
def db_init(context: typer.Context, json_output: bool = typer.Option(False, "--json")) -> None:
    def initialize() -> dict[str, bool]:
        options = _context(context)
        database, _ = _required_paths(
            (options.database, "database"), (options.artifacts, "artifacts")
        )
        Database(database).migrate()
        return {"initialized": True}

    if json_output:
        _run_json(initialize)
    else:
        _run_human(initialize, lambda _: "Database initialized.")


@character_app.command("create")
def character_create(
    context: typer.Context,
    name: Annotated[str | None, typer.Argument()] = None,
) -> None:
    def create() -> dict[str, object]:
        record = _character_service(context).create_character(_required_nonblank_text(name, "name"))
        return {
            "character_id": record.character_id,
            "name": record.name,
            "created_at": record.created_at.isoformat(),
        }

    _run_character(create)


@character_app.command("import")
def character_import(
    context: typer.Context,
    character_id: Annotated[str | None, typer.Argument()] = None,
    source: Annotated[Path | None, typer.Option("--source")] = None,
) -> None:
    def import_images() -> object:
        source_path = _required_paths((source, "source"))[0]
        return _character_service(context).import_folder(
            _required_nonblank_text(character_id, "character_id"), source_path
        )

    _run_character(import_images)


@character_app.command("status")
def character_status(
    context: typer.Context,
    pack_id: Annotated[str | None, typer.Argument()] = None,
) -> None:
    _run_character(
        lambda: _character_service(context).pack_status(_required_nonblank_text(pack_id, "pack_id"))
    )


@character_app.command("open")
def character_open(
    context: typer.Context,
    character_id: Annotated[str | None, typer.Argument()] = None,
    pack_id: Annotated[str | None, typer.Option("--pack")] = None,
) -> None:
    def issue() -> dict[str, object]:
        grants, reviewer = _character_grants(context)
        token = grants.issue(
            reviewer,
            _required_nonblank_text(character_id, "character_id"),
            _required_nonblank_text(pack_id, "pack_id"),
        )
        return {
            "url": f"/characters/claim#token={token}",
            "expires_in_seconds": int(CHARACTER_CURATION_GRANT_TTL.total_seconds()),
        }

    _run_character(issue)


@character_app.command("curate")
def character_curate(
    context: typer.Context,
    pack_id: Annotated[str | None, typer.Argument()] = None,
    reference_id: Annotated[str | None, typer.Argument()] = None,
    framing: Annotated[Framing | None, typer.Option("--framing")] = None,
    head_angle: Annotated[HeadAngle | None, typer.Option("--head-angle")] = None,
    body_visibility: Annotated[BodyVisibility | None, typer.Option("--body-visibility")] = None,
    expression: Annotated[Expression | None, typer.Option("--expression")] = None,
    lighting: Annotated[Lighting | None, typer.Option("--lighting")] = None,
    occlusion: Annotated[Occlusion | None, typer.Option("--occlusion")] = None,
    identity_confirmed: Annotated[
        bool | None,
        typer.Option("--identity-confirmed/--identity-not-confirmed"),
    ] = None,
    accept: Annotated[bool, typer.Option("--accept")] = False,
    reject: Annotated[bool, typer.Option("--reject")] = False,
) -> None:
    def curate() -> object:
        annotations = (
            framing,
            head_angle,
            body_visibility,
            expression,
            lighting,
            occlusion,
            identity_confirmed,
        )
        if any(value is None for value in annotations):
            raise InvalidInput("every character annotation is required")
        if accept == reject:
            raise InvalidInput("exactly one of --accept or --reject is required")
        assert framing is not None
        assert head_angle is not None
        assert body_visibility is not None
        assert expression is not None
        assert lighting is not None
        assert occlusion is not None
        assert identity_confirmed is not None
        annotation = CharacterAnnotation(
            framing=framing,
            head_angle=head_angle,
            body_visibility=body_visibility,
            expression=expression,
            lighting=lighting,
            occlusion=occlusion,
            identity_confirmed=identity_confirmed,
        )
        return _character_service(context).curate_reference(
            _required_nonblank_text(pack_id, "pack_id"),
            _required_nonblank_text(reference_id, "reference_id"),
            annotation=annotation,
            disposition=(CuratorDisposition.ACTIVE if accept else CuratorDisposition.REJECTED),
        )

    _run_character(curate)


@character_app.command("resolve-warning")
def character_resolve_warning(
    context: typer.Context,
    pack_id: Annotated[str | None, typer.Argument()] = None,
    left_reference_id: Annotated[str | None, typer.Argument()] = None,
    right_reference_id: Annotated[str | None, typer.Argument()] = None,
    resolution: Annotated[str | None, typer.Option("--resolution")] = None,
) -> None:
    def resolve() -> object:
        if resolution not in {
            WarningResolution.ACCEPTED.value,
            WarningResolution.DISMISSED.value,
        }:
            raise InvalidInput("warning resolution must be accepted or dismissed")
        return _character_service(context).resolve_warning(
            _required_nonblank_text(pack_id, "pack_id"),
            _required_nonblank_text(left_reference_id, "left_reference_id"),
            _required_nonblank_text(right_reference_id, "right_reference_id"),
            WarningResolution(resolution),
        )

    _run_character(resolve)


@character_app.command("preregister")
def character_preregister(
    context: typer.Context,
    pack_id: Annotated[str | None, typer.Argument()] = None,
) -> None:
    _run_character(
        lambda: _character_service(context).preregister_pack(
            _required_nonblank_text(pack_id, "pack_id")
        )
    )


@character_app.command("export-pack")
def character_export_pack(
    context: typer.Context,
    pack_id: Annotated[str | None, typer.Argument()] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    def export() -> dict[str, object]:
        output_path = _required_paths((output, "output"))[0]
        if output_path.exists() or output_path.is_symlink():
            raise InvalidInput("character pack output already exists")
        exported = _character_service(context).export_pack(
            _required_nonblank_text(pack_id, "pack_id")
        )
        encoded = exported.model_dump_json() + "\n"
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        except FileExistsError as error:
            raise InvalidInput("character pack output already exists") from error
        except (OSError, UnicodeError) as error:
            raise InvalidInput("character pack export failed") from error
        return {
            "pack_id": exported.pack.pack_id,
            "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        }

    _run_character(export)


@experiment_app.command("validate")
def validate_experiment(
    context: typer.Context,
    draft: Annotated[Path | None, typer.Option("--draft")] = None,
    capabilities: Annotated[Path | None, typer.Option("--capabilities")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def validate() -> dict[str, list[str]]:
        draft_path, capabilities_path = _required_paths(
            (draft, "draft"), (capabilities, "capabilities")
        )
        return _validate(draft_path, capabilities_path)

    if json_output:
        _run_json(validate)
    else:
        _run_human(
            validate,
            lambda value: (
                f"Valid experiment: {', '.join(value['changed_paths'])}"
                if isinstance(value, dict) and isinstance(value.get("changed_paths"), list)
                else "Valid experiment."
            ),
        )


@experiment_app.command("create")
def create_experiment(
    context: typer.Context,
    draft: Annotated[Path | None, typer.Option("--draft")] = None,
    capabilities: Annotated[Path | None, typer.Option("--capabilities")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def create() -> dict[str, object]:
        draft_path, capabilities_path = _required_paths(
            (draft, "draft"), (capabilities, "capabilities")
        )
        draft_value = _load(draft_path, ExperimentDraft)
        capabilities_value = _load(capabilities_path, AdapterCapabilities)
        return _view_data(_service(context).create(draft_value, capabilities_value))

    if json_output:
        _run_json(create)
    else:
        _run_human(
            create,
            lambda value: (
                f"Created experiment {value['experiment']['id']}."
                if isinstance(value, dict)
                and isinstance(value.get("experiment"), dict)
                and "id" in value["experiment"]
                else "Created experiment."
            ),
        )


@experiment_app.command("get")
def get_experiment(
    context: typer.Context,
    experiment_id: Annotated[str | None, typer.Argument()] = None,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    def get() -> dict[str, object]:
        return _view_data(_service(context).get(_required_text(experiment_id, "experiment_id")))

    if json_output:
        _run_json(get)
    else:
        _run_human(get, lambda value: json.dumps(value, indent=2))


@experiment_app.command("list")
def list_experiments(
    context: typer.Context,
    limit: int = typer.Option(50, "--limit", min=1),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    def list_values() -> list[dict[str, object]]:
        return [_view_data(view) for view in _service(context).list(limit)]

    if json_output:
        _run_json(list_values)
    else:

        def render(values: object) -> str:
            if not isinstance(values, list) or not values:
                return "No experiments."
            return "\n".join(
                str(value["experiment"]["id"])
                for value in values
                if isinstance(value, dict) and isinstance(value.get("experiment"), dict)
            )

        _run_human(list_values, render)


@worker_app.command("run-once")
def worker_run_once(
    context: typer.Context,
    worker_id: Annotated[str | None, typer.Option("--worker-id")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def run() -> dict[str, object]:
        stable_worker_id = _required_nonblank_text(worker_id, "worker_id")
        service = _service(context)
        return _outcome_data(_worker(context, stable_worker_id).run_once(), service)

    if json_output:
        _run_json(run)
    else:
        _run_human(run, lambda value: json.dumps(value, indent=2))


@worker_app.command("recover")
def worker_recover(
    context: typer.Context,
    worker_id: Annotated[str | None, typer.Option("--worker-id")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def recover() -> dict[str, object]:
        stable_worker_id = _required_nonblank_text(worker_id, "worker_id")
        recovered = _worker(context, stable_worker_id).recover()
        return {
            "schema_version": "worker.v1",
            "recovered_count": len(recovered),
            "jobs": [job.model_dump(mode="json") for job in recovered],
        }

    if json_output:
        _run_json(recover)
    else:
        _run_human(recover, lambda value: json.dumps(value, indent=2))


@worker_app.command("cancel")
def worker_cancel(
    context: typer.Context,
    job_id: Annotated[str | None, typer.Argument()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def cancel() -> dict[str, object]:
        service = _service(context)
        job = service.jobs.request_cancellation(_required_text(job_id, "job_id"))
        return {"schema_version": "worker.v1", "job": job.model_dump(mode="json")}

    if json_output:
        _run_json(cancel)
    else:
        _run_human(cancel, lambda value: json.dumps(value, indent=2))


@worker_app.command("status")
def worker_status(
    context: typer.Context,
    job_id: Annotated[str | None, typer.Argument()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def status() -> dict[str, object]:
        return _status_data(_service(context), _required_text(job_id, "job_id"))

    if json_output:
        _run_json(status)
    else:
        _run_human(status, lambda value: json.dumps(value, indent=2))


def _review_show(context: typer.Context, session_id: str) -> dict[str, object]:
    service = _review_service(context)
    reviewer_id = _reviewer_id(context)
    for position in ("left", "right"):
        service.artifact_for_display(session_id, position, reviewer_id)
    view = BlindReviewView(
        session_id=session_id,
        media=(
            BlindMedia(display_position="left", url=f"/reviews/{session_id}/artifacts/left"),
            BlindMedia(display_position="right", url=f"/reviews/{session_id}/artifacts/right"),
        ),
        prompt=service.prompt_for_session(session_id, reviewer_id),
    )
    return _review_data("session", view)


@review_app.command("queue")
def review_queue(
    context: typer.Context,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def queue() -> dict[str, object]:
        return _review_data("items", _review_service(context).queue())

    if json_output:
        _run_review_json(queue)
    else:
        _run_review_human(queue)


@review_app.command("start")
def review_start(
    context: typer.Context,
    experiment_id: Annotated[str | None, typer.Argument()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def start() -> dict[str, object]:
        try:
            exact_id = _required_text(experiment_id, "experiment_id")
        except InvalidInput as error:
            raise InvalidReviewInput(str(error)) from error
        view = _review_service(context).start(exact_id, _reviewer_id(context))
        return _review_data("session", view)

    if json_output:
        _run_review_json(start)
    else:
        _run_review_human(start)


@review_app.command("show")
def review_show(
    context: typer.Context,
    session_id: Annotated[str | None, typer.Argument()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def show() -> dict[str, object]:
        try:
            exact_id = _required_text(session_id, "session_id")
        except InvalidInput as error:
            raise InvalidReviewInput(str(error)) from error
        return _review_show(context, exact_id)

    if json_output:
        _run_review_json(show)
    else:
        _run_review_human(show)


def _review_submit_action(
    context: typer.Context, session_id: str | None, input_path: Path | None, *, correction: bool
) -> dict[str, object]:
    try:
        exact_session_id = _required_text(session_id, "session_id")
        exact_input = _required_paths((input_path, "input"))[0]
    except InvalidInput as error:
        raise InvalidReviewInput(str(error)) from error
    document = _review_input(exact_input)
    service = _review_service(context)
    reviewer_id = _reviewer_id(context)
    if correction:
        if document.supersedes_review_id is None:
            raise InvalidReviewInput("correction requires supersedes_review_id")
        submission = service.correct(
            exact_session_id,
            reviewer_id,
            document.supersedes_review_id,
            document.display_preference,
            document.scores,
            document.notes,
        )
    else:
        if document.supersedes_review_id is not None:
            raise InvalidReviewInput("initial submission cannot supersede a review")
        submission = service.submit(
            exact_session_id,
            reviewer_id,
            document.display_preference,
            document.scores,
            document.notes,
        )
    return _review_data("submission", submission)


@review_app.command("submit")
def review_submit(
    context: typer.Context,
    session_id: Annotated[str | None, typer.Argument()] = None,
    input_path: Annotated[Path | None, typer.Option("--input")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def action() -> dict[str, object]:
        return _review_submit_action(context, session_id, input_path, correction=False)

    if json_output:
        _run_review_json(action)
    else:
        _run_review_human(action)


@review_app.command("correct")
def review_correct(
    context: typer.Context,
    session_id: Annotated[str | None, typer.Argument()] = None,
    input_path: Annotated[Path | None, typer.Option("--input")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def action() -> dict[str, object]:
        return _review_submit_action(context, session_id, input_path, correction=True)

    if json_output:
        _run_review_json(action)
    else:
        _run_review_human(action)


@review_app.command("reveal")
def review_reveal(
    context: typer.Context,
    session_id: Annotated[str | None, typer.Argument()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def reveal() -> dict[str, object]:
        try:
            exact_id = _required_text(session_id, "session_id")
        except InvalidInput as error:
            raise InvalidReviewInput(str(error)) from error
        value = _review_service(context).reveal(exact_id, _reviewer_id(context))
        return _review_data("reveal", value)

    if json_output:
        _run_review_json(reveal)
    else:
        _run_review_human(reveal)


@feasibility_app.command("start")
def feasibility_start(
    context: typer.Context,
    adapter_id: Annotated[str | None, typer.Argument()] = None,
    input_path: Annotated[Path | None, typer.Option("--input")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def start() -> dict[str, object]:
        from video_lab.animegen_feasibility import canonical_animegen_configuration
        from video_lab.feasibility_models import FeasibilityConfiguration

        exact_adapter = _required_nonblank_text(adapter_id, "adapter_id")
        AdapterRegistry.with_defaults().capabilities(exact_adapter)
        configuration_path = _required_paths((input_path, "input"))[0]
        configuration = _load(configuration_path, FeasibilityConfiguration)
        if configuration.adapter_id != exact_adapter:
            raise InvalidInput("configuration adapter_id does not match requested adapter")
        from video_lab.runtime_identity import animegen_runtime_versions

        if dict(configuration.runtime_versions) != animegen_runtime_versions():
            raise InvalidInput(
                "configuration runtime identity does not match this video-lab process"
            )
        canonical = canonical_animegen_configuration(configuration)
        service = cast(Any, _feasibility_service(context))
        run = service.start(canonical)
        return _feasibility_data(
            service.status(run.id),
            service.repository,
            reviewer_id=_projection_reviewer_id(context),
        )

    _run_feasibility(start, json_output=json_output)


@feasibility_app.command("runtime-identity")
def feasibility_runtime_identity(
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def identity() -> dict[str, object]:
        from video_lab.runtime_identity import animegen_runtime_versions, runtime_fingerprint

        versions = animegen_runtime_versions()
        return {
            "runtime_versions": versions,
            "runtime_fingerprint": runtime_fingerprint(versions),
        }

    _run_feasibility(identity, json_output=json_output)


@feasibility_app.command("status")
def feasibility_status(
    context: typer.Context,
    run_id: Annotated[str | None, typer.Argument()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def status() -> dict[str, object]:
        service = cast(Any, _feasibility_service(context, migrate=False))
        return _feasibility_data(
            service.status(_required_nonblank_text(run_id, "run_id")),
            service.repository,
            reviewer_id=_projection_reviewer_id(context),
        )

    _run_feasibility(status, json_output=json_output)


@feasibility_app.command("resume")
def feasibility_resume(
    context: typer.Context,
    run_id: Annotated[str | None, typer.Argument()] = None,
    cache: Annotated[Path | None, typer.Option("--cache")] = None,
    inventory_evidence: Annotated[Path | None, typer.Option("--inventory-evidence")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def resume() -> dict[str, object]:
        from video_lab.feasibility_models import FeasibilityStage

        exact_run_id = _required_nonblank_text(run_id, "run_id")
        imported: Mapping[str, object] | None = None
        if inventory_evidence is not None:
            try:
                raw = json.loads(inventory_evidence.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise InvalidInput("inventory evidence is not valid JSON") from error
            if not isinstance(raw, Mapping):
                raise InvalidInput("inventory evidence must be a JSON object")
            imported = cast(Mapping[str, object], raw)
        service = cast(
            Any,
            _feasibility_service(context, cache, run_id=exact_run_id, inventory_evidence=imported),
        )
        before = service.status(exact_run_id)
        next_stage = _next_feasibility_stage(before)
        cache_stages = {
            stage.value
            for stage in FeasibilityStage
            if stage not in {FeasibilityStage.RESOLVE_SOURCES, FeasibilityStage.PUBLISH_VERDICT}
        }
        if next_stage in cache_stages and cache is None:
            raise InvalidInput("cache is required for this feasibility stage")
        generation_stages = {
            FeasibilityStage.GENERATE_MINIMUM_1.value,
            FeasibilityStage.GENERATE_MINIMUM_2.value,
            FeasibilityStage.QUALIFY_STANDARD.value,
            FeasibilityStage.VALIDATE_CONTROLLED_PAIR.value,
        }
        if next_stage in generation_stages and _context(context).artifacts is None:
            raise InvalidInput("artifacts is required for this feasibility stage")
        prior_count = len(before.attempts)
        after = service.resume(exact_run_id)
        completed = after.attempts[-1] if len(after.attempts) > prior_count else None
        return _feasibility_data(
            after,
            service.repository,
            reviewer_id=_projection_reviewer_id(context),
            completed=completed,
        )

    _run_feasibility(resume, json_output=json_output)


@feasibility_app.command("export-inventory")
def feasibility_export_inventory(
    context: typer.Context,
    run_id: Annotated[str | None, typer.Argument()] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def export() -> dict[str, object]:
        from video_lab.feasibility import sanitize_diagnostic
        from video_lab.feasibility_models import FeasibilityStage, StageOutcome

        exact_run_id = _required_nonblank_text(run_id, "run_id")
        output_path = _required_paths((output, "output"))[0]
        if output_path.exists() or output_path.is_symlink():
            raise InvalidInput("inventory evidence output already exists")
        status = _feasibility_service(context, migrate=False).status(exact_run_id)  # type: ignore[attr-defined]
        preferred = (
            FeasibilityStage.DOWNLOAD_SOURCES,
            FeasibilityStage.INVENTORY_SOURCES,
        )
        attempt = next(
            (
                item
                for stage in preferred
                for item in reversed(status.attempts)
                if item.stage is stage
                and item.outcome is StageOutcome.PASSED
                and item.evidence is not None
                and isinstance(item.evidence.details.get("inventory"), Mapping)
            ),
            None,
        )
        if attempt is None or attempt.evidence is None:
            raise InvalidInput("run has no passed source inventory evidence")
        inventory = sanitize_diagnostic(attempt.evidence.details["inventory"])
        encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(encoded, encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise InvalidInput("inventory evidence export failed") from error
        return {
            "run_id": exact_run_id,
            "source_stage": attempt.stage.value,
            "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        }

    _run_feasibility(export, json_output=json_output)


@feasibility_app.command("export-evidence")
def feasibility_export_evidence(
    context: typer.Context,
    run_id: Annotated[str | None, typer.Argument()] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Export the reviewer-scoped phase ledger without media or cache bytes."""

    def export() -> dict[str, object]:
        exact_run_id = _required_nonblank_text(run_id, "run_id")
        output_path = _required_paths((output, "output"))[0]
        if output_path.exists() or output_path.is_symlink():
            raise InvalidInput("feasibility evidence output already exists")
        service = cast(Any, _feasibility_service(context, migrate=False))
        projection = _feasibility_data(
            service.status(exact_run_id),
            service.repository,
            reviewer_id=_projection_reviewer_id(context),
        )
        encoded = json.dumps(projection, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(encoded, encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise InvalidInput("feasibility evidence export failed") from error
        return {
            "run_id": exact_run_id,
            "schema_version": projection["schema_version"],
            "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        }

    _run_feasibility(export, json_output=json_output)


@study_app.command("create")
def study_create(
    context: typer.Context,
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    adapter: Annotated[str | None, typer.Option("--adapter")] = None,
    capabilities: Annotated[Path | None, typer.Option("--capabilities")] = None,
    prompt: Annotated[str | None, typer.Option("--prompt")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
) -> None:
    """Create or resume the content-addressed study for a frozen manifest."""

    def create() -> object:
        manifest_path = _required_paths((manifest, "manifest"))[0]
        frozen = _load(manifest_path, Manifest)
        if prompt is not None and prompt != frozen.prompt:
            raise InvalidStudyInput("prompt override conflicts with the frozen manifest")
        if seed is not None and seed != frozen.seed:
            raise InvalidStudyInput("seed override conflicts with the frozen manifest")
        selected_adapter = frozen.adapter_id if adapter is None else adapter.strip()
        if not selected_adapter or selected_adapter != frozen.adapter_id:
            raise InvalidStudyInput("adapter selection conflicts with the frozen manifest")
        registered_capabilities = AdapterRegistry.with_defaults().capabilities(selected_adapter)
        selected_capabilities = (
            registered_capabilities
            if capabilities is None
            else _load(capabilities, AdapterCapabilities)
        )
        if selected_capabilities != registered_capabilities:
            raise InvalidStudyInput("capabilities conflict with the registered adapter")
        config = OptimizerConfig()
        fingerprint = adaptive_study_input_fingerprint(frozen, config)
        service = _adaptive_study_service(context, selected_capabilities)
        reviewer = _study_reviewer_id(context)
        study_digest = hashlib.sha256(
            b"video-lab.adaptive-cli-study-id.v1\x00"
            + reviewer.encode("utf-8")
            + b"\x00"
            + fingerprint.encode("ascii")
        ).hexdigest()
        study_id = f"study_{study_digest[:24]}"
        try:
            definition = service.studies.get(study_id)
        except AdaptiveStudyNotFound:
            definition = AdaptiveStudyDefinition(
                id=study_id,
                prompt=frozen.prompt,
                negative_prompt=frozen.negative_prompt,
                seed=frozen.seed,
                base_manifest=frozen,
                optimizer_config=config,
                input_fingerprint=fingerprint,
                created_at=utc_now(),
            )
        try:
            return service.create_study(definition)
        except ValidationFailure:
            raise
        except ValueError as error:
            raise InvalidStudyInput("adaptive study manifest is invalid") from error

    _run_study_json(create)


@study_app.command("list")
def study_list(
    context: typer.Context,
    limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 50,
) -> None:
    """List bounded sealed study summaries."""

    def list_values() -> object:
        capabilities = AdapterRegistry.with_defaults().capabilities("video-lab.fake.v1")
        return _adaptive_study_service(context, capabilities).list(limit=limit)

    _run_study_json(list_values)


@study_app.command("show")
def study_show(
    context: typer.Context,
    study_id: Annotated[str | None, typer.Argument()] = None,
) -> None:
    """Show the reviewer-scoped sealed state of one study."""

    def show() -> object:
        exact_study_id = _required_nonblank_text(study_id, "study_id")
        return _adaptive_study_service_for_id(context, exact_study_id).show(exact_study_id)

    _run_study_json(show)


@study_app.command("resume")
def study_resume(
    context: typer.Context,
    study_id: Annotated[str | None, typer.Argument()] = None,
) -> None:
    """Resume the next deterministic local study action."""

    def resume() -> object:
        exact_study_id = _required_nonblank_text(study_id, "study_id")
        return _adaptive_study_service_for_id(context, exact_study_id).resume(exact_study_id)

    _run_study_json(resume)


@study_app.command("submit")
def study_submit(
    context: typer.Context,
    study_id: Annotated[str | None, typer.Argument()] = None,
    review_id: Annotated[str | None, typer.Option("--review-id")] = None,
    choice: Annotated[Literal["a", "b", "tie"] | None, typer.Option("--choice")] = None,
) -> None:
    """Submit a human A/B/tie display choice to its sealed review."""

    def submit() -> object:
        exact_study_id = _required_nonblank_text(study_id, "study_id")
        exact_review_id = _required_nonblank_text(review_id, "review_id")
        if choice is None:
            raise InvalidStudyInput("choice is required")
        display_choice = {
            "a": DisplayPreference.LEFT,
            "b": DisplayPreference.RIGHT,
            "tie": DisplayPreference.TIE,
        }[choice]
        return _adaptive_study_service_for_id(context, exact_study_id).record_preference(
            exact_study_id,
            exact_review_id,
            display_choice,
        )

    _run_study_json(submit)


@validation_app.command("create")
def validation_create(
    context: typer.Context,
    study_id: Annotated[str | None, typer.Option("--study")] = None,
    panel: Annotated[Path | None, typer.Option("--panel")] = None,
) -> None:
    """Create or replay a reviewer-owned balanced validation campaign."""

    def create() -> object:
        exact_study_id = _required_nonblank_text(study_id, "study_id")
        panel_path = _required_paths((panel, "panel"))[0]
        panel_input = _load(panel_path, ValidationPanelInput)
        return _validation_campaign_service_for_study(context, exact_study_id).create(
            stage_1_id=exact_study_id, panel_input=panel_input
        )

    _run_validation_json(create)


@validation_app.command("preview")
def validation_preview(
    context: typer.Context,
    campaign_id: Annotated[str | None, typer.Argument()] = None,
) -> None:
    """Show draft-only candidate and panel sizing information."""

    def preview() -> object:
        exact_campaign_id = _required_nonblank_text(campaign_id, "campaign_id")
        return _validation_campaign_service_for_id(context, exact_campaign_id).preview(
            exact_campaign_id
        )

    _run_validation_json(preview)


@validation_app.command("preregister")
def validation_preregister(
    context: typer.Context,
    campaign_id: Annotated[str | None, typer.Argument()] = None,
) -> None:
    """Freeze the balanced panel and materialize its controlled generation jobs."""

    def preregister() -> object:
        exact_campaign_id = _required_nonblank_text(campaign_id, "campaign_id")
        return _validation_campaign_service_for_id(context, exact_campaign_id).preregister(
            exact_campaign_id
        )

    _run_validation_json(preregister)


@validation_app.command("status")
def validation_status(
    context: typer.Context,
    campaign_id: Annotated[str | None, typer.Argument()] = None,
) -> None:
    """Show the reviewer-scoped sealed campaign status."""

    def status() -> object:
        exact_campaign_id = _required_nonblank_text(campaign_id, "campaign_id")
        return _validation_campaign_service_for_id(context, exact_campaign_id).show(
            exact_campaign_id
        )

    _run_validation_json(status)


@validation_app.command("resume")
def validation_resume(
    context: typer.Context,
    campaign_id: Annotated[str | None, typer.Argument()] = None,
) -> None:
    """Advance one deterministic campaign action and return its opaque review session."""

    def resume() -> object:
        exact_campaign_id = _required_nonblank_text(campaign_id, "campaign_id")
        service = _validation_campaign_service_for_id(context, exact_campaign_id)
        view = service.resume(exact_campaign_id)
        if view.review_id is None:
            return view
        trial_id = service.reviews.validation_trial_for_session(
            campaign_id=exact_campaign_id,
            session_id=view.review_id,
            reviewer_id=service.reviewer_id,
        )
        return view.model_copy(
            update={
                "media": tuple(
                    item.model_copy(
                        update={
                            "media_ref": service.reviews.artifacts.resolve(
                                service.reviews.artifact_for_validation_display(
                                    campaign_id=exact_campaign_id,
                                    trial_id=trial_id,
                                    session_id=view.review_id,
                                    position=item.position,
                                    reviewer_id=service.reviewer_id,
                                ).id
                            )
                            .resolve()
                            .as_uri()
                        }
                    )
                    for item in view.media
                )
            }
        )

    _run_validation_json(resume)


@validation_app.command("submit")
def validation_submit(
    context: typer.Context,
    campaign_id: Annotated[str | None, typer.Argument()] = None,
    review_id: Annotated[str | None, typer.Option("--review-id")] = None,
    choice: Annotated[Literal["a", "b", "tie"] | None, typer.Option("--choice")] = None,
) -> None:
    """Submit an A/B/tie judgment to the next opaque validation review."""

    def submit() -> object:
        exact_campaign_id = _required_nonblank_text(campaign_id, "campaign_id")
        exact_review_id = _required_nonblank_text(review_id, "review_id")
        if choice is None:
            raise InvalidValidationInput("choice is required")
        return _validation_campaign_service_for_id(context, exact_campaign_id).record_choice(
            exact_campaign_id, ValidationChoice(choice), review_id=exact_review_id
        )

    _run_validation_json(submit)


@validation_app.command("analyze")
def validation_analyze(
    context: typer.Context,
    campaign_id: Annotated[str | None, typer.Argument()] = None,
) -> None:
    """Run the deterministic terminal analysis after all judgments are persisted."""

    def analyze() -> object:
        exact_campaign_id = _required_nonblank_text(campaign_id, "campaign_id")
        return _validation_campaign_service_for_id(context, exact_campaign_id).analyze(
            exact_campaign_id
        )

    _run_validation_json(analyze)


@validation_app.command("export-evidence")
def validation_export_evidence(
    context: typer.Context,
    campaign_id: Annotated[str | None, typer.Argument()] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Write terminal campaign evidence once and return a SHA-256 receipt."""

    def export() -> dict[str, object]:
        exact_campaign_id = _required_nonblank_text(campaign_id, "campaign_id")
        output_path = _required_paths((output, "output"))[0]
        if output_path.exists() or output_path.is_symlink():
            raise InvalidValidationInput("validation evidence output already exists")
        evidence = _validation_campaign_service_for_id(context, exact_campaign_id).export_evidence(
            exact_campaign_id
        )
        encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        except (OSError, UnicodeError) as error:
            raise InvalidValidationInput("validation evidence export failed") from error
        return {
            "campaign_id": exact_campaign_id,
            "schema_version": evidence["schema_version"],
            "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        }

    _run_validation_json(export)


def _character_benchmark_resume_projection(service: Any, campaign_id: str, view: Any) -> object:
    custom = getattr(service, "cli_resume_projection", None)
    if custom is not None:
        return custom(campaign_id, view)
    data = view.model_dump(mode="json")
    if view.review_id is None or view.trial_id is None:
        data["phase"] = view.next_action
        return data
    campaign = service.campaigns.get(campaign_id)
    trial = service._trial(campaign_id, view.trial_id)
    session, _experiment_id, _alias = service._authorized_session(campaign, trial, view.review_id)
    mapping = service._display_mapping(trial, session)
    media = []
    for position in ("left", "right"):
        artifact = service.experiments.artifacts.get(mapping[position]["artifact_id"])
        path = service.experiments.artifacts.resolve(artifact.id).resolve()
        media.append({"position": position, "media_ref": path.as_uri()})
    pack = service.character_packs.export_pack(campaign.character_pack_id)
    references = [
        service.experiments.artifacts.resolve(item.artifact_id).resolve().as_uri()
        for item in pack.references
    ]
    return {
        "campaign_id": campaign_id,
        "state": view.state.value,
        "phase": view.next_action,
        "progress": view.progress.model_dump(mode="json"),
        "trial_id": view.trial_id,
        "review_id": view.review_id,
        "prompt": trial.context.prompt,
        "references": references,
        "media": media,
    }


@character_benchmark_app.command("create")
def character_benchmark_create(
    context: typer.Context,
    pack_id: Annotated[str | None, typer.Option("--pack")] = None,
    stage_2: Annotated[str | None, typer.Option("--stage-2-evidence")] = None,
    identity_prompt: Annotated[str | None, typer.Option("--identity-prompt")] = None,
    panel: Annotated[Path | None, typer.Option("--panel")] = None,
) -> None:
    def create() -> object:
        exact_pack = _required_nonblank_text(pack_id, "pack")
        exact_stage_2 = _required_nonblank_text(stage_2, "stage_2_evidence")
        exact_prompt = _required_nonblank_text(identity_prompt, "identity_prompt")
        panel_input = _load(_required_paths((panel, "panel"))[0], CharacterBenchmarkPanelInput)
        return _character_benchmark_service(context, exact_stage_2).create(
            character_pack_id=exact_pack,
            stage_2_campaign_id=exact_stage_2,
            identity_prompt=exact_prompt,
            panel_input=panel_input,
        )

    _run_character_benchmark_json(create)


def _benchmark_campaign_action(
    context: typer.Context, campaign_id: str | None, method: str
) -> object:
    exact = _required_nonblank_text(campaign_id, "campaign_id")
    return getattr(_character_benchmark_service_for_id(context, exact), method)(exact)


@character_benchmark_app.command("preview")
def character_benchmark_preview(
    context: typer.Context, campaign_id: Annotated[str | None, typer.Argument()] = None
) -> None:
    _run_character_benchmark_json(
        lambda: _benchmark_campaign_action(context, campaign_id, "preview")
    )


@character_benchmark_app.command("preregister")
def character_benchmark_preregister(
    context: typer.Context, campaign_id: Annotated[str | None, typer.Argument()] = None
) -> None:
    _run_character_benchmark_json(
        lambda: _benchmark_campaign_action(context, campaign_id, "preregister")
    )


@character_benchmark_app.command("status")
def character_benchmark_status(
    context: typer.Context, campaign_id: Annotated[str | None, typer.Argument()] = None
) -> None:
    _run_character_benchmark_json(lambda: _benchmark_campaign_action(context, campaign_id, "show"))


@character_benchmark_app.command("resume")
def character_benchmark_resume(
    context: typer.Context, campaign_id: Annotated[str | None, typer.Argument()] = None
) -> None:
    def resume() -> object:
        exact = _required_nonblank_text(campaign_id, "campaign_id")
        service = _character_benchmark_service_for_id(context, exact)
        return _character_benchmark_resume_projection(service, exact, service.resume(exact))

    _run_character_benchmark_json(resume)


@character_benchmark_app.command("submit-identity")
def character_benchmark_submit_identity(
    context: typer.Context,
    campaign_id: Annotated[str | None, typer.Argument()] = None,
    review_id: Annotated[str | None, typer.Option("--review-id")] = None,
    trial_id: Annotated[str | None, typer.Option("--trial-id")] = None,
    choice: Annotated[IdentityChoice | None, typer.Option("--choice")] = None,
) -> None:
    def submit() -> object:
        exact = _required_nonblank_text(campaign_id, "campaign_id")
        review = _required_nonblank_text(review_id, "review_id")
        trial = _required_nonblank_text(trial_id, "trial_id")
        if choice is None:
            raise InvalidInput("choice is required")
        return _character_benchmark_service_for_id(context, exact).record_identity(
            exact, review, IdentityObservation(benchmark_id=exact, trial_id=trial, choice=choice)
        )

    _run_character_benchmark_json(submit)


@character_benchmark_app.command("submit-secondary")
def character_benchmark_submit_secondary(
    context: typer.Context,
    campaign_id: Annotated[str | None, typer.Argument()] = None,
    review_id: Annotated[str | None, typer.Option("--review-id")] = None,
    trial_id: Annotated[str | None, typer.Option("--trial-id")] = None,
    dimension: Annotated[SecondaryDimension | None, typer.Option("--dimension")] = None,
    choice: Annotated[SecondaryChoice | None, typer.Option("--choice")] = None,
) -> None:
    def submit() -> object:
        exact = _required_nonblank_text(campaign_id, "campaign_id")
        review = _required_nonblank_text(review_id, "review_id")
        trial = _required_nonblank_text(trial_id, "trial_id")
        if dimension is None or choice is None:
            raise InvalidInput("dimension and choice are required")
        return _character_benchmark_service_for_id(context, exact).record_secondary(
            exact,
            review,
            SecondaryObservation(
                benchmark_id=exact, trial_id=trial, dimension=dimension, choice=choice
            ),
        )

    _run_character_benchmark_json(submit)


@character_benchmark_app.command("analyze")
def character_benchmark_analyze(
    context: typer.Context, campaign_id: Annotated[str | None, typer.Argument()] = None
) -> None:
    _run_character_benchmark_json(
        lambda: _benchmark_campaign_action(context, campaign_id, "analyze")
    )


@character_benchmark_app.command("export-evidence")
def character_benchmark_export_evidence(
    context: typer.Context,
    campaign_id: Annotated[str | None, typer.Argument()] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    def export() -> dict[str, object]:
        exact = _required_nonblank_text(campaign_id, "campaign_id")
        path = _required_paths((output, "output"))[0]
        if path.exists() or path.is_symlink():
            raise InvalidInput("character benchmark evidence output already exists")
        evidence = _character_benchmark_service_for_id(context, exact).export_evidence(exact)
        document = evidence.model_dump(mode="json") if hasattr(evidence, "model_dump") else evidence
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        except (OSError, UnicodeError) as error:
            raise InvalidInput("character benchmark evidence export failed") from error
        schema = document.get("schema_version") if isinstance(document, dict) else None
        return {
            "campaign_id": exact,
            "schema_version": schema,
            "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        }

    _run_character_benchmark_json(export)


@adapter_app.command("probe")
def adapter_probe(
    adapter_id: Annotated[str | None, typer.Argument()] = None,
    phase: Annotated[ProbePhase | None, typer.Option("--phase")] = None,
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    scratch_root: Annotated[Path | None, typer.Option("--scratch-root")] = None,
    allow_generate: Annotated[bool, typer.Option("--allow-generate")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def probe() -> dict[str, object]:
        exact_adapter_id = _required_text(adapter_id, "adapter_id")
        adapter = AdapterRegistry.with_defaults().resolve(exact_adapter_id)
        if phase is None:
            if manifest is not None or scratch_root is not None or allow_generate:
                raise InvalidInput("generation options require --phase generate")
            return {
                "schema_version": "worker.v1",
                "adapter_id": exact_adapter_id,
                "available": adapter.probe(),
            }
        if not isinstance(adapter, StagedProbeAdapter):
            raise InvalidInput(f"adapter {exact_adapter_id} does not support staged probes")
        request: ProbeRequest | None = None
        if phase is ProbePhase.GENERATE:
            if not allow_generate:
                raise InvalidInput("generate probe requires --allow-generate")
            manifest_path, requested_scratch = _required_paths(
                (manifest, "manifest"), (scratch_root, "scratch_root")
            )
            generation_manifest = _load(manifest_path, Manifest)
            safe_scratch = requested_scratch.resolve()
            try:
                safe_scratch.mkdir(parents=True, exist_ok=False)
            except FileExistsError as error:
                raise InvalidInput("scratch_root must not already exist") from error
            request = ProbeRequest(
                generation=GenerationRequest(
                    job_id="cli-probe",
                    attempt_id="cli-probe",
                    manifest=generation_manifest,
                    scratch_root=safe_scratch,
                ),
                progress=lambda _: None,
                cancelled=lambda: False,
            )
        elif manifest is not None or scratch_root is not None or allow_generate:
            raise InvalidInput("generation options require --phase generate")
        report = adapter.probe_phase(phase, request)
        return {
            "schema_version": "worker.v1",
            "adapter_id": exact_adapter_id,
            "available": report.ok,
            "probe_report": asdict(report),
        }

    if json_output:
        _run_json(probe)
    else:
        _run_human(probe, lambda value: json.dumps(value, indent=2))


if __name__ == "__main__":
    app()

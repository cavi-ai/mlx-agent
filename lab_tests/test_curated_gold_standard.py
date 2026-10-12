from __future__ import annotations

import hashlib
import io
import json
import shutil
import sqlite3
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from video_lab.artifacts import Artifact, ArtifactStore
from video_lab.compilers.wan_curated import (
    WanCompilerComponents,
    WanCompilerSnapshot,
    WanCuratedBatchCompiler,
    encode_wan_curated_source,
)
from video_lab.curated_ids import curated_dataset_digest, structured_caption_digest
from video_lab.curated_models import CuratedDatasetManifestV1, render_caption_v1
from video_lab.curated_source import (
    CuratedSourceError,
    CuratedSourceLease,
    DecodedFrame,
    decoded_pixel_digest,
)
from video_lab.curated_splits import assign_identity_aware_splits
from video_lab.curated_validation import ValidationThresholdsV1, validate_curated_dataset
from video_lab.database import Database
from video_lab.dataset_cli import (
    DatasetApplicationService,
    DatasetValidationRequest,
    FrameSourceBinding,
    run_command,
)
from video_lab.dataset_compiler import DatasetCompilationService, TransformInput, transform_sequence
from video_lab.dataset_receipts import DatasetCompilationRequest, InputBinding
from video_lab.domain import canonical_json
from video_lab.trainers.wan_factory import WanPinnedLocalRoot, root_manifest_digest

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "curated_sequence_v1"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pixels() -> dict[str, np.ndarray]:
    checker = np.zeros((8, 8, 3), dtype=np.uint8)
    checker[::2, ::2] = 255
    generator = np.random.default_rng(20260729)
    return {
        "a0": checker,
        "a1": checker.copy(),
        "a2": np.full((8, 8, 3), 255, dtype=np.uint8),
        "b0": np.zeros((8, 8, 3), dtype=np.uint8),
        "c0": checker.copy(),
        "d0": generator.integers(0, 256, (8, 8, 3), dtype=np.uint8),
        "d1": generator.integers(0, 256, (8, 8, 3), dtype=np.uint8),
    }


def _decoded(array: np.ndarray) -> DecodedFrame:
    owned = np.array(array, dtype=np.uint8, copy=True).tobytes()
    provisional = DecodedFrame(8, 8, 3, 8, "RGB", "0" * 64, owned)
    return DecodedFrame(8, 8, 3, 8, "RGB", decoded_pixel_digest(provisional), owned)


def _regenerate_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    arrays = _pixels()
    frames: dict[str, dict[str, object]] = {}
    layout = {
        "sequence_a": (("a0", 0), ("a1", 1), ("a2", 2)),
        "sequence_b": (("b0", 0),),
        "sequence_c": (("c0", 0),),
        "sequence_d": (("d0", 0), ("d1", 1)),
    }
    for sequence_id, entries in layout.items():
        directory = root / sequence_id
        directory.mkdir()
        for key, index in entries:
            relative = f"{sequence_id}/{index:04d}.png"
            path = root / relative
            Image.fromarray(arrays[key], mode="RGB").save(
                path, format="PNG", optimize=False, compress_level=9
            )
            decoded = _decoded(arrays[key])
            frames[key] = {
                "schema_version": "curated-frame.v1",
                "frame_id": f"frame_{key}",
                "frame_index": index,
                "timestamp_numerator": index,
                "timestamp_denominator": 8,
                "source_path": relative,
                "source_digest": _sha(path.read_bytes()),
                "pixel_digest": decoded.pixel_digest,
                "review_state": "accepted",
                "reviewer_evidence": ["gold_review"],
            }
    caption = {
        "schema_version": "structured-caption.v1",
        "character_token": "gold_character",
        "immutable_attributes": ["blue eyes", "synthetic identity"],
        "clothing": ["red coat"],
        "action": "walking",
        "pose": None,
        "expression": "neutral",
        "environment": "studio",
        "lighting": "soft light",
        "camera": "medium shot",
        "style": "anime",
        "omissions": ["text"],
    }
    mismatched_caption = {**caption, "immutable_attributes": ["green eyes", "synthetic identity"]}
    sequences = []
    for sequence_id, split, lineage, keys, sequence_caption in (
        ("sequence_a", "train", "lineage_a", ("a0", "a1", "a2"), caption),
        ("sequence_b", "validation", "lineage_b", ("b0",), mismatched_caption),
        ("sequence_c", "holdout", "lineage_c", ("c0",), caption),
        ("sequence_d", "train", "lineage_d", ("d0", "d1"), caption),
    ):
        sequences.append(
            {
                "schema_version": "curated-sequence.v1",
                "sequence_id": sequence_id,
                "identity_id": "gold_character",
                "source_lineage": lineage,
                "split": split,
                "caption": sequence_caption,
                "frames": [frames[key] for key in keys],
            }
        )
    manifest = {
        "schema_version": "curated-dataset-manifest.v1",
        "dataset_id": "curated_gold_v1",
        "sequences": sequences,
        "split_policy": {
            "schema_version": "dataset-split-policy.v1",
            "policy_version": "gold_policy_v1",
            "seed": 17,
            "train_numerator": 1,
            "validation_numerator": 1,
            "holdout_numerator": 1,
            "denominator": 3,
        },
        "transform_recipe": {
            "schema_version": "sequence-transform-recipe.v1",
            "recipe_id": "gold_transform_v1",
            "output_width": 8,
            "output_height": 8,
            "output_frame_count": 2,
            "temporal_stride_numerator": 1,
            "temporal_stride_denominator": 1,
            "resize_mode": "center_crop",
            "interpolation": "nearest",
            "padding_color": [0, 0, 0],
            "seed": 23,
        },
    }
    (root / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")


def _load_fixture() -> tuple[CuratedDatasetManifestV1, dict[str, DecodedFrame]]:
    manifest = CuratedDatasetManifestV1.model_validate_json(
        (FIXTURE_ROOT / "manifest.json").read_bytes(), strict=True
    )
    decoded: dict[str, DecodedFrame] = {}
    for sequence in manifest.sequences:
        for frame in sequence.frames:
            with Image.open(FIXTURE_ROOT / frame.source_path) as image:
                decoded[frame.frame_id] = _decoded(np.asarray(image, dtype=np.uint8))
    return manifest, decoded


def _read_securely(root: Path, manifest: CuratedDatasetManifestV1) -> dict[str, DecodedFrame]:
    expected = tuple(
        Artifact(
            id=f"artifact_{frame.frame_id}",
            job_id=None,
            sha256=frame.source_digest,
            byte_size=(root / frame.source_path).stat().st_size,
            media_type="image/png",
            relative_path=frame.source_path,
            created_at="1970-01-01T00:00:00+00:00",
        )
        for sequence in manifest.sequences
        for frame in sequence.frames
    )
    lease = CuratedSourceLease.open(root=root, expected=expected)
    try:
        result = {
            frame.frame_id: lease.read_frame(frame)
            for sequence in manifest.sequences
            for frame in sequence.frames
        }
        lease.reverify()
    finally:
        lease.close()
    return result


def test_committed_fixture_is_canonically_regenerated(tmp_path: Path) -> None:
    regenerated = tmp_path / "curated_sequence_v1"
    _regenerate_fixture(regenerated)
    expected = {
        path.relative_to(FIXTURE_ROOT).as_posix(): path.read_bytes()
        for path in FIXTURE_ROOT.rglob("*")
        if path.is_file()
    }
    observed = {
        path.relative_to(regenerated).as_posix(): path.read_bytes()
        for path in regenerated.rglob("*")
        if path.is_file()
    }
    assert observed == expected


def test_gold_validator_binds_exact_decisions_metrics_captions_and_digests() -> None:
    manifest, decoded = _load_fixture()
    report = validate_curated_dataset(
        manifest,
        decoded,
        ValidationThresholdsV1(
            perceptual_duplicate_hamming_max=4,
            blur_variance_min="1.0000",
            scene_cut_histogram_max="0.2000",
            temporal_motion_max="0.2000",
        ),
    )
    evidence = [
        (item.frame_id, item.reason_code, item.state, item.observed_value, item.threshold)
        for item in report.frame_findings
    ]
    assert evidence == [
        ("frame_a1", "exact_duplicate", "rejected", "0.0000", "0.0000"),
        ("frame_a2", "blur_below_minimum", "rejected", "0.0000", "1.0000"),
        ("frame_a2", "scene_cut", "rejected", "0.7500", "0.2000"),
        ("frame_a2", "temporal_discontinuity", "rejected", "0.7500", "0.2000"),
        ("frame_b0", "blur_below_minimum", "rejected", "0.0000", "1.0000"),
        ("frame_b0", "perceptual_duplicate", "rejected", "0.0000", "4.0000"),
        ("frame_c0", "exact_duplicate", "rejected", "0.0000", "0.0000"),
        ("frame_d1", "scene_cut", "rejected", "0.8125", "0.2000"),
        ("frame_d1", "temporal_discontinuity", "rejected", "0.3424", "0.2000"),
    ]
    sequence_evidence = [
        (item.reason_code, item.state, item.related_sequence_ids)
        for item in report.sequence_findings
    ]
    assert sequence_evidence == [
        ("caption_identity_mismatch", "needs_review", ("sequence_a", "sequence_b")),
        ("identity_inconsistency", "needs_review", ("sequence_a", "sequence_b")),
        ("near_duplicate_split_leakage", "rejected", ("sequence_a", "sequence_b")),
        ("near_duplicate_split_leakage", "rejected", ("sequence_a", "sequence_c")),
        ("near_duplicate_split_leakage", "rejected", ("sequence_a", "sequence_c")),
    ]
    assert report.dataset_digest == curated_dataset_digest(manifest)
    assert (
        report.dataset_digest == "154bb0e68c4e879b8b675f920056fc94a39e12d5fc0fad49ecd93691c8b6a52b"
    )
    assert report.split_assignment_evidence is not None
    assert (
        report.split_assignment_evidence.producer_receipt_digest
        == "5e8255e2b5be6a3ce00c12ddcb9189f464eecdbc1f2000b380f012f0459e8a40"
    )
    assigned = assign_identity_aware_splits(
        manifest,
        manifest.split_policy,
        decoded_frames=decoded,
        evidence=report.split_assignment_evidence,
        enforce_declared=False,
    )
    assert [(item.sequence_id, item.split) for item in assigned.sequences] == [
        ("sequence_a", "validation"),
        ("sequence_b", "validation"),
        ("sequence_c", "validation"),
        ("sequence_d", "train"),
    ]
    assert [
        (frame.frame_id, frame.source_digest, frame.pixel_digest)
        for sequence in manifest.sequences
        for frame in sequence.frames
    ] == [
        (
            "frame_a0",
            "94592bb924fe9da4b5549b1d443b64eec96503f14f84c77f5b14c7a55f54d2ef",
            "54a63dfec2839e076449814a06706faf5cec3d0554fb386c3693d091e7e65dd5",
        ),
        (
            "frame_a1",
            "94592bb924fe9da4b5549b1d443b64eec96503f14f84c77f5b14c7a55f54d2ef",
            "54a63dfec2839e076449814a06706faf5cec3d0554fb386c3693d091e7e65dd5",
        ),
        (
            "frame_a2",
            "b68de61392c506377e0d673fa36e239a47b67e603bdcb9cbf61d4ddbcd9722ff",
            "ba0b37ae7fd11cc13658748ac4bf97283638b6f4b09b17471ab85d62b38e40ec",
        ),
        (
            "frame_b0",
            "43c1b79b26009d2798246be983aa5ffa1a5757f916e10cdcad7cbc5fab3f40a0",
            "aca9fb4c6c680a79644b3fc18845ef27e8249b14afe83f87c43a7a3ef7003043",
        ),
        (
            "frame_c0",
            "94592bb924fe9da4b5549b1d443b64eec96503f14f84c77f5b14c7a55f54d2ef",
            "54a63dfec2839e076449814a06706faf5cec3d0554fb386c3693d091e7e65dd5",
        ),
        (
            "frame_d0",
            "bc88469f21dac182ea4bd1c2e920ffa0e28a402e0eb636c0f33ba99b54edf924",
            "ee7e71852edcd251647c8ef8fa9cea923c9eb59f8df15e7a3d725cc5d483a7ba",
        ),
        (
            "frame_d1",
            "b687de4ab69dfa30fc8bbc4eff6350b5eb7d5498228e8099bc2393bb226423fe",
            "78c63a80c5643ea75b9894e4a8b7f2fab20fb1f65f5f656c169c76cd0f74d3f6",
        ),
    ]
    blue_caption = (
        "token: gold_character; immutable identity: blue eyes, synthetic identity; "
        "clothing: red coat; action/pose: walking; expression: neutral; environment: studio; "
        "lighting: soft light; camera: medium shot; style: anime; omissions: text"
    )
    green_caption = (
        "token: gold_character; immutable identity: green eyes, synthetic identity; "
        "clothing: red coat; action/pose: walking; expression: neutral; environment: studio; "
        "lighting: soft light; camera: medium shot; style: anime; omissions: text"
    )
    assert [render_caption_v1(item.caption) for item in manifest.sequences] == [
        blue_caption,
        green_caption,
        blue_caption,
        blue_caption,
    ]


def test_gold_human_disposition_binds_only_the_exact_review_finding() -> None:
    manifest, decoded = _load_fixture()
    thresholds = ValidationThresholdsV1(
        perceptual_duplicate_hamming_max=4,
        blur_variance_min="1.0000",
        scene_cut_histogram_max="0.2000",
        temporal_motion_max="0.2000",
    )
    pending = validate_curated_dataset(manifest, decoded, thresholds)
    finding = next(
        item
        for item in pending.sequence_findings
        if item.reason_code == "caption_identity_mismatch"
    )
    reviewed = validate_curated_dataset(
        manifest, decoded, thresholds, human_dispositions={finding.finding_id: "accepted"}
    )
    bound = next(
        item for item in reviewed.sequence_findings if item.finding_id == finding.finding_id
    )
    assert (bound.state, bound.human_disposition) == ("accepted", "accepted")


class _Tokenizer:
    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        assert len(texts) == 1
        assert kwargs == {"padding": True, "return_tensors": "pt", "truncation": True}
        return {"input_ids": torch.tensor([[7, 11]], dtype=torch.int64)}


class _Placed:
    device = "mps"
    dtype = "torch.bfloat16"


class _Encoder:
    hf_device_map = {"": "mps"}
    device = "mps"

    def parameters(self):
        return iter((_Placed(),))

    def buffers(self):
        return iter((_Placed(),))

    def __call__(self, *, input_ids: torch.Tensor) -> tuple[torch.Tensor]:
        assert torch.equal(input_ids, torch.tensor([[7, 11]]))
        return (torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16),)


class _Vae(_Encoder):
    config = SimpleNamespace(scaling_factor=2.0)

    def encode(self, video: torch.Tensor) -> SimpleNamespace:
        assert video.shape == (1, 3, 2, 8, 8)
        latent = torch.tensor([[[[[0.25]]], [[[0.5]]]]], dtype=torch.bfloat16)
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: latent))


class _Scheduler:
    config = SimpleNamespace(num_train_timesteps=1000)


def _component_root(root: Path, name: str, revision: str = "a" * 40) -> WanPinnedLocalRoot:
    path = root / name
    path.mkdir(parents=True)
    (path / "config.json").write_text(canonical_json({"name": name}), encoding="utf-8")
    return WanPinnedLocalRoot(path, revision, root_manifest_digest(path))


def _compiler(root: Path, revision: str = "a" * 40) -> WanCuratedBatchCompiler:
    snapshot = WanCompilerSnapshot(
        tokenizer=_component_root(root, "tokenizer", revision),
        text_encoder=_component_root(root, "encoder", revision),
        vae=_component_root(root, "vae", revision),
        scheduler=_component_root(root, "scheduler", revision),
    )
    components = WanCompilerComponents(_Tokenizer(), _Encoder(), _Vae(), _Scheduler())
    return WanCuratedBatchCompiler(snapshot=snapshot, component_loader=lambda _: components)


def _stage(store: ArtifactStore, payload: bytes, media_type: str) -> Artifact:
    return store.commit(store.stage_bytes(payload, media_type), None)


def _compile_gold(root: Path) -> dict[str, object]:
    root.mkdir(parents=True)
    manifest, _ = _load_fixture()
    decoded_all = _read_securely(FIXTURE_ROOT, manifest)
    thresholds = ValidationThresholdsV1(
        perceptual_duplicate_hamming_max=4,
        blur_variance_min="1.0000",
        scene_cut_histogram_max="1.0000",
        temporal_motion_max="1.0000",
    )
    pending = validate_curated_dataset(manifest, decoded_all, thresholds)
    dispositions = {
        finding.finding_id: "accepted"
        for finding in pending.sequence_findings
        if finding.state == "needs_review"
    }
    reviewed = validate_curated_dataset(
        manifest, decoded_all, thresholds, human_dispositions=dispositions
    )
    assert all(
        finding.human_disposition == "accepted"
        for finding in reviewed.sequence_findings
        if finding.reason_code in {"caption_identity_mismatch", "identity_inconsistency"}
    )
    assert reviewed.split_assignment_evidence is not None
    assigned = assign_identity_aware_splits(
        manifest,
        manifest.split_policy,
        decoded_frames=decoded_all,
        evidence=reviewed.split_assignment_evidence,
        enforce_declared=False,
    )
    assert assigned.sequences[3].split == "train"
    assigned_pending = validate_curated_dataset(assigned, decoded_all, thresholds)
    dispositions = {
        finding.finding_id: "accepted"
        for finding in assigned_pending.sequence_findings
        if finding.state == "needs_review"
    }
    database = Database(root / "lab.sqlite3")
    database.migrate()
    store = ArtifactStore(database, root / "artifacts")
    backend = _compiler(root / "components")
    compilation = DatasetCompilationService(database, store, {backend.compiler_id: backend})
    application = DatasetApplicationService(compilation, database=database, artifacts=store)
    request_root = root / "requests"
    request_root.mkdir()

    def invoke(
        command: str,
        document: dict[str, object],
        *,
        source_root: Path | None = None,
        expected_ok: bool = True,
    ) -> dict[str, object]:
        (request_root / "request.json").write_text(canonical_json(document), encoding="utf-8")
        stdout = io.StringIO()
        with (
            patch("video_lab.dataset_cli.build_dataset_application", return_value=application),
            redirect_stdout(stdout),
        ):
            try:
                run_command(
                    command=command,  # type: ignore[arg-type]
                    database_path=database.path,
                    artifact_root=store.root,
                    request_root=request_root,
                    request_file="request.json",
                    json_output=True,
                    source_root=source_root,
                )
            except Exception as error:
                if expected_ok and (
                    command != "validate" or getattr(error, "exit_code", None) != 65
                ):
                    raise
                if not expected_ok and getattr(error, "exit_code", None) not in {
                    65,
                    67,
                    70,
                    71,
                }:
                    raise
        envelope = json.loads(stdout.getvalue())
        assert envelope["ok"] is expected_ok
        return envelope["data"] if expected_ok else envelope["error"]

    validation_request = DatasetValidationRequest(
        operation_key="validate_gold",
        manifest=assigned,
        frame_bindings=tuple(
            FrameSourceBinding(
                frame_id=frame.frame_id,
                artifact_id="fixture_" + frame.frame_id,
                relative_path=frame.source_path,
                sha256=frame.source_digest,
                byte_size=(FIXTURE_ROOT / frame.source_path).stat().st_size,
            )
            for sequence in assigned.sequences
            for frame in sequence.frames
        ),
        thresholds=thresholds,
        human_dispositions=dispositions,
    )
    validation = invoke(
        "validate", validation_request.model_dump(mode="json"), source_root=FIXTURE_ROOT
    )["validation_receipt"]
    rejected_sequences = {
        finding.sequence_id for finding in reviewed.frame_findings if finding.state == "rejected"
    } | {
        finding.sequence_id for finding in reviewed.sequence_findings if finding.state == "rejected"
    }
    assert rejected_sequences == {"sequence_a", "sequence_b", "sequence_c"}
    clean_manifest = assigned.model_copy(update={"sequences": (assigned.sequences[3],)})
    decoded = {
        frame_id: value for frame_id, value in decoded_all.items() if frame_id.startswith("frame_d")
    }
    accepted = validate_curated_dataset(
        clean_manifest,
        decoded,
        ValidationThresholdsV1(
            perceptual_duplicate_hamming_max=0,
            blur_variance_min="0.0000",
            scene_cut_histogram_max="1.0000",
            temporal_motion_max="1.0000",
        ),
    )
    assert accepted.state == "accepted"
    sequence = clean_manifest.sequences[0]
    transformed = transform_sequence(
        tuple(TransformInput(frame.frame_id, decoded[frame.frame_id]) for frame in sequence.frames),
        clean_manifest.transform_recipe,
    )
    source = encode_wan_curated_source(
        transformed.frames,
        transform_receipt=transformed.receipt,
        caption=sequence.caption,
        source_digest=curated_dataset_digest(assigned),
        validation_evidence={
            "complete_manifest_digest": curated_dataset_digest(assigned),
            "dispositions_digest": _sha(canonical_json(dispositions).encode()),
            "report_digest": validation["report_artifact"]["sha256"],
            "request_digest": validation["request_digest"],
            "split_evidence_digest": _sha(
                canonical_json(validation["report"]["split_assignment_evidence"]).encode()
            ),
            "validation_receipt_id": validation["receipt_id"],
        },
        selected_sequence_ids=(sequence.sequence_id,),
        selected_frame_source_digests=tuple(
            (frame.frame_id, frame.source_digest) for frame in sequence.frames
        ),
    )
    recipe = canonical_json(
        {
            "schema_version": "wan-curated-compiler-recipe.v1",
            "transform_recipe": clean_manifest.transform_recipe.model_dump(mode="json"),
            "timestep": 750,
            "timestep_boundary": 500,
            "latent_channels": 2,
            "vae_scaling_factor": 2.0,
        }
    ).encode()
    source_artifact = _stage(store, source, "application/json")
    recipe_artifact = _stage(store, recipe, "application/json")
    request = DatasetCompilationRequest(
        operation_key="compile_gold",
        compiler_id=backend.compiler_id,
        source=InputBinding(
            role="source_manifest",
            artifact_id=source_artifact.id,
            sha256=source_artifact.sha256,
            byte_size=source_artifact.byte_size,
            media_type=source_artifact.media_type,
        ),
        recipe=InputBinding(
            role="transform_recipe",
            artifact_id=recipe_artifact.id,
            sha256=recipe_artifact.sha256,
            byte_size=recipe_artifact.byte_size,
            media_type=recipe_artifact.media_type,
        ),
        output_roles=("batch", "manifest", "provenance"),
    )
    receipt = invoke("compile", request.model_dump(mode="json"))
    inspected = invoke(
        "inspect",
        {"schema_version": "dataset-inspect-request.v1", "receipt_id": receipt["receipt_id"]},
    )
    reproduced = invoke(
        "reproduce",
        {
            "schema_version": "dataset-reproduce-request.v1",
            "receipt_id": receipt["receipt_id"],
            "operation_key": "reproduce_gold",
        },
    )
    reproduced_view = invoke(
        "inspect",
        {
            "schema_version": "dataset-inspect-request.v1",
            "receipt_id": reproduced["receipt_id"],
        },
    )
    output_bytes = tuple(
        store.resolve(output["artifact_id"]).read_bytes() for output in receipt["outputs"]
    )
    return {
        "request": request.model_dump(mode="json"),
        "receipt": receipt,
        "inspection": inspected,
        "reproduction": reproduced,
        "reproduction_inspection": reproduced_view,
        "output_bytes": output_bytes,
        "source_bytes": source,
        "recipe_bytes": recipe,
        "invoke": invoke,
        "database": database,
        "store": store,
        "validation": validation,
        "application": application,
    }


def test_gold_pipeline_is_byte_reproducible_across_independent_local_roots(
    tmp_path: Path,
) -> None:
    first = _compile_gold(tmp_path / "first")
    second = _compile_gold(tmp_path / "second")
    assert first["output_bytes"] == second["output_bytes"]
    for result in (first, second):
        receipt = result["receipt"]
        inspection = result["inspection"]
        reproduction = result["reproduction"]
        reproduced_view = result["reproduction_inspection"]
        assert inspection["receipt"] == receipt
        assert reproduction["reproduces_receipt_id"] == receipt["receipt_id"]
        assert reproduced_view["receipt"] == reproduction
        assert [item["sha256"] for item in reproduction["outputs"]] == [
            item["sha256"] for item in receipt["outputs"]
        ]
    assert first["inspection"]["tensor_inventory"] == second["inspection"]["tensor_inventory"]
    assert first["inspection"]["provenance"] == second["inspection"]["provenance"]


def test_gold_tamper_matrix_changes_or_rejects_exact_authority(tmp_path: Path) -> None:
    result = _compile_gold(tmp_path / "compiled")
    receipt = result["receipt"]

    source_root = tmp_path / "mutated-source"
    shutil.copytree(FIXTURE_ROOT, source_root)
    frame_path = source_root / "sequence_d/0000.png"
    payload = frame_path.read_bytes()
    frame_path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    manifest, _ = _load_fixture()
    clean = manifest.model_copy(update={"sequences": (manifest.sequences[3],)})
    with pytest.raises(CuratedSourceError, match="curated source descriptor mismatch"):
        _read_securely(source_root, clean)

    caption = clean.sequences[0].caption
    changed_caption = caption.model_copy(update={"action": "running"})
    assert structured_caption_digest(changed_caption) != structured_caption_digest(caption)
    changed_recipe = clean.transform_recipe.model_copy(
        update={"seed": clean.transform_recipe.seed + 1}
    )
    assert (
        hashlib.sha256(canonical_json(changed_recipe).encode()).hexdigest()
        != hashlib.sha256(canonical_json(clean.transform_recipe).encode()).hexdigest()
    )

    changed_model_outputs = _compiler(tmp_path / "changed-model", "b" * 40).compile_batch(
        result["source_bytes"], result["recipe_bytes"]
    )
    assert changed_model_outputs[2].data != result["output_bytes"][2]

    database = Database(tmp_path / "compiled/lab.sqlite3")
    store = ArtifactStore(database, tmp_path / "compiled/artifacts")
    output = receipt["outputs"][0]
    output_path = store.resolve(output["artifact_id"])
    original_output = output_path.read_bytes()
    output_path.chmod(0o600)
    output_path.write_bytes(original_output[:-1] + bytes([original_output[-1] ^ 1]))
    with pytest.raises(Exception, match="artifact"):
        DatasetCompilationService(database, store, {}).inspect(receipt["receipt_id"])

    with (
        database.connect() as connection,
        pytest.raises(
            sqlite3.IntegrityError, match="dataset compilation operation authority mismatch"
        ),
    ):
        connection.execute(
            "UPDATE dataset_compilation_operations SET request_digest=? WHERE receipt_id=?",
            ("0" * 64, receipt["receipt_id"]),
        )


@pytest.mark.parametrize("authority", ["source", "output", "validation_report"])
def test_gold_artifact_tampering_is_rejected_through_actual_cli(
    authority: str, tmp_path: Path
) -> None:
    result = _compile_gold(tmp_path / authority)
    receipt = result["receipt"]
    store = result["store"]
    if authority == "source":
        artifact_id = receipt["source"]["artifact_id"]
    elif authority == "output":
        artifact_id = receipt["outputs"][0]["artifact_id"]
    else:
        artifact_id = result["validation"]["report_artifact"]["artifact_id"]
    path = store.resolve(artifact_id)
    payload = path.read_bytes()
    path.chmod(0o600)
    path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    command = "reproduce" if authority == "source" else "inspect"
    document = (
        {
            "schema_version": "dataset-reproduce-request.v1",
            "receipt_id": receipt["receipt_id"],
            "operation_key": "tampered_reproduction",
        }
        if command == "reproduce"
        else {
            "schema_version": "dataset-inspect-request.v1",
            "receipt_id": receipt["receipt_id"],
        }
    )
    error = result["invoke"](command, document, expected_ok=False)
    assert error["code"] == "integrity_failure"


@pytest.mark.parametrize(
    "authority",
    [
        "caption_action",
        "character_token",
        "recipe_seed",
        "component_manifest",
        "split_policy",
        "lineage",
        "declared_split",
        "rogue_normalization",
    ],
)
def test_gold_semantic_tampering_is_rejected_through_actual_compile_cli(
    authority: str, tmp_path: Path
) -> None:
    result = _compile_gold(tmp_path / authority)
    request = dict(result["request"])
    request["operation_key"] = "tampered_" + authority
    store = result["store"]
    if authority == "component_manifest":
        path = tmp_path / authority / "components" / "tokenizer" / "config.json"
        path.write_text('{"name":"replaced"}', encoding="utf-8")
    else:
        role = "recipe" if authority == "recipe_seed" else "source"
        document = json.loads(result[f"{role}_bytes"])
        if authority == "caption_action":
            document["caption"]["action"] = "running"
        elif authority == "character_token":
            document["caption"]["character_token"] = "rogue_character"
        elif authority == "recipe_seed":
            document["transform_recipe"]["seed"] += 1
        else:
            key = {
                "split_policy": "split_evidence_digest",
                "lineage": "complete_manifest_digest",
                "declared_split": "request_digest",
                "rogue_normalization": "dispositions_digest",
            }[authority]
            document["validation_evidence"][key] = "0" * 64
        artifact = _stage(store, canonical_json(document).encode(), "application/json")
        request[role] = {
            "role": "transform_recipe" if role == "recipe" else "source_manifest",
            "artifact_id": artifact.id,
            "sha256": artifact.sha256,
            "byte_size": artifact.byte_size,
            "media_type": artifact.media_type,
        }
    error = result["invoke"]("compile", request, expected_ok=False)
    expected = "conflict" if authority == "declared_split" else "rejected"
    assert error["code"] == expected


@pytest.mark.parametrize(
    ("attack", "reason"),
    [
        ("unknown_sequence", "validation_selection_sequence_unknown"),
        ("multiple_sequences", "validation_selection_multiple_sequences_unsupported"),
        ("non_train_sequence", "validation_selection_split_invalid"),
        ("coherent_caption", "validation_selection_caption_mismatch"),
        ("coherent_frame", "validation_selection_frames_mismatch"),
        ("coherent_source_digest", "validation_selection_source_digests_mismatch"),
        ("frame_order", "validation_selection_frames_mismatch"),
        ("omitted_frame", "validation_selection_frames_mismatch"),
        ("extra_frame", "validation_selection_frames_mismatch"),
    ],
)
def test_validation_selection_rejects_recomputed_unrelated_authority_before_backend(
    attack: str, reason: str, tmp_path: Path
) -> None:
    result = _compile_gold(tmp_path / attack)
    document = json.loads(result["source_bytes"])
    if attack == "unknown_sequence":
        document["selected_sequence_ids"] = ["sequence_unknown"]
    elif attack == "multiple_sequences":
        document["selected_sequence_ids"] = ["sequence_a", "sequence_d"]
    elif attack == "non_train_sequence":
        document["selected_sequence_ids"] = ["sequence_a"]
    elif attack == "coherent_caption":
        document["caption"]["action"] = "running"
        document["caption_digest"] = _sha(canonical_json(document["caption"]).encode())
    elif attack == "coherent_frame":
        document["transform_receipt"]["inputs"][0]["frame_id"] = "frame_a0"
        document["transform_receipt_digest"] = _sha(
            canonical_json(document["transform_receipt"]).encode()
        )
    elif attack == "coherent_source_digest":
        document["selected_frame_source_digests"][0][1] = "0" * 64
    elif attack == "omitted_frame":
        document["transform_receipt"]["inputs"].pop()
        document["transform_receipt_digest"] = _sha(
            canonical_json(document["transform_receipt"]).encode()
        )
    elif attack == "extra_frame":
        document["transform_receipt"]["inputs"].append(
            dict(document["transform_receipt"]["inputs"][0], frame_id="frame_extra")
        )
        document["transform_receipt_digest"] = _sha(
            canonical_json(document["transform_receipt"]).encode()
        )
    else:
        document["transform_receipt"]["inputs"].reverse()
        document["transform_receipt_digest"] = _sha(
            canonical_json(document["transform_receipt"]).encode()
        )
    artifact = _stage(result["store"], canonical_json(document).encode(), "application/json")
    request_document = dict(result["request"])
    request_document["operation_key"] = "selection_" + attack
    request_document["source"] = {
        "role": "source_manifest",
        "artifact_id": artifact.id,
        "sha256": artifact.sha256,
        "byte_size": artifact.byte_size,
        "media_type": artifact.media_type,
    }

    class BackendMustNotRun:
        calls = 0

        def compile(self, request: object) -> object:
            self.calls += 1
            raise AssertionError("backend_called")

    backend = BackendMustNotRun()
    application = DatasetApplicationService(
        backend, database=result["database"], artifacts=result["store"]  # type: ignore[arg-type]
    )
    with pytest.raises(Exception, match=reason):
        application.compile(
            DatasetCompilationRequest.model_validate_json(
                canonical_json(request_document), strict=True
            )
        )
    assert backend.calls == 0


@pytest.mark.parametrize(
    "attack",
    ["receipt_id_swap", "fabricated_receipt", "request_operation_mismatch", "row_digest"],
)
def test_fabricated_validation_authority_is_rejected_before_backend(
    attack: str, tmp_path: Path
) -> None:
    result = _compile_gold(tmp_path / attack)
    request_document = dict(result["request"])
    request_document["operation_key"] = "fabricated_" + attack
    if attack == "receipt_id_swap":
        source = json.loads(result["source_bytes"])
        source["validation_evidence"]["validation_receipt_id"] = "dvr_" + "0" * 32
        artifact = _stage(result["store"], canonical_json(source).encode(), "application/json")
        request_document["source"] = {
            "role": "source_manifest",
            "artifact_id": artifact.id,
            "sha256": artifact.sha256,
            "byte_size": artifact.byte_size,
            "media_type": artifact.media_type,
        }
    else:
        database = result["database"]
        validation = result["validation"]
        with database.connect() as connection:
            connection.execute("DROP TRIGGER guard_curated_validation_update_v49")
            if attack == "row_digest":
                connection.execute(
                    "UPDATE curated_validation_operations SET receipt_digest=? "
                    "WHERE receipt_id=?",
                    ("0" * 64, validation["receipt_id"]),
                )
            else:
                receipt = dict(validation)
                if attack == "request_operation_mismatch":
                    receipt["operation_key"] = "swapped_operation"
                else:
                    receipt["report_artifact"] = dict(request_document["source"])
                    receipt["report_artifact"]["role"] = "validation_report"
                receipt_json = canonical_json(receipt)
                connection.execute(
                    "UPDATE curated_validation_operations SET receipt_json=?,receipt_digest=? "
                    "WHERE receipt_id=?",
                    (receipt_json, _sha(receipt_json.encode()), validation["receipt_id"]),
                )

    class BackendMustNotRun:
        calls = 0

        def compile(self, request: object) -> object:
            self.calls += 1
            raise AssertionError("backend_called")

    backend = BackendMustNotRun()
    application = DatasetApplicationService(
        backend, database=result["database"], artifacts=result["store"]  # type: ignore[arg-type]
    )
    request = DatasetCompilationRequest.model_validate_json(
        canonical_json(request_document), strict=True
    )
    with pytest.raises(Exception, match="validation (receipt|authority|report)"):
        application.compile(request)
    assert backend.calls == 0


def test_fabricated_compilation_receipt_using_real_artifacts_is_rejected_pre_backend(
    tmp_path: Path,
) -> None:
    result = _compile_gold(tmp_path / "fabricated-compilation")
    receipt = dict(result["receipt"])
    outputs = [dict(item) for item in receipt["outputs"]]
    outputs[0] = {**outputs[1], "role": "batch"}
    receipt["outputs"] = outputs
    receipt_json = canonical_json(receipt)
    with result["database"].connect() as connection:
        connection.execute("DROP TRIGGER guard_dataset_compilation_operation_authority_v48")
        connection.execute(
            "UPDATE dataset_compilation_operations SET receipt_json=?,receipt_digest=? "
            "WHERE receipt_id=?",
            (receipt_json, _sha(receipt_json.encode()), receipt["receipt_id"]),
        )
    with pytest.raises(Exception, match="durable compilation receipt mismatch"):
        result["application"].reproduce(
            receipt["receipt_id"], operation_key="fabricated_compilation_reproduction"
        )

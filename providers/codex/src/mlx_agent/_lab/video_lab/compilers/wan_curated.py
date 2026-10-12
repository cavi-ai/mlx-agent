"""Strictly local deterministic compiler for curated Wan training batches."""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
import random
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from video_lab.curated_models import (
    SequenceTransformRecipeV1,
    StructuredCaptionV1,
    render_caption_v1,
)
from video_lab.dataset_compiler import (
    BackendOutput,
    DatasetCompilationError,
    TransformReceipt,
    validate_transformed_sequence,
)
from video_lab.domain import canonical_json
from video_lab.trainers.wan_batch import (
    build_wan_batch_manifest,
    validate_wan_batch_bytes,
)
from video_lab.trainers.wan_factory import (
    WanPinnedLocalRoot,
    WanSnapshotCopyLimits,
    require_component_on_mps,
    root_manifest_digest,
    secure_snapshot_pinned_root,
)

_RNG_LOCK = threading.Lock()
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class WanCompilerLimits:
    max_source_bytes: int = 4 * 1024 * 1024 * 1024
    max_recipe_bytes: int = 1024 * 1024
    max_frames: int = 256
    max_pixels: int = 4096 * 4096 * 256
    max_tensor_bytes: int = 16 * 1024 * 1024 * 1024
    max_snapshot_files: int = 100_000
    max_snapshot_bytes: int = 64 * 1024 * 1024 * 1024
    max_snapshot_depth: int = 32


DEFAULT_WAN_COMPILER_LIMITS = WanCompilerLimits()


@dataclass(frozen=True, slots=True)
class WanCompilerSnapshot:
    tokenizer: WanPinnedLocalRoot
    text_encoder: WanPinnedLocalRoot
    vae: WanPinnedLocalRoot
    scheduler: WanPinnedLocalRoot


@dataclass(frozen=True, slots=True)
class WanCompilerComponents:
    tokenizer: object
    text_encoder: object
    vae: object
    scheduler: object


class _CompilerRecipe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    schema_version: Literal["wan-curated-compiler-recipe.v1"]
    transform_recipe: SequenceTransformRecipeV1
    timestep: int = Field(ge=0)
    timestep_boundary: int = Field(gt=0)
    latent_channels: int = Field(gt=0)
    vae_scaling_factor: float = Field(gt=0)


class _CuratedSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    schema_version: Literal["wan-curated-source.v1"]
    shape: tuple[int, int, int, int]
    dtype: Literal["uint8"]
    frames_base64: str
    frame_digest: str
    transform_receipt: TransformReceipt
    transform_receipt_digest: str
    caption: StructuredCaptionV1
    caption_digest: str
    source_digest: str
    validation_evidence: dict[str, str]
    selected_sequence_ids: tuple[str, ...]
    selected_frame_source_digests: tuple[tuple[str, str], ...]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_wan_curated_source(
    frames: np.ndarray,
    *,
    transform_receipt: TransformReceipt,
    caption: StructuredCaptionV1,
    source_digest: str,
    validation_evidence: Mapping[str, str],
    selected_sequence_ids: tuple[str, ...],
    selected_frame_source_digests: tuple[tuple[str, str], ...],
) -> bytes:
    """Encode the Task 4 transformed array and its exact evidence canonically."""
    array = np.asarray(frames)
    if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("wan_compiler_source_invalid")
    frame_bytes = array.tobytes(order="C")
    receipt_bytes = canonical_json(transform_receipt).encode()
    caption_bytes = canonical_json(caption).encode()
    payload = {
        "schema_version": "wan-curated-source.v1",
        "shape": list(array.shape),
        "dtype": "uint8",
        "frames_base64": base64.b64encode(frame_bytes).decode("ascii"),
        "frame_digest": _sha(frame_bytes),
        "transform_receipt": transform_receipt.model_dump(mode="json"),
        "transform_receipt_digest": _sha(receipt_bytes),
        "caption": caption.model_dump(mode="json"),
        "caption_digest": _sha(caption_bytes),
        "source_digest": source_digest,
        "validation_evidence": dict(sorted(validation_evidence.items())),
        "selected_sequence_ids": list(selected_sequence_ids),
        "selected_frame_source_digests": [list(item) for item in selected_frame_source_digests],
    }
    return canonical_json(payload).encode()


class WanCuratedBatchCompiler:
    compiler_id = "wan_curated_v1"

    def __init__(
        self,
        *,
        snapshot: WanCompilerSnapshot,
        component_loader: Callable[[WanCompilerSnapshot], WanCompilerComponents] | None = None,
        importer: Callable[[str], object] = importlib.import_module,
        limits: WanCompilerLimits = DEFAULT_WAN_COMPILER_LIMITS,
    ) -> None:
        self._snapshot = snapshot
        self._importer = importer
        self._uses_default_loader = component_loader is None
        self._loader = component_loader or self._load_components
        self._limits = limits

    def validate_recipe(self, recipe_bytes: bytes) -> None:
        self._parse_recipe(recipe_bytes)
        self._snapshot_evidence(self._snapshot)

    def compile_batch(self, source_bytes: bytes, recipe_bytes: bytes) -> tuple[BackendOutput, ...]:
        if len(source_bytes) > self._limits.max_source_bytes:
            raise ValueError("wan_compiler_budget_exceeded")
        recipe = self._parse_recipe(recipe_bytes)
        source, frames = self._parse_source(source_bytes, recipe.transform_recipe)
        evidence_before = self._snapshot_evidence(self._snapshot)
        vae_seed = _derived_seed(recipe.transform_recipe.seed, "wan-vae-latent.v1")
        noise_seed = _derived_seed(recipe.transform_recipe.seed, "wan-flow-noise.v1")
        with _fenced_rng(self._importer, seed=vae_seed) as torch:
            components = self._loader(self._snapshot)
            try:
                evidence_after_load = self._snapshot_evidence(self._snapshot)
            except ValueError:
                raise ValueError("wan_compiler_snapshot_replaced") from None
            if evidence_after_load != evidence_before:
                raise ValueError("wan_compiler_snapshot_replaced")
            require_component_on_mps(components.text_encoder)
            require_component_on_mps(components.vae)
            timestep_max = self._scheduler_timesteps(components.scheduler)
            if not 0 <= recipe.timestep <= timestep_max:
                raise ValueError("wan_compiler_timestep_invalid")
            if not 0 < recipe.timestep_boundary <= timestep_max:
                raise ValueError("wan_compiler_selector_invalid")
            actual_scale = float(
                getattr(getattr(components.vae, "config", None), "scaling_factor", 0)
            )
            if actual_scale != recipe.vae_scaling_factor:
                raise ValueError("wan_compiler_vae_scale_invalid")
            rendered = render_caption_v1(source.caption)
            tokens = cast(Any, components.tokenizer)(
                [rendered], padding=True, return_tensors="pt", truncation=True
            )
            if not isinstance(tokens, Mapping) or set(tokens) != {"input_ids"}:
                raise ValueError("wan_compiler_tokenization_invalid")
            token_ids = tokens["input_ids"]
            model_token_ids = token_ids.to(device="mps") if self._uses_default_loader else token_ids
            encoder_hidden = cast(Any, components.text_encoder)(input_ids=model_token_ids)[0]
            video = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).unsqueeze(0)
            video = video.to(dtype=torch.float32).div(127.5).sub(1.0)
            if self._uses_default_loader:
                video = video.to(device="mps")
            clean = cast(Any, components.vae).encode(video).latent_dist.sample()
            clean = (
                clean.detach().to(device="cpu", dtype=torch.bfloat16) * recipe.vae_scaling_factor
            )
            encoder_hidden = encoder_hidden.detach().to(device="cpu", dtype=torch.bfloat16)
            if (
                clean.ndim < 3
                or int(clean.shape[0]) != 1
                or int(clean.shape[1]) != recipe.latent_channels
            ):
                raise ValueError("wan_compiler_latent_shape_invalid")
            generator = torch.Generator(device="cpu").manual_seed(noise_seed)
            noise = torch.randn(clean.shape, dtype=clean.dtype, device="cpu", generator=generator)
            timestep = torch.tensor([float(recipe.timestep)], dtype=torch.float32)
            sigma = (timestep / float(timestep_max)).to(dtype=clean.dtype)
            while sigma.ndim < clean.ndim:
                sigma = sigma.unsqueeze(-1)
            selector = (timestep < recipe.timestep_boundary).to(dtype=torch.int64)
            if int(selector.item()) != (1 if recipe.timestep < recipe.timestep_boundary else 0):
                raise ValueError("wan_compiler_selector_invalid")
            tensors = {
                "clean_latents": clean.contiguous(),
                "noise_latents": noise.contiguous(),
                "hidden_states": (sigma * noise + (1.0 - sigma) * clean).contiguous(),
                "target": (noise - clean).contiguous(),
                "timestep": timestep.contiguous(),
                "transformer_selector": selector.contiguous(),
                "encoder_hidden_states": encoder_hidden.to(dtype=torch.bfloat16).contiguous(),
            }
            self._validate_tensor_budget(tensors)
            serializer = cast(Any, self._importer("safetensors.torch"))
            batch = cast(bytes, serializer.save({name: tensors[name] for name in sorted(tensors)}))
            manifest = build_wan_batch_manifest(
                batch,
                batch_size=1,
                timestep_min=0,
                timestep_max=timestep_max,
                timestep_boundary=recipe.timestep_boundary,
            )
            validate_wan_batch_bytes(manifest, batch, torch_module=torch)
        if self._snapshot_evidence(self._snapshot) != evidence_before:
            raise ValueError("wan_compiler_snapshot_replaced")
        provenance = canonical_json(
            {
                "schema_version": "wan-curated-batch-provenance.v1",
                "source_digest": source.source_digest,
                "source_document_digest": _sha(source_bytes),
                "frame_digest": source.frame_digest,
                "transform_receipt_digest": source.transform_receipt_digest,
                "caption_digest": source.caption_digest,
                "rendered_caption": rendered,
                "token_ids": token_ids.detach().to("cpu").tolist(),
                "recipe_digest": _sha(recipe_bytes),
                "component_digests": evidence_before,
                "seed": recipe.transform_recipe.seed,
                "batch_digest": _sha(batch),
                "manifest_digest": _sha(manifest),
            }
        ).encode()
        return (
            BackendOutput("batch", batch, "application/vnd.safetensors"),
            BackendOutput("manifest", manifest, "application/json"),
            BackendOutput("provenance", provenance, "application/json"),
        )

    def _parse_recipe(self, payload: bytes) -> _CompilerRecipe:
        if len(payload) > self._limits.max_recipe_bytes:
            raise ValueError("wan_compiler_budget_exceeded")
        try:
            parsed = json.loads(payload)
            if canonical_json(parsed).encode() != payload:
                raise ValueError
            parsed["transform_recipe"] = SequenceTransformRecipeV1.model_validate_json(
                canonical_json(parsed["transform_recipe"])
            )
            return _CompilerRecipe.model_validate(parsed)
        except (UnicodeError, json.JSONDecodeError, ValueError):
            raise ValueError("wan_compiler_recipe_invalid") from None

    def _parse_source(
        self, payload: bytes, transform: SequenceTransformRecipeV1
    ) -> tuple[_CuratedSource, np.ndarray]:
        try:
            parsed = json.loads(payload)
            if canonical_json(parsed).encode() != payload:
                raise ValueError
            parsed["shape"] = tuple(parsed["shape"])
            parsed["selected_sequence_ids"] = tuple(parsed["selected_sequence_ids"])
            parsed["selected_frame_source_digests"] = tuple(
                tuple(item) for item in parsed["selected_frame_source_digests"]
            )
            parsed["caption"] = StructuredCaptionV1.model_validate_json(
                canonical_json(parsed["caption"])
            )
            parsed["transform_receipt"] = TransformReceipt.model_validate_json(
                canonical_json(parsed["transform_receipt"])
            )
            source = _CuratedSource.model_validate(parsed)
            if len(source.source_digest) != 64 or any(c not in _HEX for c in source.source_digest):
                raise ValueError
            raw = base64.b64decode(source.frames_base64, validate=True)
            if _sha(raw) != source.frame_digest:
                raise ValueError
            if (
                _sha(canonical_json(source.transform_receipt).encode())
                != source.transform_receipt_digest
            ):
                raise ValueError
            if _sha(canonical_json(source.caption).encode()) != source.caption_digest:
                raise ValueError
            if (
                source.transform_receipt.recipe_id != transform.recipe_id
                or source.transform_receipt.seed != transform.seed
            ):
                raise ValueError
            expected_shape = (
                transform.output_frame_count,
                transform.output_height,
                transform.output_width,
                3,
            )
            if source.shape != expected_shape or len(raw) != int(np.prod(expected_shape)):
                raise ValueError
            if (
                source.shape[0] > self._limits.max_frames
                or int(np.prod(source.shape)) > self._limits.max_pixels * 3
            ):
                raise ValueError("budget")
            frames = np.frombuffer(raw, dtype=np.uint8).reshape(source.shape).copy()
            frames.flags.writeable = False
            try:
                validate_transformed_sequence(frames, source.transform_receipt, transform)
            except DatasetCompilationError:
                raise ValueError from None
            return source, frames
        except ValueError as error:
            if str(error) == "budget":
                raise ValueError("wan_compiler_budget_exceeded") from None
            raise ValueError("wan_compiler_source_invalid") from None
        except (UnicodeError, json.JSONDecodeError, TypeError):
            raise ValueError("wan_compiler_source_invalid") from None

    def _snapshot_evidence(self, snapshot: WanCompilerSnapshot) -> dict[str, dict[str, str]]:
        evidence: dict[str, dict[str, str]] = {}
        for name in ("tokenizer", "text_encoder", "vae", "scheduler"):
            root = cast(WanPinnedLocalRoot, getattr(snapshot, name))
            if (
                type(root) is not WanPinnedLocalRoot
                or len(root.revision) != 40
                or any(c not in _HEX for c in root.revision)
                or root.manifest_digest is None
                or root.manifest_digest
                != root_manifest_digest(
                    root.path,
                    max_files=self._limits.max_snapshot_files,
                    max_bytes=self._limits.max_snapshot_bytes,
                    max_depth=self._limits.max_snapshot_depth,
                )
            ):
                raise ValueError("wan_compiler_snapshot_invalid")
            evidence[name] = {"revision": root.revision, "manifest_digest": root.manifest_digest}
        return evidence

    def _scheduler_timesteps(self, scheduler: object) -> int:
        value = getattr(getattr(scheduler, "config", None), "num_train_timesteps", None)
        if type(value) is not int or value <= 0:
            raise ValueError("wan_compiler_scheduler_invalid")
        return value

    def _validate_tensor_budget(self, tensors: Mapping[str, Any]) -> None:
        byte_size = sum(int(item.numel()) * int(item.element_size()) for item in tensors.values())
        if byte_size > self._limits.max_tensor_bytes:
            raise ValueError("wan_compiler_budget_exceeded")
        if any(
            not bool(item.isfinite().all().item())
            for name, item in tensors.items()
            if name != "transformer_selector"
        ):
            raise ValueError("wan_compiler_tensor_invalid")

    def _load_components(self, snapshot: WanCompilerSnapshot) -> WanCompilerComponents:
        torch = cast(Any, self._importer("torch"))
        transformers = cast(Any, self._importer("transformers"))
        diffusers = cast(Any, self._importer("diffusers"))
        with _private_snapshot(snapshot, self._limits) as local:
            common = {"local_files_only": True}
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(local.tokenizer.path), **common
            )
            encoder = transformers.UMT5EncoderModel.from_pretrained(
                str(local.text_encoder.path), torch_dtype=torch.bfloat16, **common
            ).to(device="mps", dtype=torch.bfloat16)
            vae = diffusers.AutoencoderKLWan.from_pretrained(
                str(local.vae.path), torch_dtype=torch.bfloat16, **common
            ).to(device="mps", dtype=torch.bfloat16)
            scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(local.scheduler.path), **common
            )
            return WanCompilerComponents(tokenizer, encoder, vae, scheduler)


def _derived_seed(seed: int, domain: str) -> int:
    digest = hashlib.sha256(f"{domain}\0{seed}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@contextmanager
def _fenced_rng(importer: Callable[[str], object], *, seed: int) -> Iterator[Any]:
    torch = cast(Any, importer("torch"))
    with _RNG_LOCK:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        mps_state = None
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
                try:
                    mps_state = torch.mps.get_rng_state().clone()
                except Exception:
                    mps_state = None
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                try:
                    torch.mps.manual_seed(seed)
                except Exception:
                    pass
            yield torch
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)


@contextmanager
def _private_snapshot(
    snapshot: WanCompilerSnapshot, limits: WanCompilerLimits
) -> Iterator[WanCompilerSnapshot]:
    root = Path(tempfile.mkdtemp(prefix="wan-compiler-")).resolve()
    try:
        roots: dict[str, WanPinnedLocalRoot] = {}
        for name in ("tokenizer", "text_encoder", "vae", "scheduler"):
            source = cast(WanPinnedLocalRoot, getattr(snapshot, name))
            roots[name] = secure_snapshot_pinned_root(
                source,
                root / name,
                limits=WanSnapshotCopyLimits(
                    max_files=limits.max_snapshot_files,
                    max_bytes=limits.max_snapshot_bytes,
                    max_depth=limits.max_snapshot_depth,
                ),
            )
        yield WanCompilerSnapshot(**roots)
    finally:
        for item in root.rglob("*"):
            if item.is_dir():
                os.chmod(item, 0o700, follow_symlinks=False)
            else:
                os.chmod(item, 0o600, follow_symlinks=False)
        for item in sorted(root.rglob("*"), reverse=True):
            item.rmdir() if item.is_dir() else item.unlink()
        root.rmdir()

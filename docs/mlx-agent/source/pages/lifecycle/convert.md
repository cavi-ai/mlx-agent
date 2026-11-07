# Convert

Convert quantizes a model to MLX (4, 5, 6 or 8 bits) as a receipt-tracked batch job, from a cached Hugging Face repo or from a GGUF file already on disk.

```bash
python3 scripts/mlx-agent convert start --repo <repo> --q-bits 4
python3 scripts/mlx-agent convert start --repo <repo> --q-bits 4 --confirm --preview-hash <hash>
python3 scripts/mlx-agent convert status
```

## Gates

- Source checkpoint already local: cached repository, explicit source path, or GGUF file. Conversion does not download it.
- Selected converter backend already installed (the default is `mlx_lm.convert`).
- Fresh output path (never overwrites).
- Conversion receipt checks and a shared compute lease prevent conflicting heavy jobs.

The preview renders the exact `mlx_lm.convert` argv before anything runs. `convert status` cross-checks the receipt against the live process and records the exit once (`done` when the output directory exists, `failed` otherwise).

MLX-LM, MLX-VLM, MLX-Audio and MLX-Embeddings conversions and GGUF files without a port accept 4, 5, 6 and 8 bits.
MLX-Video and mflux converters, exported music checkpoints and the GGUF ports
(DeepSeek-V4 DSpark, Qwen3.5) accept 4 and 8. Ports convert at the widths they were
checked at: Audio8 at 4 and 8, Laya at 6 and 8, TripoSG, TRELLIS.2 and Pixal3D at 8 and 32 (unquantized).
`intake resolve` reports the offered widths as `q_bits`, and `convert scan` reports each GGUF's
widths the same way; a vision projector or a drafter no port converts reports none.

Result lands at `<model>-MLX-<bits>bit` under the current directory by default.
Use `--out` to choose a persistent model folder. Inference commands take the
converted directory explicitly; serving and wiring apply only to compatible tasks.

## TripoSG image-to-3D

Install the `mlx-3d` backend, then download or select the original `VAST-AI/TripoSG`
Diffusers checkpoint. Its transformer, VAE and DINOv2 weights must all be present.
Conversion runs locally without importing model repository code. `--q-bits 32`
keeps full-precision weights and reproduces the PyTorch reference geometry;
`--q-bits 8` is smaller, but its meshes can contain stray surface fragments.

```bash
python3 scripts/mlx-agent convert start --repo VAST-AI/TripoSG --source-path /path/to/TripoSG --backend mlx-3d --model-type triposg --q-bits 32 --out /path/to/models/TripoSG-MLX-32bit --json
# Review the plan, then repeat with --confirm --preview-hash <hash>.
python3 scripts/mlx-agent convert mesh --path /path/to/models/TripoSG-MLX-32bit --image /path/to/object.png --out /path/to/output.glb --steps 50 --seed 42 --resolution 512 --max-faces 500000 --json
```

Use a foreground object on a white or transparent background. Background removal
is not performed. The output is an untextured GLB mesh. The result reports
generation time, loading time, vertex/face counts and peak MLX memory, plus
`stage_seconds` (`load`, `conditioning`, `sampling`, `decode_dense`,
`decode_refine`, `marching_cubes`, `decimation`, `export`) and `queries`, the
number of field points queried in the dense pass and in the refinement. The
output path must be new.

`--resolution` is the side of the final grid: 32, 64, 128, 256 or 512 (default
512). Geometry follows the upstream TripoSG octree extraction: a dense pass at
half the final side, then one refinement of the cells near the surface, so
higher resolutions query more points. `--max-faces` (default 500000, 1000 to
10000000) reduces a larger mesh with quadric edge collapse. The initial latent
noise is `numpy.random.default_rng(seed).standard_normal`, so a seed
reproduces the PyTorch reference noise. The upstream CUDA flash decoder is not
reproduced. TripoSG is not a chat server.

## TRELLIS.2 image-to-3D

TRELLIS.2 reads files from two other repositories: the sparse-structure decoder
from `microsoft/TRELLIS-image-large` and the DINOv3 image encoder from
`facebook/dinov3-vitl16-pretrain-lvd1689m`, a gated repository whose terms must
be accepted on Hugging Face first. `intake fetch --model-type trellis2` downloads
the shape checkpoints with those files, and `intake resolve` counts them in
`download_bytes`. `convert start` reads them from the Hugging Face cache and
refuses with `requirement_not_cached` when one is missing. `--q-bits 8`
quantizes the image encoder and the flow transformers; the decoders stay full
precision.

```bash
python3 scripts/mlx-agent intake fetch microsoft/TRELLIS.2-4B --model-type trellis2 --json
python3 scripts/mlx-agent convert start --repo microsoft/TRELLIS.2-4B --backend mlx-3d --model-type trellis2 --q-bits 32 --out /path/to/models/TRELLIS.2-MLX-32bit --json
# Review each plan, then repeat it with --confirm --preview-hash <hash>.
python3 scripts/mlx-agent convert mesh --path /path/to/models/TRELLIS.2-MLX-32bit --image /path/to/object.png --out /path/to/output.glb --json
```

`convert mesh` takes `--resolution` 512, 1024 (default) or 1536, and its steps
(12) and guidance (7.5) default to the checkpoint's sampler settings. At 512 one
shape flow runs. At 1024 and 1536 the 512 latent is decoded to its voxels, which
seed a second flow at the higher level; when they exceed 49152 tokens the level
drops by 128, down to 1024, and the result's `resolution` reports the level
used. The image is cropped to the pixels with alpha above 0.8; an opaque image
keeps its full frame and background removal is not performed. The output is an
untextured GLB mesh; `stage_seconds` reports `conditioning`,
`sparse_structure`, `shape_latent_512`, `shape_latent_high`, `decode`, `mesh`,
`decimation` and `export`.

## Pixal3D image-to-3D

Pixal3D conditions TRELLIS.2-style shape flows on image features projected onto
the voxel grid through an estimated camera. It reads the DINOv3 image encoder
from the gated `facebook/dinov3-vitl16-pretrain-lvd1689m`, the MoGe-2 camera
model from `Ruicheng/moge-2-vitl` and the NAF feature upsampler
`naf_release.pth` from the NAF GitHub release. `intake fetch --model-type
pixal3d` downloads all three with the shape checkpoints, and `convert start`
refuses with `requirement_not_cached` until they are present. `--q-bits 8`
quantizes the image encoder and the flow transformers; the decoders, MoGe-2 and
NAF stay full precision.

```bash
python3 scripts/mlx-agent intake fetch TencentARC/Pixal3D --model-type pixal3d --json
python3 scripts/mlx-agent convert start --repo TencentARC/Pixal3D --backend mlx-3d --model-type pixal3d --q-bits 32 --out /path/to/models/Pixal3D-MLX-32bit --json
# Review each plan, then repeat it with --confirm --preview-hash <hash>.
python3 scripts/mlx-agent convert mesh --path /path/to/models/Pixal3D-MLX-32bit --image /path/to/object.png --out /path/to/output.glb --json
```

`convert mesh` takes `--resolution` 1024 or 1536 (default); steps (12) and
guidance (7.5) default to the checkpoint's sampler settings. The image is
cropped to a square 1.1 times the object's larger side. MoGe-2 estimates the
horizontal field of view, reported in radians as `camera_angle_x`, which places
the camera for every stage. The sparse structure and the 512 shape flow run
first; their voxels seed the high-resolution flow, which drops by 128 down to
1024 while the voxels exceed 49152 tokens. The output is an untextured GLB mesh;
`stage_seconds` reports `camera`, `sparse_structure`, `shape_latent_512`,
`shape_latent_high`, `decode`, `mesh`, `decimation` and `export`.

## GGUF sources

Most local weights arrive as GGUF. `convert scan` inventories them; `convert start --gguf` converts one.

```bash
python3 scripts/mlx-agent convert scan
python3 scripts/mlx-agent convert scan --gguf-root ~/models --pending-only --json
python3 scripts/mlx-agent convert start --gguf ~/models/model-Q4_K_M.gguf --q-bits 4
```

`scan` is read-only and needs nothing but Python. It parses bounded GGUF headers under each `--gguf-root` (repeatable; well-known local model directories by default) and gives every file a status:

| Status | Meaning |
| --- | --- |
| `pending` | No MLX output found — the convert list. |
| `converted` | An MLX output exists; `evidence` says whether it matched by `provenance`, `receipt`, or `name`. |
| `shard` | A non-first shard of a split model; convert the `-00001-of-` file. |
| `companion` | A projector (`mmproj`) or similar sidecar, not convertible alone. |

A directory counts as an MLX output only when its `config.json` carries a `quantization` block or this tool left an `mlx-converter.json` marker, so ordinary PyTorch checkpoints in the cache are never mistaken for conversions. Cached snapshots of `mlx-community` repositories also count, including unquantized (bf16) ones.

Each output carries a `task` label and, except Strata profiles, a `backend` object: the backend that runs it for that task (`mlx-lm`, `mlx-vlm`, `mlx-audio`, …) and whether it is `installed`. A config without a `model_type` is matched by the folder's (or cached repository's) leading word, as the backends themselves do; Kokoro resolves to `mlx-audio` text to speech.

Duplicate groups use two labels. `exact` groups match a sampled content signature
or the same header structure and quantization; members outside `keep` count
toward `reclaimable_bytes`. This classification is not a full-file byte comparison.
The content signature uses size plus the first and last mebibyte, so different
interior bytes can share it. `variant` groups have different quantizations and
are informational, with no removal recommendation. Review identity and any
conversion provenance before deleting a copy. Scan only reports.

MLX model folders are grouped too (`format: "mlx"`). An `exact` MLX group holds
the same weight files at the same sizes and content signatures; without
signatures it also needs the same model name and bit width. `keep` is a
converter output first, then a Hugging Face cache snapshot (named by its
repository), then the oldest copy; cache revisions sharing blobs count as one
copy. `variant` MLX groups share a model name (without its parameter count)
and architecture at different bit widths. Every group carries a `group_id`.

Ollama stores at or below the scan roots (a folder holding `blobs` beside
`manifests` or `manifests-v2`) are read too and listed under `ollama`: name,
manifest `path`, `format` (`gguf` or `safetensors`), quantization and bytes. A
GGUF model joins the GGUF groups by its model blob; when it duplicates a file,
the Ollama copy is kept, since only Ollama removes its blobs. A safetensors
model joins the MLX `variant` groups through its `config.json` layer. Models
listed in both manifest trees count once, and models sharing a blob are one copy.

Conversion runs two steps in one job: `transformers` dequantizes the GGUF back to Hugging Face weights, then `mlx_lm.convert` quantizes those to MLX. The intermediate checkpoint is full precision and removed afterwards, so budget roughly twice the source model's fp16 size in free disk while the job runs. It needs `torch`, `transformers`, and `gguf` importable by the same interpreter; missing pieces are named and never installed for you.

Qwen3.5-family GGUFs (`qwen35`, including Qwen3.6 and Qwen3.8 checkpoints) convert through a bundled port instead, which needs only `gguf` and `mlx` and writes no intermediate checkpoint. The port writes the MLX-VLM `qwen3_5_text` layout, drops the multi-token-prediction block and rebuilds the tokenizer from the GGUF vocabulary. When the GGUF's folder holds exactly one Qwen3-VL projector (`mmproj`) that matches the model's width, the plan names it and the output becomes the `qwen3_5` vision-language model. These outputs are served through the `mlx-vlm` backend. A GGUF holding only a multi-token-prediction block is a speculative-decoding drafter and is refused at plan time.

`convert scan` labels speculative-decoding drafters (`dflash` and `eagle3` GGUFs, multi-token-prediction-only GGUFs, configs with `dspark_target_layer_ids`) `speculative_draft` and reports their `draft` target. A DeepSeek-V4 DSpark `dflash` GGUF converts through the bundled `deepseek_v4_dspark` port to an MLX drafter that runs only beside its target; other drafters are refused at plan time (`unsupported_draft`).

Quality is capped by the source: a Q4 GGUF converted to MLX 4bit has been quantized twice. Convert the original fp16 weights when you have them.

The output carries `mlx-converter.json` recording its GGUF source, so later
scans can associate the output with that source using provenance evidence.

## Other checkpoints and backends

Use [Intake](intake.md) to identify the model type and required backend.
`--backend` and `--model-type` select a declared converter; install an optional
backend through a separately reviewed [backend plan](backends.md). Use
`--subfolder` for a cached checkpoint folder, or `--source-path` for the exact
downloaded directory associated with `--repo`. Conversion does not fetch missing
weights. GGUF conversion uses `--gguf` rather than an optional backend override.

After conversion finishes, use [Serve](serve.md) for a compatible HTTP model or
[Local inference](local-inference.md) for supported speech, classification,
image, and video tasks. Inspect status and logs before relying on outputs.

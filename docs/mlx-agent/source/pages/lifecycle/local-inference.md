# Local inference

Task-specific commands run compatible local model directories through declared
backends. Choose a model with [Intake](intake.md), install its [backend](backends.md)
if needed, and finish [conversion](convert.md) before running these commands.

These commands execute directly; they do not use the preview/confirmation contract
of downloads and lifecycle starts. Review inputs, output paths, and resource needs.

For retained classification and media results, including generated meshes,
and cancellation across client exits,
use [Durable inference tasks](tasks.md).

## Speech

```bash
python3 scripts/mlx-agent convert transcribe --path /absolute/local/asr-model --audio ./recording.wav --json
python3 scripts/mlx-agent convert speak --path /absolute/local/tts-model --text "Hello." --out /absolute/output/speech.wav --json
```

Transcription accepts an audio file and optional `--language`. Speech output
requires a new absolute `.wav` path; voice, speed, and language depend on the model.
Speech generation has a 600-second execution timeout. For longer runs, pass
`--timeout` to `convert speak`; it accepts 1 to 7,200 seconds.

## Music

```bash
python3 scripts/mlx-agent convert music --path /absolute/local/music-model --caption "A gentle piano melody" --lyrics '[instrumental]' --out /absolute/output/music.wav --duration 15 --steps 30 --seed 42 --json
```

The isolated audio backend supports MiniMax Music 3 from local weights. Duration
is a maximum request greater than zero and up to 360 seconds; steps range from
1 to 30. Lyrics may include section tags. Output is a new mono or stereo PCM
WAV. Reported duration, generation/load time, real-time factor, and memory do
not measure listening quality.

The checkpoint needs a real local tokenizer at `tokenizer/tokenizer.json` or
root `tokenizer.json`; a synthetic tokenizer fallback is refused. Already-exported
quantized checkpoints can be converted from a local `--source-path`, using the
model's layer policy while preserving tokenizer and scheduler assets.

Supported Music3 layouts automatically reuse causal depth-prefix work and
repeated prompt encodings. Retained workers can reuse caption/lyric encodings
across jobs. These paths preserve sampling controls and checkpoint precision;
floating-point changes can produce different audio from the same seed.

The `ace-step` backend runs ACE-Step 1.5 checkpoints such as `acestep-v15-turbo`
and `acestep-v15-xl-sft`. Pass the DiT folder inside the ACE-Step `checkpoints/`
folder; `vae` and `Qwen3-Embedding-0.6B` must sit beside it. When an
`acestep-5Hz-lm-*` planner sits there too, it plans the song's structure and
metadata before the DiT renders it. Turbo checkpoints take 1 to 20 steps (8 is
typical) at shift 3; the others take up to 64 steps at guidance 7.

```bash
python3 scripts/mlx-agent convert music --path /absolute/ACE-Step-1.5/checkpoints/acestep-v15-turbo --caption "minimal tech house, four-on-the-floor kick" --out /absolute/output/music.wav --duration 120 --steps 8 --seed 7 --bpm 120 --time-signature 4 --json
```

`--bpm` (30 to 300), `--time-signature` (2, 3, 4 or 6 beats per bar) and
`--reference` (an absolute audio file whose style the take follows) apply to
ACE-Step checkpoints only. ACE-Step loads from a temporary folder of links to the
checkpoint files, so the model folder is never written.

## Vision and classification

```bash
python3 scripts/mlx-agent convert describe --path /absolute/local/vision-model --image ./image.png --prompt "Describe this image." --json
python3 scripts/mlx-agent convert describe --path /absolute/local/vision-model --video ./clip.mp4 --prompt "Summarize this clip." --max-tokens 256 --json
python3 scripts/mlx-agent convert decide --path /absolute/local/classification-model --request ./request.json --json
```

Describe supports image or video input, bounded answer length, and media sampling
controls. Decide reads a JSON request with `state` and `questions` fields for a
compatible classification model. A text-generation model directory is not
interchangeable with these task-specific models.

## 3D generation

`convert mesh` generates a GLB from a local PNG, JPEG or WebP image through
TripoSG, TRELLIS.2 or Pixal3D. Supported resolutions and defaults come from the
converted model configuration. Use `--steps`, `--seed`, `--resolution`,
`--guidance` and `--max-faces` to select generation and export settings.

TripoSG automatically bounds point-query batches and compiles VAE query
execution for each generation. Fixed decoder conditioning is prepared once;
compiled queries retain no conditioning across jobs. Weights, sampling steps,
guidance and requested resolution are unchanged.

TripoSG defaults to 50 steps, resolution 512 and guidance 7. Adaptive denoiser
caching is optional and off by default:

```bash
python3 scripts/mlx-agent convert mesh --path /absolute/local/mesh-model --image /absolute/input/object.png --out /absolute/output/object.glb --denoiser-cache adaptive --cache-memory-mib 256 --json
```

Caching may change geometry. The feature budget is separate from model memory;
jobs above it use full computation. Adaptive mode requires TripoSG; other mesh
architectures use `--denoiser-cache off`. JSON metadata reports full/reused
steps and the feature-memory reservation. No sampling steps are removed.

## Image and video generation

Wan2.1 T2V supports optional adaptive denoiser caching. Add
`--denoiser-cache adaptive` to reuse transformer residuals during sufficiently
stable steps. This approximation can change generated frames. Caching is off
by default. `--cache-memory-mib 256` sets the retained-feature budget; jobs
that exceed it use full computation. JSON results include `denoiser_cache`
statistics with full and reused step counts, reserved bytes and refresh reasons.
For example, a short Wan2.1 clip can use
`--width 320 --height 192 --frames 33 --denoiser-cache adaptive --cache-memory-mib 256`.
Higher resolutions or frame counts need a larger feature budget to enable reuse.

```bash
python3 scripts/mlx-agent convert generate --path /absolute/local/image-model --prompt "A green apple on a table" --out /absolute/output/image.png --json
python3 scripts/mlx-agent convert video --path /absolute/local/video-model --prompt "A green apple rolling across a table" --out /absolute/output/video.mp4 --json
```

Use a new absolute output path ending in `.png` or `.mp4`. Width, height, steps,
frame counts, and task support depend on the backend/model. Command help gives
bounds and defaults:

```bash
python3 scripts/mlx-agent convert video --help
python3 scripts/mlx-agent convert describe --help
```

Use `--timeout` to bound execution. Keep sensitive inputs and generated media
private. Synthetic adoption probes do not describe the privacy properties of
user-supplied inference inputs or their logs.

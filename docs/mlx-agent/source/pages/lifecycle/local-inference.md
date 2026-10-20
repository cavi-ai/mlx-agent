# Local inference

Task-specific commands run compatible local model directories through declared
backends. Choose a model with [Intake](intake.md), install its [backend](backends.md)
if needed, and finish [conversion](convert.md) before running these commands.

These commands execute directly; they do not use the preview/confirmation contract
of downloads and lifecycle starts. Review inputs, output paths, and resource needs.

For retained classification/music results and cancellation across client exits,
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

## Image and video generation

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

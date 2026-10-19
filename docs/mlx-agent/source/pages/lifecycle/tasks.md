# Durable inference tasks

Use `task` for classification, music generation, transcription, speech
synthesis, image generation and image understanding that must outlive the
calling process. The engine retains accepted inputs, status, results and output
files.
CLI and MCP clients use the same task IDs and storage. Existing synchronous
`convert decide` and `convert music` commands remain available.

Tasks require an existing compatible local model and its installed backend.
They do not download weights or install a runtime. Inspect supported kinds with:

```bash
python3 scripts/mlx-agent task capabilities --json
```

Capabilities describe the contract; they do not certify a selected model's
readiness, output quality or memory fit.

## Start a task

Save a UTF-8 JSON file using schema `task-request/1`. Classification uses the
same `state` and typed `questions` content as `convert decide`:

```json
{
  "schema": "task-request/1",
  "kind": "classification",
  "model_path": "/absolute/local/classification-model",
  "parameters": {
    "state": "A customer requests a refund for a duplicate charge.",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": ["billing", "technical support"]
      }
    }
  }
}
```

Music uses kind `music_generation`:

```json
{
  "schema": "task-request/1",
  "kind": "music_generation",
  "model_path": "/absolute/local/music-model",
  "parameters": {
    "caption": "A quiet piano instrumental",
    "lyrics": "[instrumental]",
    "duration": 15,
    "steps": 30,
    "seed": 42
  }
}
```

Music defaults to instrumental lyrics, 15 seconds, 30 steps and seed 42 when
those settings are omitted. Duration is greater than zero and at most 360
seconds, steps are 1–30, and seed is an unsigned 32-bit integer. Caption and
lyrics limits are 2,000 and 10,000 characters. Requests are limited to 1 MiB.

Speech tasks use `task-request/2`. Transcription accepts an absolute WAV, FLAC
or MP3 path and an optional lowercase language code. The engine copies the
audio into private task storage before accepting the task; later execution
uses and verifies that snapshot even if the original file changes or disappears.

```json
{
  "schema": "task-request/2",
  "kind": "transcribe",
  "model_path": "/absolute/local/speech-model",
  "parameters": {"audio_path": "/absolute/audio.wav", "language": "en"}
}
```

Speech synthesis accepts text and optional backend-supported `voice`, `speed`
and `lang_code` values. Omitted options retain the operation's own defaults.
Successful synthesis publishes a WAV only after the engine validates that it
contains readable audio.

```json
{
  "schema": "task-request/2",
  "kind": "speak",
  "model_path": "/absolute/local/speech-model",
  "parameters": {"text": "Hello from a local speech model."}
}
```

Image tasks also use `task-request/2`. Image generation accepts a prompt and
optional `width`, `height`, `steps` and `seed`; omitted values use the existing
image operation defaults. The operation validates its supported ranges. A
successful task retains `output.png` only after the selected backend's Pillow
runtime confirms the PNG can be fully decoded.

```json
{
  "schema": "task-request/2",
  "kind": "image_generation",
  "model_path": "/absolute/local/image-model",
  "parameters": {"prompt": "A quiet forest at dawn"}
}
```

Image understanding accepts an absolute PNG, JPEG or WebP `image_path`, a
prompt, and optional `max_tokens`, `temperature` and `max_pixels`. The engine
copies the image into private task storage before accepting the task and uses
that verified snapshot during inference. The result retains the full
`describe/1` text and the exact input artifact reference.

```json
{
  "schema": "task-request/2",
  "kind": "image_understanding",
  "model_path": "/absolute/local/vision-model",
  "parameters": {"image_path": "/absolute/photo.webp", "prompt": "Describe the scene."}
}
```

```bash
python3 scripts/mlx-agent task start --request /absolute/request.json --json
python3 scripts/mlx-agent task status --id TASK_UUID --json
python3 scripts/mlx-agent task list --limit 20 --json
python3 scripts/mlx-agent task result --id TASK_UUID --json
python3 scripts/mlx-agent task cancel --id TASK_UUID --json
python3 scripts/mlx-agent task verify-model --id TASK_UUID --json
```

Replace `TASK_UUID` with the returned `data.task.id`. Start captures the request
privately before returning. The task supervisor uses the existing host compute
lease: busy compute becomes a retained, actionable failure; tasks do not queue.
Status and history never launch inference. History is newest first, accepts a
limit of 1–100, and reports `has_more` when additional records exceed the limit
or response size bound.

## State, cancellation and results

States are `starting`, `running`, `succeeded`, `failed` and `cancelled`.
Cancellation records intent first. The supervisor sends TERM to its owned
execution group, waits up to five seconds, then uses KILL if necessary and
waits up to five more seconds. `cancelled` requires confirmation that the group
stopped. Uncertain process ownership produces an error without signaling an
unrelated process. A lost supervisor is reported as interrupted; unresolved
execution continues to block host compute. Inspect that error before retrying.

Results preserve the underlying `decide/1`, `music/1`, `transcribe/1`,
`speak/1`, `generate/1` or `describe/1` metadata. Classification answers and
image descriptions are model judgments. Successful music and speech synthesis
confirm an output WAV; image generation confirms a fully decodable PNG. Tasks
do not rate output quality. Generated media artifacts have a content hash, and
retrieval refuses a missing or changed output. Task completion does not confer
a conversion verification or comparison award.

`result` includes an owned JSON artifact reference with path, SHA-256 and byte
size. Small results are also returned inline; results over 8,192 bytes use
`inline: false`. Command responses are bounded to 12,288 bytes; fewer history
rows or an artifact reference keep the response valid JSON. Read the referenced
local file for a larger result. Request text and process command lines are not
included in task summaries.

Before reusing a retained request, `verify-model` checks the recorded model,
backend manifest and executable identity against their current local values.
Success returns the task ID, model path, recorded fingerprint and `matches: true`.
A changed or missing dependency returns a structured error. This read-only check
does not start inference, claim compute, or change the task record. Historical
status and results remain accessible when the model has changed or disappeared.
Verification describes the files at check time; a later Start captures its own
current identity and checks it again during execution.

## Agent access through MCP

The existing `mlx_agent_execute` tool accepts capability `task`. Pass a task
command in `arguments`; the narrow grammar always requests JSON output:

```json
{
  "name": "mlx_agent_execute",
  "arguments": {
    "capability": "task",
    "arguments": "start --request /absolute/request.json --state-dir /absolute/task-state"
  }
}
```

Use the same capability for `status`, `list`, `result`, `cancel` and `verify-model`, with the
returned task ID. Prompt content stays inside the request file. Quote paths
containing spaces; MCP paths must be absolute and obey its existing bounded
argument grammar. Large results use the same local artifact references as CLI
calls, so the MCP response does not contain truncated result JSON.

## Local storage

Default storage is `~/Library/Application Support/mlx-agent/tasks` on macOS and
`${XDG_STATE_HOME:-~/.local/state}/mlx-agent/tasks` elsewhere. `tasks.sqlite3`
holds records; `artifacts/<UUID>/` holds that task's request, logs and outputs.
Directories and files are private to the current user. Task storage refuses
symlink traversal. Accepted input content is retained only after explicit Start;
the caller's original request and model files are not changed.

Every task command accepts an absolute `--state-dir`. Pass the same value from
each client to share the same history. It does not create a separate host compute
lease. The app and other clients must use commands rather than edit the database.

There is no automatic deletion of task history or artifacts, and these outputs
are independent of comparison caches. Keep sensitive input files, task snapshots,
logs and generated media private. Copy a chosen output explicitly when exporting.

Model provenance records file identities and hashes of small JSON files, plus
backend manifest and executable identity. The engine checks these before work
and before success. This metadata fingerprint does not hash every weight byte
or promise identical outputs across runs. Unavailable runtime version facts are
left absent.

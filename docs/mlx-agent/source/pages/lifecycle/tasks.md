# Durable inference tasks

Use `task` for classification and music generation that must outlive the calling
process. The engine retains accepted inputs, status, results and output files.
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

```bash
python3 scripts/mlx-agent task start --request /absolute/request.json --json
python3 scripts/mlx-agent task status --id TASK_UUID --json
python3 scripts/mlx-agent task list --limit 20 --json
python3 scripts/mlx-agent task result --id TASK_UUID --json
python3 scripts/mlx-agent task cancel --id TASK_UUID --json
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

Results preserve the underlying `decide/1` or `music/1` metadata. Classification
answers are model judgments. Successful music execution confirms an output WAV;
it does not rate listening quality. Music artifacts have a content hash, and
retrieval refuses a missing or changed output. Task completion does not confer
a conversion verification or comparison award.

`result` includes an owned JSON artifact reference with path, SHA-256 and byte
size. Small results are also returned inline; results over 8,192 bytes use
`inline: false`. Command responses are bounded to 12,288 bytes; fewer history
rows or an artifact reference keep the response valid JSON. Read the referenced
local file for a larger result. Request text and process command lines are not
included in task summaries.

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

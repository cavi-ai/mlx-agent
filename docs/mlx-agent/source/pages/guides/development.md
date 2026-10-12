# Development

The core runs from a checkout with Python 3.9+ and the standard library. Execute
it with `python3 scripts/mlx-agent`; a model runtime is needed only for commands
that run models. Development validators require `requirements-dev.txt`.

## Repository layout

| Path | Responsibility |
| --- | --- |
| `src/mlx_agent/` | CLI, discovery, verification, transactions, and model lifecycle |
| `src/mlx_agent/resources/` | Backend manifests and locks, runtime recipes, probe assets, reference packs |
| `plugin.json`, `skills/`, `commands/`, `agents/` | Canonical provider capabilities and instructions |
| `providers/` | Generated native plugins and portable skill definitions |
| `schemas/` | Result, state, receipt, and integration contracts |
| `tests/` | Unit, integration, and contract tests |
| `docs/mlx-agent/source/` | Documentation navigation and publishable pages |

The legacy `skills/mlx-scout/scripts/scout.py` launcher wraps the shared core.
Use `scripts/mlx-agent` for new CLI examples. Installed AgentSkills packages
carry their own runtime; Codex and Agy packages share one runtime at their
provider package root.

## Local checks

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
PYTHONPATH=src .venv/bin/python scripts/validate_contracts.py
PYTHONPATH=src .venv/bin/python scripts/validate_json_schemas.py
PYTHONPATH=src .venv/bin/python scripts/generate_adapters.py --check
PYTHONPATH=src .venv/bin/python scripts/render_compatibility.py --check
PYTHONPATH=src .venv/bin/python scripts/validate_release_version.py
git diff --check
```

CI also compiles core Python sources, excluding the optional `_lab` engine,
and tests on Linux and macOS with Python 3.9 and 3.12. The separate optional
engine job uses Python 3.12 and its own dependency/test setup. See
[Experiments and training](../lifecycle/laboratory.md). Fixture tests establish contracts; they do
not establish model behavior on a real host.

The opt-in release live smoke automatically selects only direct local Ollama,
LM Studio, or `mlx_lm` backends. LiteLLM is excluded from automatic live selection
because its loopback endpoint can route to remote or paid backends. Use it for
a live check only after establishing that the selected route stays local.

## Check documentation artifacts

From the repository root, choose a temporary staging destination:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_docs.py --destination .tmp/docs-check
PYTHONPATH=src .venv/bin/python scripts/build_docs.py --check --destination .tmp/docs-check
PYTHONPATH=src .venv/bin/python -m unittest tests.contracts.test_docs tests.contracts.test_docs_artifact
```

The build copies source pages and navigation into an artifact with a content
digest. `--check` checks its file inventory, digest, and provenance shape; it
does not compare each page's bytes with the current source. Rebuild the staging
directory after editing pages. Release builds pass the tagged commit explicitly
with `--commit`; a local build is not evidence that a release has been published.

See [Version and support](../release/version-and-support.md) for distribution.

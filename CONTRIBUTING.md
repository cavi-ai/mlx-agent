# Contributing

Use Python 3.11 or later for the provider-neutral core. Model execution requires
Apple Silicon and the appropriate runtime; optional backend locks target Python
3.12. See [Development](docs/mlx-agent/source/pages/guides/development.md) for the
repository layout and documentation workflow.

## Validate a change

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
PYTHONPATH=src .venv/bin/python -m compileall -q -x '/_lab/' src skills scripts tests
PYTHONPATH=src .venv/bin/python scripts/validate_release_version.py
PYTHONPATH=src .venv/bin/python scripts/validate_contracts.py
PYTHONPATH=src .venv/bin/python scripts/validate_json_schemas.py
PYTHONPATH=src .venv/bin/python scripts/generate_adapters.py --check
PYTHONPATH=src .venv/bin/python scripts/render_compatibility.py --check
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
git diff --check
```

Adapters are generated from `plugin.json`, the canonical skills, and the shared
runtime. Update their sources, run `scripts/generate_adapters.py`, then check
the result. The compatibility table in the README is generated from
`compatibility/providers.json`; use `scripts/render_compatibility.py --write`
after changing the matrix.

## Documentation

Edit pages under `docs/mlx-agent/source/pages` and register them in
`docs/mlx-agent/source/navigation.json`. Keep examples aligned with CLI help and
preserve links from the [documentation index](docs/README.md). Previously
published `docs/mlx-agent/v*/` directories are release artifacts; edit source
pages instead. Build into a separate staging directory when checking a change.

## Pull requests

`main` accepts signed commits whose four Test matrix jobs and experiment-engine
job pass. Describe the behavior changed and the relevant validation. Include a regression
test for behavior changes. Keep model weights, credentials, local receipts,
generated research packs, and temporary outputs out of the change. Report bugs
with the command, package/runtime versions, and a redacted error; inspect logs
and receipts before attaching them.

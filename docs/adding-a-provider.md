# Add a provider

1. Add the provider to canonical `plugin.json` with the shared `scout`, `adopt`, and `wire` capability mapping, user/project roots, artifacts, and the documented invocation form.
2. Extend `src/mlx_agent/providers.py` only when that host needs a layout invariant. Keep the installer receipt-owned; do not mutate the host's unrelated configuration or install its CLI.
3. Teach `scripts/generate_adapters.py` to render deterministic artifacts and inventory every generated file. Run `PYTHONPATH=src python3 scripts/generate_adapters.py --check`.
4. Write adapter and installer contracts first, including hostile transport input, user/project isolation, update/uninstall, and an unrelated working-directory bundle launch.
5. Add an entry to `compatibility/providers.json` with `scopes`, `config_paths`, a Scout/Adopt/Wire `capabilities` invocation mapping, minimum/last-tested versions, and `last_smoke_test` `{status, date, summary}`. Record five separate evidence statuses: schema, install round trip, native discovery, bundle execution, and model-backed invocation. Use `not-run` or `blocked` honestly; neither is a support claim. Run `python3 scripts/render_compatibility.py --write`, then check the generated README block back in.
6. Add an install guide and link it from the README. Update the manual/scheduled smoke workflow only for automation that the host's licensing and interaction model permit.

Finish with the full unittest discovery suite, contract validator, generator drift check, whitespace check, and a real Python 3.9 CI matrix job.

# Hermes Agent

Hermes loads each installed skill as `/mlx-scout`, `/mlx-adopt`, `/mlx-wire`, `/mlx-bench`, `/mlx-doctor`, `/mlx-watch`, and `/mlx-fleet`. Every skill folder carries its own launcher and Python runtime.

Use the receipt-owned installer; it never edits `config.yaml`:

```bash
python3 scripts/mlx-agent install hermes --scope user --dry-run --json
python3 scripts/mlx-agent install hermes --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent update hermes --scope user --dry-run --json
python3 scripts/mlx-agent uninstall hermes --scope user --dry-run --json
python3 scripts/mlx-agent doctor hermes --scope user --json
```

User scope installs into `$HERMES_HOME/skills` when `HERMES_HOME` is set, otherwise `~/.hermes/skills`. Each Hermes profile has its own home: set `HERMES_HOME` to the profile directory to install for that profile. Project scope installs into `<project>/.hermes/skills`:

```bash
python3 scripts/mlx-agent install hermes --scope project --project /absolute/project/path --dry-run --json
```

Hermes releases with project skill discovery load `<project>/.hermes/skills` only from trusted repositories, after `hermes skills trust`; Hermes 0.19.1 reads only `$HERMES_HOME/skills`. Run `hermes skills list` and confirm the seven `mlx-*` skills appear. `doctor hermes` runs `hermes --version`, which also runs the Hermes update check.

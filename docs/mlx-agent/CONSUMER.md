# Consuming the mlx-agent documentation artifact

`docs/mlx-agent/v<version>/` is an immutable, self-validating documentation artifact in the bobby-browser pattern.

## Contents

- `manifest.json` — `{package, product, version, contentSha256, publicBasePath, stableAlias, release}`.
- `navigation.json` — `{title, version, sections[]}`; `version` equals the manifest version and every navigation path exists in the artifact.
- One markdown page per navigation entry.

`release` records `{tag: "v<version>", commit}` — the exact tagged repository commit the release workflow built from.

## Contract for hosts

1. Copy the complete `v<version>/` directory; never edit generated pages.
2. Validate: navigation version equals manifest version; every navigation target exists; `contentSha256` equals sha256 over every file except `manifest.json`, in lexical order, each entry as `path \0 bytes \0`.
3. Serve the artifact at `publicBasePath` and the newest artifact at `stableAlias`.

## Regenerating

```bash
python3 scripts/build_docs.py          # build a local artifact from docs/mlx-agent/source
```

The artifact version follows `mlx_agent.__version__`; the tag workflow builds it in staging with the exact tagged commit before packaging. `tests/contracts/test_docs_artifact.py` enforces integrity in CI.

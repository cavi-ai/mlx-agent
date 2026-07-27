# Fleet

Fleet wires an entire per-role routing configuration in one transaction.

```bash
python3 scripts/mlx-agent fleet render --path ./router.yaml \
  --assign coding=mlx-community/Qwen3-32B-4bit \
  --assign vision=mlx-community/Qwen3-VL-8B-Instruct-4bit
python3 scripts/mlx-agent fleet apply --path ./router.yaml --from-adoption ./adopt.json
python3 scripts/mlx-agent fleet apply --path ./router.yaml --from-adoption ./adopt.json --confirm --preview-hash <hash>
```

## Inputs

- `--assign role=repo` (repeatable) for explicit picks, or
- `--from-adoption <state.json>` to consume a completed adopt handoff's recommendations.

Roles are canonical (`general`, `coding`, `reasoning`, `vision`, `embedding`, `tool-use`); each at most once. Vision defaults to `mlx-vlm` (:8083), text roles to `mlx_lm` (:8080); `--runtime-map role=runtime` overrides.

## Guarantees

- Renders one bounded LiteLLM router YAML with a managed header and an exact five-line entry grammar per role.
- Every referenced model is checked against local inventories; `--allow-missing` records a reviewed warning instead of failing.
- Applies through the identical preview-confirm-receipt-rollback transaction as wire.

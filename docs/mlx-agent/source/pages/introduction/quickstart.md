# Quickstart

Five minutes from install to a verified, wired local model.

## 1. Discover

Ask your agent, or run the core directly:

```bash
python3 scripts/mlx-agent discover --role coding --limit 3
```

You get candidates bucketed by role with real download sizes, license and gating, reasoning detection, and whether weights fit your RAM. Add `--context 32768` to tighten fit to weights plus KV cache at that context.

## 2. Verify

Run the resumable adoption workflow:

```bash
python3 scripts/mlx-agent adopt start --state ./adopt.json --role coding
python3 scripts/mlx-agent adopt resume --state ./adopt.json
```

Adopt shortlists, probes each candidate against your local runtime (deterministic coding/reasoning/vision/embedding/tool-use probes), and recommends with evidence. Add `--measure` to bench verified candidates before ranking.

## 3. Measure

```bash
python3 scripts/mlx-agent bench run --repo <repo> --runtime mlx_lm
```

Reports TTFT, decode/prefill tok/s, and run spread for a model already served locally. Never starts servers, never downloads.

## 4. Wire

```bash
python3 scripts/mlx-agent wire render <repo> --target mlx_lm --path ./provider.json
python3 scripts/mlx-agent wire apply <repo> --target mlx_lm --path ./provider.json
# inspect the diff, then:
python3 scripts/mlx-agent wire apply <repo> --target mlx_lm --path ./provider.json --confirm --preview-hash <hash>
```

Every mutation is a reviewed transaction with a receipt and byte-exact rollback:

```bash
python3 scripts/mlx-agent wire rollback <receipt>
```

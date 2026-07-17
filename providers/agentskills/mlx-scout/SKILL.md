---
name: mlx-scout
description: Discover MLX models suitable for the current host.
---

# MLX Scout

canonical capability ID: mlx-agent.scout

Run the provider-neutral discovery command:

`python3 ../../../scripts/mlx-agent discover <arguments>`

Present its evidence and recommendations as returned. Discovery must not download model weights or change configuration. If a later download or configuration mutation would help, describe the exact CLI preview first and obtain explicit user confirmation before it.

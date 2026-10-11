"""Optional MLX inference child; invoked only for reviewed local evaluations."""
import json
from pathlib import Path
import sys

def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    variant = sys.argv[2]
    if variant not in ("baseline", "adapter"):
        raise ValueError("Unknown evaluation variant.")
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    mx.random.seed(plan["seed"])
    model, tokenizer = load(plan["base_path"], adapter_path=plan["adapter_path"] if variant == "adapter" else None)
    text = generate(model, tokenizer, prompt=plan["prompt"], max_tokens=plan["max_tokens"],
                    sampler=make_sampler(temp=plan["temperature"]), verbose=False)
    Path(sys.argv[3]).write_text(json.dumps({"text": text}))

if __name__ == "__main__":
    main()

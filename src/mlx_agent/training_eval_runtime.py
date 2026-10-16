"""Optional MLX inference child; invoked only for reviewed local evaluations."""
import json
from pathlib import Path
import sys

def generate_outputs(plan, variant, load, generate, make_sampler, seed):
    if variant not in ("baseline", "adapter"):
        raise ValueError("Unknown evaluation variant.")
    model, tokenizer = load(plan["base_path"], adapter_path=plan["adapter_path"] if variant == "adapter" else None)
    prompts = plan["prompt_suite"]["prompts"] if "prompt_suite" in plan else [
        {"id": "prompt", "text": plan["prompt"], "max_tokens": plan["max_tokens"]}]
    samples = []
    for entry in prompts:
        seed(plan["seed"])
        text = generate(model, tokenizer, prompt=entry["text"], max_tokens=entry["max_tokens"],
                        sampler=make_sampler(temp=plan["temperature"]), verbose=False)
        samples.append({"id": entry["id"], "text": text})
    return {"samples": samples} if "prompt_suite" in plan else {"text": samples[0]["text"]}


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    variant = sys.argv[2]
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    outputs = generate_outputs(plan, variant, load, generate, make_sampler, mx.random.seed)
    Path(sys.argv[3]).write_text(json.dumps(outputs))

if __name__ == "__main__":
    main()

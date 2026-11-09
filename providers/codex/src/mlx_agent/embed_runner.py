"""Generate text embeddings using the backend's trained pooling and normalization."""
import argparse
import json
import os
from pathlib import Path
import time

if __package__:
    from .runner_model_cache import load_model
else:
    from runner_model_cache import load_model


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--request', required=True)
    args = parser.parse_args(argv)
    model_path = Path(args.model)
    if not model_path.is_absolute() or not model_path.is_dir():
        raise ValueError('Embedding inference requires existing local weights.')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    from mlx_embeddings import load, generate
    parameters = json.loads(Path(args.request).read_text())
    started = time.perf_counter()
    model, tokenizer = load_model(load, str(model_path), tokenizer_config={'local_files_only': True, 'trust_remote_code': False})
    loaded = time.perf_counter()
    result = generate(model, tokenizer, texts=parameters['input'], max_length=parameters.get('max_length', 512))
    # Models supply their own trained pooling; an unsupported output is refused.
    vectors = result.text_embeds.tolist()
    print(json.dumps({'embeddings': vectors, 'load_seconds': loaded - started,
                      'seconds': time.perf_counter() - loaded}, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

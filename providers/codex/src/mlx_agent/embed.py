"""Text embeddings from an existing local checkpoint in the pinned embedding backend."""
import json
import math
from pathlib import Path
import subprocess

from .backends import backend_python, is_installed, load_manifests, load_registries, lookup, normalize_type
from .inference_runner import InferenceError, RunnerContract, invoke

TEXT_MODELS = frozenset(('bert', 'modernbert', 'qwen3', 'xlm_roberta', 'lfm2', 'llama_bidirec', 'gemma3_text'))


class EmbedError(InferenceError):
    pass


def validate_parameters(parameters):
    texts = parameters.get('input')
    length = parameters.get('max_length', 512)
    if (set(parameters) - {'input', 'max_length'} or not isinstance(texts, list) or not 1 <= len(texts) <= 128
            or any(not isinstance(text, str) or not text.strip() or len(text) > 32768 for text in texts)
            or type(length) is not int or not 1 <= length <= 8192):
        raise EmbedError('invalid_request', 'Embedding input needs 1–128 nonempty texts and max_length from 1 to 8192.',
                         'Provide a bounded input array of text strings.')
    return parameters


def validate_vectors(vectors, count):
    if not isinstance(vectors, list) or len(vectors) != count or not vectors or not isinstance(vectors[0], list) or not vectors[0]:
        raise ValueError('The embedding batch shape is invalid.')
    width = len(vectors[0])
    if any(not isinstance(row, list) or len(row) != width or
           any(type(value) not in (int, float) or not math.isfinite(value) for value in row) for row in vectors):
        raise ValueError('Embeddings must be a finite rectangular matrix.')
    return [[float(value) for value in row] for row in vectors]


def plan_embed(model_path, request_path, manifests=None, registries=None, root=None):
    model, request = Path(model_path), Path(request_path)
    try:
        if not model.is_absolute() or not model.is_dir():
            raise ValueError('An existing absolute local model directory is required.')
        config = json.loads((model / 'config.json').read_text())
        parameters = validate_parameters(json.loads(request.read_text()))
        if normalize_type(config.get('model_type')) not in TEXT_MODELS:
            raise ValueError('No vetted text embedding adapter implements this architecture.')
    except (OSError, ValueError, AttributeError) as error:
        raise EmbedError('model_unreadable', str(error), 'Select a supported local text embedding checkpoint.') from error
    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests, root) if registries is None else registries
    if not any(hit['category'] == 'embedding' for hit in lookup(config.get('model_type'), manifests, registries)):
        raise EmbedError('not_embedding_model', 'The local architecture has no embedding adapter.', 'Select a supported text embedding model.')
    manifest = manifests['mlx-embeddings']
    if not is_installed(manifest, root):
        raise EmbedError('backend_not_installed', 'The embedding backend is not installed.', 'Install mlx-embeddings first.')
    return {'backend': manifest['id'], 'model': str(model), 'request': str(request), 'count': len(parameters['input']),
            'argv': [str(backend_python(manifest, root)), str(Path(__file__).with_name('embed_runner.py')),
                     '--model', str(model), '--request', str(request)]}


CONTRACT = RunnerContract(verb='embed', error=EmbedError,
    accept=lambda value, plan: isinstance(value.get('embeddings'), list), action='embed the text',
    remediation='Check the local embedding checkpoint and backend.', activity='Text embedding',
    timeout_remediation='Use a smaller batch.', environment={'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})


def run_embed(plan, manifests=None, root=None, timeout=300, runner=subprocess.run):
    result = invoke(plan, CONTRACT, manifests, root, timeout, runner)
    try:
        vectors = validate_vectors(result['embeddings'], plan['count'])
    except ValueError as error:
        raise EmbedError('invalid_embeddings', str(error), 'Check the embedding adapter output.') from error
    return {'schema': 'embed/1', 'backend': plan['backend'], 'model': plan['model'], 'embeddings': vectors,
            'dimensions': len(vectors[0]), 'load_seconds': result.get('load_seconds'), 'seconds': result.get('seconds')}

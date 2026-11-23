"""Plan and run a source-oriented local image-segmentation task."""

import json
import subprocess
from pathlib import Path

from .backends import (
    backend_python,
    is_installed,
    load_manifests,
    load_registries,
    lookup,
)
from .inference_runner import InferenceError, RunnerContract, invoke


class SegmentError(InferenceError):
    """A segmentation failure safe to return to the caller."""


def plan_segment(model_path, parameters, output, image, directory):
    model, source, target = (Path(value).expanduser().absolute() for value in (model_path, image, output))
    try:
        config = json.loads((model / 'config.json').read_text())
    except (OSError, ValueError) as error:
        raise SegmentError('model_unreadable', 'The model config cannot be read.', 'Select a local SAM 2 model.') from error
    manifests = load_manifests()
    registries = load_registries(manifests)
    hits = [hit for hit in lookup(config.get('model_type'), manifests, registries)
            if hit['category'] == 'image_segmentation' and hit['backend'] == 'sam2']
    if not hits or not is_installed(manifests['sam2']):
        raise SegmentError('backend_not_installed', 'The selected segmentation backend is unavailable.',
                           'Install the SAM 2 backend and select a supported model.')
    if not source.is_file() or target.exists() or target.is_symlink() or not target.parent.is_dir():
        raise SegmentError('invalid_arguments', 'Input or output path is invalid.', 'Choose an existing image and new output path.')
    argv = [str(backend_python(manifests['sam2'])), str(Path(__file__).with_name('segment_runner.py')),
            '--module', hits[0]['module'], '--model', str(model), '--image', str(source), '--out', str(target),
            '--box', json.dumps(parameters['box']), '--points', json.dumps(parameters.get('points', [])),
            '--labels', json.dumps(parameters.get('labels', [])), '--threshold', str(parameters['mask_threshold'])]
    return {'backend': 'sam2', 'model': str(model), 'out': str(target), 'argv': argv}


CONTRACT = RunnerContract(verb='segment', error=SegmentError,
    accept=lambda value, plan: value.get('path') == plan['out'], action='generate the mask',
    remediation='Check the model and source image.', activity='Image segmentation',
    timeout_remediation='Retry using a new task.')


def run_segment(plan, manifests=None, root=None, timeout=3600, runner=subprocess.run):
    result = invoke(plan, CONTRACT, manifests, root, timeout, runner)
    return dict(result, schema='segment/1', backend=plan['backend'], model=plan['model'])

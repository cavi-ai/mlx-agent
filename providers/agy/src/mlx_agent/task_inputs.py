"""Named, ordered conditioning inputs for durable media tasks."""

from .task_contracts import TaskError

MAX_INPUT_FILES = 16
MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024


def needs_named(request):
    parameters = request['parameters']
    return (request['kind'] == 'mesh_generation' and bool(parameters.get('views') or parameters.get('guide'))
            or request['kind'] == 'video_generation' and bool(parameters.get('image') or parameters.get('audio')))


def specifications(request):
    """Return the named inputs in planner order, including repeated view images."""
    parameters = request['parameters']
    kind = request['kind']
    if kind == 'video_generation':
        return [(name, None, parameters[name], media) for name, media in
                (('image', 'image'), ('audio', 'ltx_audio')) if name in parameters]
    if kind == 'mesh_generation':
        items = [('image_path', None, parameters['image_path'], 'image')]
        items += [('views', index, path, 'image') for index, path in enumerate(parameters.get('views', []))]
        if 'guide' in parameters:
            items.append(('guide', None, parameters['guide'], 'mesh'))
        return items
    raise TaskError('invalid_task', 'This task does not use named inputs.')

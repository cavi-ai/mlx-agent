"""Public endpoint capabilities shared by discovery and workflow roles."""


def descriptor(port, runtime, service_id, state, auth_required):
    if (type(port) is not int or not 1 <= port <= 65535
            or runtime not in ('tasks', 'mlx_lm', 'mlx-vlm', 'strata', 'splash')
            or not isinstance(service_id, str) or len(service_id) != 64
            or state not in ('ready', 'starting', 'unknown') or type(auth_required) is not bool):
        raise ValueError('Invalid endpoint identity.')
    origin = 'http://127.0.0.1:' + str(port)
    return {'origin': origin, 'api_base': origin + '/v1', 'service_id': service_id,
            'runtime': runtime, 'state': state, 'auth_required': auth_required}


def operations(runtime, task_kinds):
    if runtime == 'tasks':
        paths = [('POST', '/v1/tasks'), ('GET', '/v1/tasks/{id}'),
                 ('GET', '/v1/tasks/{id}/result'), ('GET', '/v1/tasks/{id}/artifacts/{name}'),
                 ('POST', '/v1/tasks/{id}/cancel')]
        if 'embedding' in task_kinds:
            paths.append(('POST', '/v1/embeddings'))
        if 'classification' in task_kinds:
            paths.append(('POST', '/v1/systemone'))
    else:
        paths = [('POST', '/v1/chat/completions'), ('POST', '/v1/messages'),
                 ('POST', '/v1/responses'), ('GET', '/v1/responses/{id}')]
    paths.append(('GET', '/v1/models'))
    return [{'method': method, 'path': path} for method, path in paths]

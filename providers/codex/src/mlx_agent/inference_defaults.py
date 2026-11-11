"""Validated defaults for chat requests through a managed model gateway."""
import hashlib
import math


def validate(system_prompt=None, temperature=None, top_p=None):
    result = {}
    if system_prompt is not None:
        if not isinstance(system_prompt, str) or '\x00' in system_prompt:
            raise ValueError('System prompt must be text without null characters.')
        try:
            size = len(system_prompt.encode('utf-8'))
        except UnicodeError:
            raise ValueError('System prompt must be valid UTF-8 text.') from None
        if size > 8192:
            raise ValueError('System prompt must not exceed 8192 UTF-8 bytes.')
        if system_prompt:
            result['system_prompt'] = system_prompt
    for name, value, lower, upper in (('temperature', temperature, 0, 2), ('top_p', top_p, 0, 1)):
        if value is not None:
            if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError('{} must be between {} and {}.'.format(name, lower, upper))
            result[name] = float(value)
    return result


def apply(body, defaults):
    result = dict(body)
    for field in ('temperature', 'top_p'):
        if field in defaults:
            result.setdefault(field, defaults[field])
    messages = body.get('messages')
    if (defaults.get('system_prompt') and isinstance(messages, list)
            and all(isinstance(message, dict) for message in messages)
            and not any(message.get('role') in ('system', 'developer') for message in messages)):
        result['messages'] = [{'role': 'system', 'content': defaults['system_prompt']}] + messages
    return result


def summary(defaults):
    result = {key: value for key, value in defaults.items() if key != 'system_prompt'}
    if defaults.get('system_prompt'):
        result['system_prompt_sha256'] = hashlib.sha256(defaults['system_prompt'].encode()).hexdigest()
    return result

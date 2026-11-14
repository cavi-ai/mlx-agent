"""Dependency-free TRELLIS request component contract."""
from pathlib import Path

BASE = ('image_encoder', 'ss_flow', 'ss_decoder', 'shape_flow_512', 'shape_decoder')
HIGH = 'shape_flow_1024'
TEXTURE = ('tex_flow_512', 'tex_flow_1024', 'tex_decoder')
NAMES = frozenset(BASE + (HIGH,) + TEXTURE)


def textured(config):
    return all(name in config.get("components", {}) for name in TEXTURE)


def required_components(config, resolution):
    if type(resolution) is not int or resolution not in (512, 1024, 1536):
        raise ValueError('TRELLIS.2 resolution must be 512, 1024 or 1536.')
    names = BASE + ((HIGH,) if resolution > 512 else ())
    if textured(config):
        names += (('tex_flow_512' if resolution == 512 else 'tex_flow_1024'), 'tex_decoder')
    return names


def checkpoint_bytes(path, names):
    if any(name not in NAMES for name in names):
        raise ValueError('Unknown TRELLIS component.')
    return {name: (Path(path) / (name + '.safetensors')).stat().st_size for name in names}

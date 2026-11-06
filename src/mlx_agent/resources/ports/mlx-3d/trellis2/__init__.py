"""Native MLX image-to-mesh inference for TRELLIS.2."""


def load(path):
    from .pipeline import load as load_model
    return load_model(path)


def generate(model, image, out, **kwargs):
    from .pipeline import generate as generate_mesh
    return generate_mesh(model, image, out, **kwargs)

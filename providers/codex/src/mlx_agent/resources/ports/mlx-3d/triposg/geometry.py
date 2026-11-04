"""Bounded field extraction and a self-contained binary glTF mesh writer."""

import json
import struct

import numpy as np


def extract_mesh(decoder, resolution=128, chunk_size=2048):
    import mlx.core as mx
    from scipy.ndimage import binary_dilation, zoom
    from skimage.measure import marching_cubes

    low, high = -1.005, 1.005

    def evaluate(indices, size):
        results = []
        for start in range(0, len(indices), chunk_size):
            points = indices[start:start + chunk_size].astype(np.float32) * ((high - low) / (size - 1)) + low
            values = decoder(mx.array(points[None]))
            mx.eval(values)
            results.append(np.array(values, dtype=np.float32).reshape(-1))
        return np.concatenate(results)

    size = min(32, resolution)
    indices = np.stack(np.meshgrid(*([np.arange(size)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
    field = evaluate(indices, size).reshape(size, size, size)
    while size < resolution:
        next_size = min(2 * (size - 1) + 1, resolution)
        field = zoom(field, (next_size / size,) * 3, order=1)
        surface = np.zeros(field.shape, dtype=bool)
        for axis in range(3):
            left, right = [slice(None)] * 3, [slice(None)] * 3
            left[axis], right[axis] = slice(None, -1), slice(1, None)
            crossings = np.signbit(field[tuple(left)]) != np.signbit(field[tuple(right)])
            surface[tuple(left)] |= crossings
            surface[tuple(right)] |= crossings
        indices = np.argwhere(binary_dilation(surface, iterations=2))
        if not len(indices):
            raise ValueError("The decoded field contains no surface; try another image or seed.")
        field[tuple(indices.T)] = evaluate(indices, next_size)
        size = next_size
    if not np.isfinite(field).all() or not field.min() < 0 < field.max():
        raise ValueError("The decoded field contains no finite zero crossing.")
    vertices, faces, _, _ = marching_cubes(field, level=0, spacing=((high - low) / (size - 1),) * 3)
    return vertices + low, faces


def write_glb(path, vertices, faces):
    vertices = np.ascontiguousarray(vertices, dtype="<f4")
    faces = np.ascontiguousarray(faces, dtype="<u4")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be finite three-dimensional positions.")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.max() >= len(vertices):
        raise ValueError("Mesh faces must index existing vertices.")
    positions, indices = vertices.tobytes(), faces.tobytes()
    binary = positions + indices
    document = {
        "asset": {"version": "2.0", "generator": "mlx-agent"}, "scene": 0,
        "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962},
                        {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices), "target": 34963}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3",
                       "min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
                      {"bufferView": 1, "componentType": 5125, "count": faces.size, "type": "SCALAR"}],
    }
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    with open(path, "xb") as stream:
        stream.write(struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(binary)))
        stream.write(struct.pack("<II", len(encoded), 0x4E4F534A))
        stream.write(encoded)
        stream.write(struct.pack("<II", len(binary), 0x004E4942))
        stream.write(binary)

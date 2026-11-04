"""Iso-surface extraction, triangle-cap decimation and a self-contained binary glTF mesh writer."""

import json
import struct
import time

import numpy as np

from .octree import DEFAULT_BOUNDS, extract_grid


def marching_cubes(grid, bounds, depth):
    """Lewiner marching cubes at level 0, vertices mapped from grid units into `bounds`."""
    from skimage.measure import marching_cubes as march

    vertices, faces, _, _ = march(np.asarray(grid, dtype=np.float32), 0, method="lewiner")
    low = np.asarray(bounds[:3], dtype=np.float32)
    size = np.asarray(bounds[3:], dtype=np.float32) - low
    return (vertices / (2 ** depth) * size + low).astype(np.float32), np.ascontiguousarray(faces)


def extract_mesh(query, depth=9, bounds=DEFAULT_BOUNDS, seconds=None):
    """Mesh the zero level of `query` on a (2^depth)^3 octree grid; returns vertices, faces and the points queried per pass.

    `query` maps float16 points (N, 3) to logits (N,). A dict passed as `seconds` receives the dense, refinement and
    marching-cubes times.
    """
    grid, counts = extract_grid(query, bounds, depth - 1, depth, seconds)
    if not np.isfinite(grid).all() or not grid.min() < 0 < grid.max():
        raise ValueError("The decoded field contains no finite zero crossing.")
    started = time.perf_counter()
    vertices, faces = marching_cubes(grid, bounds, depth)
    if seconds is not None:
        seconds["marching_cubes"] = time.perf_counter() - started
    return vertices, faces, counts


def decimate(vertices, faces, max_faces):
    """Quadric edge collapse down to `max_faces` triangles; meshes at or under the cap pass through."""
    if len(faces) <= max_faces:
        return vertices, faces
    from fast_simplification import simplify

    points, triangles = simplify(np.ascontiguousarray(vertices, dtype=np.float32),
                                 np.ascontiguousarray(faces, dtype=np.int32), target_count=max_faces)
    return points.astype(np.float32), triangles


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

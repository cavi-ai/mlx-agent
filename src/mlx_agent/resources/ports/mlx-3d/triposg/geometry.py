"""Iso-surface extraction, triangle-cap decimation and a self-contained binary glTF mesh writer."""

import io
import json
import math
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


def decimate(vertices, faces, max_faces, aggression=7):
    """Quadric edge collapse down to `max_faces` triangles; meshes at or under the cap pass through.

    A lower `aggression` raises the error threshold more slowly, which joins fewer separate sheets along an edge.
    """
    if len(faces) <= max_faces:
        return vertices, faces
    from fast_simplification import simplify

    points, triangles = simplify(np.ascontiguousarray(vertices, dtype=np.float32),
                                 np.ascontiguousarray(faces, dtype=np.int32), target_count=max_faces, agg=aggression)
    return points.astype(np.float32), triangles


def _material_atlas(materials, faces):
    """Bake independent triangle islands into a linear occlusion/roughness/metallic PNG."""
    from PIL import Image

    count = len(faces)
    columns = math.isqrt(count - 1) + 1
    rows = (count + columns - 1) // columns
    width, height = columns * 8, rows * 8
    if width > 8192 or height > 8192:
        raise ValueError("Material atlas exceeds the 8192-pixel export bound; lower the face cap.")
    atlas = np.zeros((height, width, 3), dtype=np.uint8)
    tiles = atlas.reshape(rows, 8, columns, 8, 3).transpose(0, 2, 1, 3, 4)
    yy, xx = np.mgrid[0:8, 0:8]
    u, v = (xx - 1) / 5, (yy - 1) / 5
    weights = np.stack((1 - u - v, u, v), axis=-1).clip(0)
    weights /= weights.sum(axis=-1, keepdims=True)
    for start in range(0, count, 4096):
        end = min(start + 4096, count)
        ids = np.arange(start, end)
        baked = np.einsum("hwv,fvc->fhwc", weights, materials[faces[start:end]])
        tiles[ids // columns, ids % columns] = np.rint(baked * 255).astype(np.uint8)
    ids = np.arange(count)
    origins = np.stack(((ids % columns) * 8, (ids // columns) * 8), axis=1)
    corners = np.array([[1.5, 1.5], [6.5, 1.5], [1.5, 6.5]])
    uv = ((origins[:, None, :] + corners) / [width, height]).astype("<f4").reshape(-1, 2)
    stream = io.BytesIO()
    Image.fromarray(atlas, "RGB").save(stream, format="PNG")
    return uv, stream.getvalue()


def write_glb(path, vertices, faces, colors=None, materials=None):
    """A binary glTF mesh; ``colors`` (V, 3), linear RGB in [0, 1], become the COLOR_0 vertex attribute and
    ``materials`` (V, 3), linear occlusion, roughness and metallic in [0, 1], become COLOR_1 and a PBR texture."""
    vertices = np.ascontiguousarray(vertices, dtype="<f4")
    faces = np.ascontiguousarray(faces, dtype="<u4")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be finite three-dimensional positions.")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.max() >= len(vertices):
        raise ValueError("Mesh faces must index existing vertices.")
    if colors is not None:
        colors = np.ascontiguousarray(colors, dtype="<f4")
        if colors.shape != vertices.shape or not np.isfinite(colors).all() or colors.min() < 0 or colors.max() > 1:
            raise ValueError("Vertex colors must be one RGB value in [0, 1] per vertex.")
    if materials is not None:
        # glTF numbers COLOR_n attributes from 0 without gaps.
        if colors is None:
            raise ValueError("Vertex materials need vertex colors.")
        materials = np.ascontiguousarray(materials, dtype="<f4")
        if (materials.shape != vertices.shape or not np.isfinite(materials).all()
                or materials.min() < 0 or materials.max() > 1):
            raise ValueError("Vertex materials must be one occlusion, roughness and metallic value in [0, 1] "
                             "per vertex.")
    image = None
    if materials is not None:
        uv, image = _material_atlas(materials, faces)
        references = faces.ravel()
        vertices, colors, materials = vertices[references], colors[references], materials[references]
        faces = np.arange(len(references), dtype="<u4").reshape(-1, 3)
    positions, indices = vertices.tobytes(), faces.tobytes()
    binary = positions + indices
    document = _document(vertices, faces, len(positions), len(indices))
    for name, values in (("COLOR_0", colors), ("COLOR_1", materials)):
        if values is None:
            continue
        document["meshes"][0]["primitives"][0]["attributes"][name] = len(document["accessors"])
        document["accessors"].append({"bufferView": len(document["bufferViews"]), "componentType": 5126,
                                      "count": len(values), "type": "VEC3"})
        document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": values.nbytes,
                                        "target": 34962})
        binary += values.tobytes()
    if image is not None:
        primitive = document["meshes"][0]["primitives"][0]
        primitive["attributes"]["TEXCOORD_0"] = len(document["accessors"])
        document["accessors"].append({"bufferView": len(document["bufferViews"]), "componentType": 5126,
                                      "count": len(uv), "type": "VEC2"})
        document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": uv.nbytes,
                                        "target": 34962})
        binary += uv.tobytes()
        image_view = len(document["bufferViews"])
        document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(image)})
        binary += image
        primitive["material"] = 0
        document["images"] = [{"bufferView": image_view, "mimeType": "image/png"}]
        document["samplers"] = [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}]
        document["textures"] = [{"sampler": 0, "source": 0}]
        document["materials"] = [{"pbrMetallicRoughness": {"metallicFactor": 1, "roughnessFactor": 1,
            "metallicRoughnessTexture": {"index": 0, "texCoord": 0}},
            "occlusionTexture": {"index": 0, "texCoord": 0, "strength": 1}}]
    document["buffers"][0]["byteLength"] = len(binary)
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    with open(path, "xb") as stream:
        stream.write(struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(binary)))
        stream.write(struct.pack("<II", len(encoded), 0x4E4F534A))
        stream.write(encoded)
        stream.write(struct.pack("<II", len(binary), 0x004E4942))
        stream.write(binary)


def _document(vertices, faces, positions, indices):
    return {
        "asset": {"version": "2.0", "generator": "mlx-agent"}, "scene": 0,
        "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": positions + indices}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": positions, "target": 34962},
                        {"buffer": 0, "byteOffset": positions, "byteLength": indices, "target": 34963}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3",
                       "min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
                      {"bufferView": 1, "componentType": 5125, "count": faces.size, "type": "SCALAR"}],
    }

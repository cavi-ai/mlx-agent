"""Surface voxels of a guide mesh, used in place of the sampled sparse structure."""

import json
import os
import struct
from pathlib import Path

import numpy as np

COMPONENTS = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTHS = {"SCALAR": 1, "VEC3": 3}
TRIANGLES = 4
CHUNK_POINTS = 1 << 21


def _chunks(path):
    data = Path(path).read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError("The guide must be a binary glTF (.glb) file.")
    _, version, length = struct.unpack_from("<III", data)
    if version != 2 or length != len(data):
        raise ValueError("The guide GLB header does not match the file.")
    document, binary, offset = None, b"", 12
    while offset + 8 <= len(data):
        size, kind = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8:offset + 8 + size]
        if kind == 0x4E4F534A:
            document = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        offset += 8 + size
    if not isinstance(document, dict):
        raise ValueError("The guide GLB has no JSON chunk.")
    return document, binary


def _accessor(document, binary, index, kind, components):
    accessor = document["accessors"][index]
    if "sparse" in accessor or "bufferView" not in accessor or accessor.get("type") != kind:
        raise ValueError("Guide accessors must be dense {0} data.".format(kind))
    if accessor["componentType"] not in components:
        raise ValueError("The guide uses an unsupported {0} component type.".format(kind))
    view = document["bufferViews"][accessor["bufferView"]]
    if view.get("buffer", 0) != 0:
        raise ValueError("Guide data must be in the GLB binary chunk.")
    dtype = np.dtype(COMPONENTS[accessor["componentType"]]).newbyteorder("<")
    width, count = WIDTHS[kind], accessor["count"]
    stride = view.get("byteStride") or dtype.itemsize * width
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    if count and start + stride * (count - 1) + dtype.itemsize * width > len(binary):
        raise ValueError("A guide accessor reads past the GLB binary chunk.")
    rows = np.ndarray((count, width), dtype=dtype, buffer=binary, offset=start, strides=(stride, dtype.itemsize))
    return np.array(rows)


def _matrix(node):
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    x, y, z, w = node.get("rotation", (0, 0, 0, 1))
    rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * np.asarray(node.get("scale", (1, 1, 1)), dtype=np.float64)[None]
    matrix[:3, 3] = node.get("translation", (0, 0, 0))
    return matrix


def read(path):
    """World-space glTF (y-up) vertices and triangle indices of every triangle primitive in the default scene."""
    document, binary = _chunks(path)
    nodes = document.get("nodes", [])
    scenes = document.get("scenes")
    if scenes:
        roots = scenes[document.get("scene", 0)].get("nodes", [])
    else:
        children = {child for node in nodes for child in node.get("children", [])}
        roots = [index for index in range(len(nodes)) if index not in children]
    vertices, faces, count = [], [], 0
    pending = [(index, np.eye(4)) for index in roots]
    while pending:
        index, parent = pending.pop()
        node = nodes[index]
        world = parent @ _matrix(node)
        pending.extend((child, world) for child in node.get("children", []))
        for primitive in document["meshes"][node["mesh"]]["primitives"] if "mesh" in node else []:
            if primitive.get("mode", TRIANGLES) != TRIANGLES or "POSITION" not in primitive.get("attributes", {}):
                continue
            positions = _accessor(document, binary, primitive["attributes"]["POSITION"], "VEC3", (5126,))
            if "indices" in primitive:
                indices = _accessor(document, binary, primitive["indices"], "SCALAR", (5121, 5123, 5125))[:, 0]
            else:
                indices = np.arange(len(positions))
            if len(indices) % 3 or (len(indices) and indices.max() >= len(positions)):
                raise ValueError("Guide triangles must index existing vertices.")
            vertices.append(positions.astype(np.float64) @ world[:3, :3].T + world[:3, 3])
            faces.append(indices.reshape(-1, 3).astype(np.int64) + count)
            count += len(positions)
    if not faces or not sum(len(group) for group in faces):
        raise ValueError("The guide GLB holds no triangles.")
    vertices, faces = np.concatenate(vertices), np.concatenate(faces)
    if not np.isfinite(vertices).all():
        raise ValueError("Guide vertices must be finite.")
    return vertices, faces


def _cells(corners, level, resolution):
    side = np.arange(level + 1)
    i, j = np.meshgrid(side, side, indexing="ij")
    keep = i + j <= level
    u, v = i[keep] / level, j[keep] / level
    weights = np.stack([1 - u - v, u, v], axis=1)
    found = []
    for start in range(0, len(corners), max(1, CHUNK_POINTS // len(weights))):
        points = np.einsum("sk,fkd->fsd", weights, corners[start:start + max(1, CHUNK_POINTS // len(weights))])
        cells = np.floor((points.reshape(-1, 3) + 0.5) * resolution).astype(np.int64)
        found.append(np.unique(np.clip(cells, 0, resolution - 1), axis=0))
    return found


def surface_coords(path, resolution):
    """Integer (x, y, z) voxels at ``resolution`` that the guide's surface passes through, after centering the guide
    in the unit cube with its largest extent spanning it; TRELLIS axes are z-up with the glTF front facing -y.
    Also returns the glTF-axes center and the extent that map the unit cube back onto the guide."""
    vertices, faces = read(path)
    upright = np.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], axis=1)
    used = upright[np.unique(faces)]
    low, high = used.min(axis=0), used.max(axis=0)
    extent = float((high - low).max())
    if not extent > 0:
        raise ValueError("The guide mesh has no extent.")
    center = (low + high) / 2
    corners = ((upright - center) / extent)[faces]
    # Samples at most half a cell apart along each edge reach every voxel the triangle crosses.
    edges = np.linalg.norm(corners - corners[:, [1, 2, 0]], axis=2).max(axis=1)
    levels = np.maximum(np.ceil(edges * 2 * resolution), 1).astype(np.int64)
    found = []
    for level in np.unique(levels):
        found.extend(_cells(corners[levels == level], int(level), resolution))
    coords = np.unique(np.concatenate(found), axis=0).astype(np.int32)
    return coords, np.array([center[0], center[2], -center[1]]), extent


def place(path, center, extent):
    """Rewrite a generated unit-cube GLB's positions as ``position * extent + center``, the guide's frame."""
    document, binary = _chunks(path)
    binary = bytearray(binary)
    accessor = document["accessors"][document["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    view = document["bufferViews"][accessor["bufferView"]]
    if view.get("byteStride", 12) != 12 or accessor["componentType"] != 5126:
        raise ValueError("Generated positions must be packed float32 triples.")
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    positions = np.frombuffer(bytes(binary), dtype="<f4", count=accessor["count"] * 3, offset=start)
    placed = (positions.reshape(-1, 3).astype(np.float64) * extent + center).astype("<f4")
    binary[start:start + placed.nbytes] = placed.tobytes()
    accessor["min"], accessor["max"] = placed.min(axis=0).tolist(), placed.max(axis=0).tolist()
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    temporary = Path(str(path) + ".placing")
    with open(temporary, "xb") as stream:
        stream.write(struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(binary)))
        stream.write(struct.pack("<II", len(encoded), 0x4E4F534A))
        stream.write(encoded)
        stream.write(struct.pack("<II", len(binary), 0x004E4942))
        stream.write(binary)
    os.replace(temporary, path)

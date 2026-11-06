"""Triangle mesh from a flexible dual grid: one vertex per voxel, one quad per crossed edge whose four voxels exist."""

import numpy as np

EDGE_NEIGHBORS = np.array([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
], dtype=np.int64)
SPLIT_02 = np.array([0, 1, 2, 0, 2, 3])
SPLIT_13 = np.array([0, 1, 3, 3, 1, 2])


def to_mesh(coords, dual_vertices, crossed, split_weight, grid_size, aabb=((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))):
    """Vertices (V, 3) float32 and triangles (F, 3) int64; quads split along the heavier split-weight diagonal."""
    coords = np.asarray(coords, dtype=np.int64)
    low, high = (np.asarray(value, dtype=np.float32) for value in aabb)
    voxel = (high - low) / grid_size
    vertices = ((coords + dual_vertices) * voxel + low).astype(np.float32)
    span = int(coords.max()) + 2
    keys = (coords[:, 0] * span + coords[:, 1]) * span + coords[:, 2]
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    corners = (coords[:, None, None, :] + EDGE_NEIGHBORS[None])[np.asarray(crossed, dtype=bool)]
    corner_keys = (corners[..., 0] * span + corners[..., 1]) * span + corners[..., 2]
    position = np.minimum(np.searchsorted(ordered, corner_keys), len(ordered) - 1)
    found = ordered[position] == corner_keys
    quads = order[position][found.all(axis=1)]
    weight = np.asarray(split_weight, dtype=np.float32).reshape(-1)[quads]
    first = (weight[:, 0] * weight[:, 2] > weight[:, 1] * weight[:, 3])[:, None]
    triangles = np.where(first, quads[:, SPLIT_02], quads[:, SPLIT_13]).reshape(-1, 3)
    return vertices, triangles

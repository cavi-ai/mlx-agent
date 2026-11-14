"""A closed, outward-facing solid around a dual-grid mesh, which is open, unoriented and non-manifold."""

import numpy as np

PAD = 3


def closed_surface(vertices, faces, grid, band=0.75):
    """Vertices (V, 3) float32 and triangles (F, 3) of the solid's surface, ``band`` cells outside the mesh.

    The mesh lies in the unit cube centered on the origin, sampled on ``grid`` cells per axis. Cells within ``band`` of
    the mesh are solid, and so is every cell that no straight line along an axis reaches from outside: the inside of
    a hollow shell fills, except the part seen straight through an opening. Each edge joins two triangles.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes

    points = (np.asarray(vertices, dtype=np.float64) + 0.5) * grid
    corners = points[np.asarray(faces, dtype=np.int64)]
    # Dual-grid edges span about one cell, so corners, edge midpoints and centers lie within half a cell of any point.
    samples = np.concatenate([points, corners.mean(axis=1), (corners[:, 0] + corners[:, 1]) / 2,
                              (corners[:, 1] + corners[:, 2]) / 2, (corners[:, 2] + corners[:, 0]) / 2])
    cells = np.floor(samples).astype(np.int64)
    origin = cells.min(axis=0) - PAD
    shape = tuple(cells.max(axis=0) - origin + PAD + 1)
    occupied = np.zeros(shape, dtype=bool)
    occupied[tuple((cells - origin).T)] = True
    near = ndimage.binary_dilation(occupied, structure=np.ones((3, 3, 3), dtype=bool), iterations=2)
    index = np.argwhere(near)
    far = band + 1.0
    distance, _ = cKDTree(samples).query(index + origin + 0.5, distance_upper_bound=far, workers=-1)
    field = np.full(shape, far, dtype=np.float32)
    field[tuple(index.T)] = np.minimum(distance, far)
    empty = field > band
    outside = np.zeros(shape, dtype=bool)
    for axis in range(3):
        outside |= np.logical_and.accumulate(empty, axis=axis)
        outside |= np.flip(np.logical_and.accumulate(np.flip(empty, axis), axis=axis), axis)
    level = field - band
    # An empty cell hidden along every axis is solid, so only the seen side of the band crosses zero.
    level[empty & ~outside] = -band
    surface, triangles, _, _ = marching_cubes(level, 0, method="lewiner")
    surface = ((surface + origin + 0.5) / grid - 0.5).astype(np.float32)
    a, b, c = (surface[triangles[:, i]].astype(np.float64) for i in range(3))
    if np.einsum("ij,ij->", a, np.cross(b, c)) < 0:
        triangles = triangles[:, ::-1]
    return surface, np.ascontiguousarray(triangles)


def without_fins(vertices, faces):
    """The mesh without triangles that share all three vertices with another, and without the vertices only they used.

    Edge collapse folds a thin wall into such back-to-back pairs; they enclose nothing and put four faces on an edge.
    """
    faces = np.asarray(faces)
    _, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True)
    faces = faces[counts[inverse.reshape(-1)] == 1]
    used, compact = np.unique(faces, return_inverse=True)
    return np.asarray(vertices)[used], np.ascontiguousarray(compact.reshape(-1, 3).astype(faces.dtype))

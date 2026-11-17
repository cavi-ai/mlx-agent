"""A closed, outward-facing solid around a dual-grid mesh, which is open, unoriented and non-manifold."""

import numpy as np

PAD = 3
FIELD_CACHE_LIMIT = 64 * 1024 ** 2


def closed_surface(vertices, faces, grid, band=0.75, *, field_cache_bytes=FIELD_CACHE_LIMIT):
    """Fill cells hidden along every axis, using bounded tiles and 2D axis projections.

    The mesh is sampled at the requested grid resolution; axis-visible openings remain open.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes
    from itertools import product

    if type(field_cache_bytes) is not int or not 0 <= field_cache_bytes <= FIELD_CACHE_LIMIT:
        raise ValueError("Field cache must be between 0 and 64 MiB.")
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("A closed surface requires at least one triangle.")
    points = (np.asarray(vertices, dtype=np.float64) + 0.5) * grid
    corners = points[triangles]
    samples = np.concatenate([points[np.unique(triangles)], corners.mean(axis=1),
                              (corners[:, 0] + corners[:, 1]) / 2,
                              (corners[:, 1] + corners[:, 2]) / 2,
                              (corners[:, 2] + corners[:, 0]) / 2])
    cells = np.unique(np.floor(samples).astype(np.int64), axis=0)
    origin = cells.min(axis=0) - PAD
    cells -= origin
    shape = cells.max(axis=0) + PAD + 1
    plane_sizes = [int(shape[1] * shape[2]), int(shape[0] * shape[2]), int(shape[0] * shape[1])]
    if sum(plane_sizes) * 2 * np.dtype(np.int32).itemsize > 64 * 1024 * 1024:
        raise ValueError("Axis projections exceed the 64 MiB reconstruction bound.")
    projection_shapes = [(shape[1], shape[2]), (shape[0], shape[2]), (shape[0], shape[1])]
    first = [np.full(s, shape[axis], dtype=np.int32) for axis, s in enumerate(projection_shapes)]
    last = [np.full(s, -1, dtype=np.int32) for s in projection_shapes]
    tile = 64
    group_keys, inverse = np.unique(cells // tile, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    offsets = np.searchsorted(inverse[order], np.arange(len(group_keys) + 1))
    groups = {tuple(key): cells[order[offsets[i]:offsets[i + 1]]] for i, key in enumerate(group_keys)}
    candidates = set()
    limits = (shape - 1) // tile
    for key in groups:
        for delta in product((-1, 0, 1), repeat=3):
            neighbor = tuple(key[i] + delta[i] for i in range(3))
            if all(0 <= neighbor[i] <= limits[i] for i in range(3)):
                candidates.add(neighbor)
    tree = cKDTree(samples)
    far = band + 1.0

    def field(start, end):
        low = np.maximum(start - 2, 0)
        high = np.minimum(end + 2, shape)
        occupied = np.zeros(tuple(high - low), dtype=bool)
        for key in product(*(range(int(low[i] // tile), int((high[i] - 1) // tile) + 1) for i in range(3))):
            group = groups.get(key)
            if group is None:
                continue
            selected = group[np.all((group >= low) & (group < high), axis=1)]
            if len(selected):
                occupied[tuple((selected - low).T)] = True
        near = ndimage.binary_dilation(occupied, structure=np.ones((3, 3, 3), dtype=bool), iterations=2)
        near = near[tuple(slice(int(start[i] - low[i]), int(end[i] - low[i])) for i in range(3))]
        index = np.argwhere(near)
        values = np.full(tuple(end - start), far, dtype=np.float32)
        if len(index):
            distance, _ = tree.query(index + start + origin + 0.5, distance_upper_bound=far, workers=-1)
            values[tuple(index.T)] = np.minimum(distance, far)
        return values

    field_cache = {}
    retained_bytes = 0
    try:
        for key in sorted(candidates):
            start = np.asarray(key) * tile
            end = np.minimum(start + tile, shape)
            values = field(start, end)
            if retained_bytes + values.nbytes <= field_cache_bytes and np.any(values < far):
                field_cache[key] = values
                retained_bytes += values.nbytes
            index = np.argwhere(values <= band) + start
            if not len(index):
                continue
            for axis, other in enumerate(((1, 2), (0, 2), (0, 1))):
                coords = (index[:, other[0]], index[:, other[1]])
                np.minimum.at(first[axis], coords, index[:, axis])
                np.maximum.at(last[axis], coords, index[:, axis])

        surfaces, faces_out = [], []
        vertex_offset = 0
        for key in product(*(range(int((shape[i] - 2) // tile) + 1) for i in range(3))):
            start = np.asarray(key) * tile
            end = np.minimum(start + tile + 1, shape)
            region = tuple(slice(int(start[i]), int(end[i])) for i in range(3))
            possible = True
            for axis, other in enumerate(((1, 2), (0, 2), (0, 1))):
                section = (region[other[0]], region[other[1]])
                if not np.any((first[axis][section] < end[axis]) & (last[axis][section] >= start[axis])):
                    possible = False
                    break
            nearby = any(tuple(key[i] + delta[i] for i in range(3)) in candidates
                         for delta in product((0, 1), repeat=3))
            if not possible and not nearby:
                continue
            cached = field_cache.pop(key, None)
            if cached is None:
                values = field(start, end)
            else:
                retained_bytes -= cached.nbytes
                interior = np.asarray(cached.shape)
                extent = end - start
                values = np.empty(tuple(extent), dtype=np.float32)
                values[tuple(slice(0, int(size)) for size in interior)] = cached
                del cached
                # Three disjoint border slabs partition the added coordinates.
                slabs = (([interior[0], 0, 0], [extent[0], extent[1], extent[2]]),
                         ([0, interior[1], 0], [interior[0], extent[1], extent[2]]),
                         ([0, 0, interior[2]], [interior[0], interior[1], extent[2]]))
                for low, high in slabs:
                    low, high = np.asarray(low), np.asarray(high)
                    if np.all(low < high):
                        values[tuple(slice(int(low[i]), int(high[i])) for i in range(3))] = field(start + low, start + high)
            inside = np.ones(tuple(end - start), dtype=bool)
            x, y, z = (np.arange(start[i], end[i]) for i in range(3))
            inside &= (x[:, None, None] >= first[0][region[1], region[2]][None, :, :]) & (x[:, None, None] <= last[0][region[1], region[2]][None, :, :])
            inside &= (y[None, :, None] >= first[1][region[0], region[2]][:, None, :]) & (y[None, :, None] <= last[1][region[0], region[2]][:, None, :])
            inside &= (z[None, None, :] >= first[2][region[0], region[1]][:, :, None]) & (z[None, None, :] <= last[2][region[0], region[1]][:, :, None])
            empty = values > band
            level = values - band
            level[empty & inside] = -band
            if not (level.min() <= 0 < level.max()):
                continue
            surface, triangles, _, _ = marching_cubes(level, 0, method="lewiner")
            surface = ((surface + start + origin + 0.5) / grid - 0.5).astype(np.float32)
            surfaces.append(surface)
            faces_out.append(triangles + vertex_offset)
            vertex_offset += len(surface)
    finally:
        field_cache.clear()
    if not surfaces:
        raise ValueError("The mesh has no reconstructable closed surface.")
    surface, inverse = np.unique(np.concatenate(surfaces), axis=0, return_inverse=True)
    triangles = inverse[np.concatenate(faces_out)].astype(np.int32)
    valid = (triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2]) & (triangles[:, 2] != triangles[:, 0])
    triangles = triangles[valid]
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

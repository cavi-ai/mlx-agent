"""A closed, outward-facing solid around a dual-grid mesh, which is open, unoriented and non-manifold."""

import json
from pathlib import Path

import numpy as np

PAD = 3
OCCUPANCY_KEY_MIN_ROWS = 65536


def _unique_occupancy(rows):
    """Deduplicate owned integer cells in lexicographic order without rank overflow."""
    if not len(rows) or len(rows) < OCCUPANCY_KEY_MIN_ROWS:
        return np.unique(rows, axis=0)
    low = [int(value) for value in rows.min(axis=0)]
    high = [int(value) for value in rows.max(axis=0)]
    spans = [high[i] - low[i] + 1 for i in range(3)]
    limit = min(np.iinfo(np.intp).max, np.iinfo(np.int64).max)
    if spans[0] * spans[1] * spans[2] > limit:
        return np.unique(rows, axis=0)
    rows -= np.asarray(low, dtype=np.int64)
    keys = np.ravel_multi_index(tuple(rows.T), tuple(spans))
    del rows
    keys = np.unique(keys)
    cells = np.empty((len(keys), 3), dtype=np.int64)
    cells[:, 2] = keys % spans[2]
    keys //= spans[2]
    cells[:, 1] = keys % spans[1]
    keys //= spans[1]
    cells[:, 0] = keys
    cells += np.asarray(low, dtype=np.int64)
    return cells


def _expanded_occupancy(occupied):
    """Expand two cells in each axis, preserving the zero boundary."""
    if not occupied.any():
        return occupied
    from scipy import ndimage
    return ndimage.maximum_filter(occupied, size=5, mode="constant", cval=0)


def reconstruction_settings(mesh=None):
    defaults = json.loads(Path(__file__).with_name("reconstruction.json").read_text(encoding="utf-8"))
    settings = {}
    for key, minimum in (("field_cache_bytes", 0), ("projection_budget_bytes", 1)):
        value = mesh.get(key, defaults[key]) if isinstance(mesh, dict) else defaults[key]
        if type(value) is not int or value < minimum:
            raise ValueError(f"mesh.{key} must be an integer of at least {minimum} bytes.")
        settings[key] = value
    return settings


def closed_surface(vertices, faces, grid, band=0.75, *, mesh=None):
    """Fill cells hidden along every axis, using bounded tiles and 2D axis projections.

    The mesh is sampled at the requested grid resolution; axis-visible openings remain open. The first pass caches
    sparse distances up to the configured byte budget. The second pass recomputes any regions that did not fit.
    """
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes
    from itertools import product

    settings = reconstruction_settings(mesh)
    cache_budget = settings["field_cache_bytes"]
    projection_budget = settings["projection_budget_bytes"]
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("A closed surface requires at least one triangle.")
    points = (np.asarray(vertices, dtype=np.float64) + 0.5) * grid
    corners = points[triangles]
    samples = np.concatenate([points[np.unique(triangles)], corners.mean(axis=1),
                              (corners[:, 0] + corners[:, 1]) / 2,
                              (corners[:, 1] + corners[:, 2]) / 2,
                              (corners[:, 2] + corners[:, 0]) / 2])
    cells = _unique_occupancy(np.floor(samples).astype(np.int64))
    origin = cells.min(axis=0) - PAD
    cells -= origin
    shape = cells.max(axis=0) + PAD + 1
    plane_sizes = [int(shape[1] * shape[2]), int(shape[0] * shape[2]), int(shape[0] * shape[1])]
    if sum(plane_sizes) * 2 * np.dtype(np.int32).itemsize > projection_budget:
        raise ValueError(f"Axis projections exceed mesh.projection_budget_bytes ({projection_budget} bytes).")
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
        near = _expanded_occupancy(occupied)
        near = near[tuple(slice(int(start[i] - low[i]), int(end[i] - low[i])) for i in range(3))]
        index = np.argwhere(near)
        values = np.full(tuple(end - start), far, dtype=np.float32)
        if len(index):
            distance, _ = tree.query(index + start + origin + 0.5, distance_upper_bound=far, workers=-1)
            values[tuple(index.T)] = np.minimum(distance, far)
        return values

    # Candidate tiles with finite distances are either stored or recomputed as needed.
    stored = {}
    uncached = set()
    retained_bytes = 0

    def assembled(key, start, end):
        values = np.full(tuple(end - start), far, dtype=np.float32)
        extent = end - start
        for delta in product((0, 1), repeat=3):
            neighbor = tuple(key[i] + delta[i] for i in range(3))
            if neighbor in uncached:
                low = np.asarray(neighbor) * tile
                high = np.minimum(low + tile, end)
                if np.all(low < high):
                    region = tuple(slice(int(a), int(b)) for a, b in zip(low - start, high - start))
                    values[region] = field(low, high)
                continue
            entry = stored.get(neighbor)
            if entry is None:
                continue
            # A neighbor tile contributes only cells on its low faces along the axes it is offset on.
            local, distance = entry[:2] if not any(delta) else entry[2:]
            cells = local.astype(np.intp) + np.asarray(delta) * tile
            kept = np.all(cells < extent, axis=1)
            values[tuple(cells[kept].T)] = distance[kept]
        return values

    try:
        for key in sorted(candidates):
            start = np.asarray(key) * tile
            end = np.minimum(start + tile, shape)
            values = field(start, end)
            local = np.argwhere(values < far).astype(np.uint8)
            if len(local):
                distance = values[tuple(local.T)]
                low = np.any(local == 0, axis=1)
                entry = (local, distance, local[low], distance[low])
                size = sum(array.nbytes for array in entry)
                if retained_bytes + size <= cache_budget:
                    stored[key] = entry
                    retained_bytes += size
                else:
                    uncached.add(key)
                del entry, local, distance
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
            values = assembled(key, start, end)
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
        stored.clear()
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

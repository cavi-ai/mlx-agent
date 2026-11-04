"""Hierarchical octree extraction of a logit grid from a point-query function, in numpy.

A dense pass at 2^(depth-1) is followed by one refinement that queries only the cells near the surface. The
float16 arithmetic (grid points, band test, normalized query coordinates and their index round trip) follows
TripoSG's reference extraction, so the query sets and the final grid match it.
"""

import time

import numpy as np

DEFAULT_BOUNDS = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005)
BAND_THRESHOLD = 1.0


def linspace_half(low, high, count):
    """float16 linspace: float16 end points, float32 steps, counted from both ends."""
    start = np.float32(np.float16(np.float32(low)))
    end = np.float32(np.float16(np.float32(high)))
    step = (end - start) / np.float32(count - 1)
    index = np.arange(count, dtype=np.float32)
    values = np.where(index < count // 2, start + step * index, end - step * (np.float32(count - 1) - index))
    return values.astype(np.float16)


def dense_points(bounds, depth):
    """All 2^depth cubed grid points as float16 (N, 3) in x-major order, and the side length."""
    side = 2 ** depth
    axes = [linspace_half(bounds[axis], bounds[axis + 3], side) for axis in range(3)]
    xs, ys, zs = np.meshgrid(*axes, indexing="ij")
    return np.stack((xs, ys, zs), axis=-1).reshape(-1, 3), side


def band_cells(grid, threshold=BAND_THRESHOLD):
    """Interior cells whose squashed logit sigmoid(l) * 2 - 1 lies strictly inside (-threshold, threshold)."""
    core = grid[1:-1, 1:-1, 1:-1].astype(np.float32)
    with np.errstate(over="ignore"):
        squashed = (1.0 / (1.0 + np.exp(-core))).astype(np.float16)
    value = squashed * np.float16(2) - np.float16(1)
    return np.argwhere(np.abs(value) < np.float16(threshold)) + 1


def _dilate(mask, radius):
    for axis in range(3):
        grown = mask.copy()
        for shift in range(1, radius + 1):
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[axis], high[axis] = slice(shift, None), slice(None, -shift)
            grown[tuple(low)] |= mask[tuple(high)]
            grown[tuple(high)] |= mask[tuple(low)]
        mask = grown
    return mask


def child_cells(cells, side):
    """Mark `cells` on a side^3 grid, grow the mark by 2 cells (1 from side 512), return the 8 children of each."""
    mask = np.zeros((side, side, side), dtype=bool)
    mask[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    parents = np.argwhere(_dilate(mask, 2 if side < 512 else 1)).astype(np.int16)
    offsets = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    return np.concatenate([parents * 2 + np.array(offset, dtype=np.int16) for offset in offsets])


def refine_level(grid, query, depth, extent):
    """One octree level: nearest-upsample `grid` to 2^depth and overwrite the cells near the surface.

    `query` maps float16 points (N, 3) to logits (N,); `extent` is the half width of the bounds. Returns the new
    grid and the number of points queried.
    """
    side = 2 ** depth
    offset = side / 2
    refined = grid.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
    children = child_cells(band_cells(grid), side // 2).astype(np.float16)
    shifted = (children.astype(np.float32) - np.float32(offset)).astype(np.float16)
    points = (shifted.astype(np.float32) * np.float32(extent / offset)).astype(np.float16)
    values = np.asarray(query(points)).astype(np.float16)
    scaled = (points.astype(np.float32) * np.float32(offset / extent)).astype(np.float16)
    index = (scaled.astype(np.float32) + np.float32(offset)).astype(np.float16).astype(np.int32)
    refined[index[:, 0], index[:, 1], index[:, 2]] = values
    return refined, len(points)


def extract_grid(query, bounds=DEFAULT_BOUNDS, dense_depth=8, final_depth=9, seconds=None):
    """The (2^final_depth)^3 float16 logit grid and the point count queried per pass.

    When `seconds` is a dict it receives `decode_dense` and `decode_refine`, the wall time of the dense pass and
    of all refinement levels.
    """
    started = time.perf_counter()
    points, side = dense_points(bounds, dense_depth)
    grid = np.asarray(query(points)).astype(np.float16).reshape(side, side, side)
    counts = [len(points)]
    del points
    dense = time.perf_counter()
    for depth in range(dense_depth + 1, final_depth + 1):
        grid, count = refine_level(grid, query, depth, abs(bounds[0]))
        counts.append(count)
    if seconds is not None:
        seconds["decode_dense"] = dense - started
        seconds["decode_refine"] = time.perf_counter() - dense
    return grid, counts

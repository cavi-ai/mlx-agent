"""Voxel centers projected through the estimated front camera onto image feature maps."""

import math

import mlx.core as mx
import numpy as np

AXES = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
SENSOR_WIDTH = 32.0


def distance(camera_angle_x, mesh_scale=1.0):
    """Camera distance at which the unit cube's half width spans the image half width."""
    return 0.5 / mesh_scale / math.tan(camera_angle_x / 2)


def grid_points(coords, resolution, mesh_scale=1.0):
    """World positions of voxel ``coords`` (N, 3) on a ``resolution`` grid over [-1, 1], scaled into the unit cube."""
    axis = np.linspace(-1, 1, resolution, dtype=np.float32)
    points = axis[np.asarray(coords, dtype=np.int64)] @ AXES.T
    return points / mesh_scale / 2


def pixels(points, camera_angle_x, camera_distance, image_resolution):
    """(N, 2) pixel positions (x, y) of world ``points`` seen by a camera at (0, -distance, 0) looking along +y."""
    transform = np.array([[1, 0, 0, 0], [0, 0, -1, -camera_distance], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    camera = (homogeneous @ np.linalg.inv(transform).T)[:, :3]
    focal = 16.0 / math.tan(camera_angle_x / 2) * image_resolution / SENSOR_WIDTH
    depth = -camera[:, 2] + 1e-8
    return np.stack([focal * camera[:, 0] / depth + image_resolution / 2,
                     -focal * camera[:, 1] / depth + image_resolution / 2], axis=1).astype(np.float32)


def sample(feature_map, positions, image_resolution):
    """Bilinear samples (N, C) of a (H, W, C) map at pixel ``positions``, like ``grid_sample`` with
    ``align_corners=False`` and border padding."""
    height, width = feature_map.shape[:2]
    ndc = (positions + 0.5) / image_resolution * 2 - 1
    x = np.clip(((ndc[:, 0] + 1) * width - 1) / 2, 0, width - 1)
    y = np.clip(((ndc[:, 1] + 1) * height - 1) / 2, 0, height - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = (x - x0).astype(np.float32)[:, None], (y - y0).astype(np.float32)[:, None]
    flat = feature_map.reshape(height * width, -1)
    corner = lambda row, col: mx.take(flat, mx.array((row * width + col).astype(np.int32)), axis=0)
    top = corner(y0, x0) * mx.array(1 - fx) + corner(y0, x1) * mx.array(fx)
    bottom = corner(y1, x0) * mx.array(1 - fx) + corner(y1, x1) * mx.array(fx)
    return top * mx.array(1 - fy) + bottom * mx.array(fy)

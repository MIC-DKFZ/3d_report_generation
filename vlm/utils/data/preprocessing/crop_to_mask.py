import math
import numpy as np


def enforce_min_size(box, min_size_xyz, img_shape):
    if not min_size_xyz:
        return box  # (z0, z1, y0, y1, x0, x1)

    z0, z1, y0, y1, x0, x1 = box
    need_x = max(0, min_size_xyz[0] - (x1 - x0))
    need_y = max(0, min_size_xyz[1] - (y1 - y0))
    need_z = max(0, min_size_xyz[2] - (z1 - z0))

    # symmetric expansion, then clip to image size
    x0 = max(0, x0 - need_x // 2)
    x1 = min(img_shape[2], x1 + need_x - need_x // 2)
    y0 = max(0, y0 - need_y // 2)
    y1 = min(img_shape[1], y1 + need_y - need_y // 2)
    z0 = max(0, z0 - need_z // 2)
    z1 = min(img_shape[0], z1 + need_z - need_z // 2)

    return [z0, z1, y0, y1, x0, x1]


def expand_bounding_box(box, x_upper, y_upper, z_upper, scale=0.04):
    z_min, z_max, y_min, y_max, x_min, x_max = box
    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    ex, ey, ez = dx * scale / 2, dy * scale / 2, dz * scale / 2

    new_min_x = max(0, math.floor(x_min - ex))
    new_max_x = min(x_upper, math.ceil(x_max + ex))
    new_min_y = max(0, math.floor(y_min - ey))
    new_max_y = min(y_upper, math.ceil(y_max + ey))
    new_min_z = max(0, math.floor(z_min - ez))
    new_max_z = min(z_upper, math.ceil(z_max + ez))

    return [new_min_z, new_max_z, new_min_y, new_max_y, new_min_x, new_max_x]


def get_mask_bounding_box(mask_arr):
    """Return bounding box (z_min, z_max, y_min, y_max, x_min, x_max) of non-zero mask voxels."""
    coords = np.argwhere(mask_arr > 0)
    if coords.size == 0:
        raise ValueError("Mask is empty (no voxels with value > 0).")
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0) + 1  # +1 to include the last voxel
    return z_min, z_max, y_min, y_max, x_min, x_max

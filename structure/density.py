import numpy as np
from scipy.ndimage import gaussian_filter

from structure.config import CONFIG


def build_density_field(positions):

    settings = CONFIG["density"]

    grid_size = settings["grid_size"]
    smoothing = settings["smoothing_scale_mpc"]

    positions = np.asarray(positions, dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Positions must have shape (N, 3).")

    if len(positions) == 0:
        raise ValueError("No galaxy positions supplied.")

    if not np.all(np.isfinite(positions)):
        raise ValueError("Positions contain non-finite values.")

    # --------------------------------------------------
    # 1. Define spatial grid
    # --------------------------------------------------

    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)

    extent = maximum - minimum
    box_size = float(extent.max())

    if box_size <= 0:
        raise ValueError("Invalid spatial extent.")

    voxel_size = box_size / grid_size

    center = (minimum + maximum) / 2.0

    grid_minimum = center - box_size / 2.0
    grid_maximum = center + box_size / 2.0

    edges = [
        np.linspace(
            grid_minimum[i],
            grid_maximum[i],
            grid_size + 1,
        )
        for i in range(3)
    ]

    # --------------------------------------------------
    # 2. Count galaxies in voxels
    # --------------------------------------------------

    counts, _ = np.histogramdd(
        positions,
        bins=edges,
    )

    counts = counts.astype(np.float32)

    # --------------------------------------------------
    # 3. Smooth density field
    # --------------------------------------------------

    sigma = smoothing / voxel_size

    smoothed = gaussian_filter(
        counts,
        sigma=sigma,
        mode="constant",
    )

    # --------------------------------------------------
    # 4. Calculate provisional relative density
    # --------------------------------------------------

    occupied = counts > 0

    if not np.any(occupied):
        raise ValueError("No occupied voxels.")

    mean_density = float(
        smoothed[occupied].mean()
    )

    if mean_density <= 0:
        raise ValueError("Invalid mean density.")

    relative_density = (
        smoothed / mean_density
    ).astype(np.float32)

    # --------------------------------------------------
    # 5. Metadata
    # --------------------------------------------------

    metadata = {

        "grid_size": int(grid_size),

        "coordinate_system": "comoving_cartesian",

        "coordinate_unit": "Mpc",

        "grid_minimum_mpc": grid_minimum.tolist(),

        "grid_maximum_mpc": grid_maximum.tolist(),

        "grid_center_mpc": center.tolist(),

        "box_size_mpc": box_size,

        "voxel_size_mpc": voxel_size,

        "smoothing_scale_mpc": smoothing,

        "sigma_voxels": float(sigma),

        "galaxy_count": int(len(positions)),

        "occupied_voxels": int(occupied.sum()),

        "mean_smoothed_occupied_density": mean_density,

        "normalization": "occupied_voxel_mean",

        "selection_correction": False,

        "survey_mask_available": False,

        "density_interpretation": (
            "Provisional relative observed galaxy density. "
            "Not a selection-corrected matter density field."
        ),
    }

    return relative_density, metadata


def calculate_density_contrast(density):

    density = np.asarray(density, dtype=np.float32)

    mean_density = float(density.mean())

    if mean_density <= 0:
        raise ValueError("Invalid mean density.")

    contrast = (
        density / mean_density
    ) - 1.0

    return contrast.astype(np.float32)

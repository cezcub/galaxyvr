import json
import struct

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter

from structure.config import CONFIG, INPUT_FILE, PROJECT_ROOT

MAGIC = 0x44534944  # DSID

OUTPUT_DIR = (
    PROJECT_ROOT
    / "UnityProject"
    / "Assets"
    / "StreamingAssets"
    / "DESI"
)

OUTPUT_VOLUME = OUTPUT_DIR / "density.bin"
OUTPUT_METADATA = OUTPUT_DIR / "density_metadata.json"


def load_positions():

    print("Loading processed DESI catalog...")

    required = [
        "x_mpc",
        "y_mpc",
        "z_cart_mpc",
    ]

    columns = required

    df = pd.read_parquet(
        INPUT_FILE,
        columns=columns,
    )

    positions = df[
        required
    ].to_numpy(dtype=np.float64)

    valid = np.all(
        np.isfinite(positions),
        axis=1,
    )

    positions = positions[valid]

    if len(positions) == 0:
        raise ValueError("No valid galaxy positions found.")

    print(f"Valid galaxies: {len(positions):,}")

    return positions


def build_density_field(positions):

    print("\nBuilding density field...")

    settings = CONFIG["density"]

    grid_size = int(settings["grid_size"])

    smoothing_mpc = float(
        settings["smoothing_scale_mpc"]
    )

    if grid_size < 2:
        raise ValueError("Grid size must be at least 2.")

    if smoothing_mpc < 0:
        raise ValueError("Smoothing scale cannot be negative.")

    # Determine spatial bounds.
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)

    extent = maximum - minimum

    box_size = float(extent.max())

    if not np.isfinite(box_size) or box_size <= 0:
        raise ValueError("Invalid spatial extent.")

    center = (minimum + maximum) / 2.0

    grid_min = center - box_size / 2.0

    voxel_size = box_size / grid_size

    print(f"Box size: {box_size:.2f} Mpc")
    print(f"Grid: {grid_size}³")
    print(f"Voxel size: {voxel_size:.2f} Mpc")

    # Convert positions to voxel coordinates.
    voxel_positions = np.floor(
        (positions - grid_min) / voxel_size
    ).astype(np.int64)

    voxel_positions = np.clip(
        voxel_positions,
        0,
        grid_size - 1,
    )

    # Construct galaxy count grid.
    density = np.zeros(
        (grid_size, grid_size, grid_size),
        dtype=np.float32,
    )

    np.add.at(
        density,
        (
            voxel_positions[:, 0],
            voxel_positions[:, 1],
            voxel_positions[:, 2],
        ),
        1.0,
    )

    # Gaussian smoothing.
    sigma_voxels = smoothing_mpc / voxel_size

    if sigma_voxels > 0:

        density = gaussian_filter(
            density,
            sigma=sigma_voxels,
            mode="constant",
        )

    # Normalize to mean density.
    #
    # This creates a dimensionless overdensity-like field.
    # Values near 1 represent the mean density of the
    # sampled, smoothed grid.

    mean_density = float(density.mean())

    if mean_density <= 0:
        raise ValueError("Density field is empty.")

    density /= mean_density

    # Convert to Unity-compatible axis order.
    # Keep the physical coordinate ordering consistent:
    # X, Y, Z.

    density = np.ascontiguousarray(
        density,
        dtype="<f4",
    )

    metadata = {
        "format": "DESI_DENSITY_V1",
        "grid_size": grid_size,
        "dimensions": [
            grid_size,
            grid_size,
            grid_size,
        ],
        "axis_order": "XYZ",
        "box_size_mpc": box_size,
        "voxel_size_mpc": voxel_size,
        "grid_min_mpc": grid_min.tolist(),
        "grid_max_mpc": (
            grid_min + box_size
        ).tolist(),
        "smoothing_scale_mpc": smoothing_mpc,
        "sigma_voxels": sigma_voxels,
        "mean_density_before_normalization": mean_density,
        "normalization": "global_grid_mean",
        "density_type": "smoothed_galaxy_number_overdensity",
        "galaxy_count": int(len(positions)),
        "data_type": "float32",
        "endianness": "little",
    }

    print(f"Mean density before normalization: {mean_density:.6f}")
    print(f"Density range: {density.min():.6f} - {density.max():.6f}")

    return density, metadata


def export_density_data(density, metadata):

    print("\nExporting density volume...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid_size = metadata["grid_size"]

    expected_shape = (
        grid_size,
        grid_size,
        grid_size,
    )

    if density.shape != expected_shape:
        raise ValueError(
            f"Unexpected density shape: {density.shape}"
        )

    density = np.asarray(
        density,
        dtype="<f4",
        order="C",
    )

    with open(OUTPUT_VOLUME, "wb") as f:

        # Header:
        # uint32 magic
        # uint32 grid size
        # float32 voxel size (Mpc)

        f.write(
            struct.pack(
                "<IIf",
                MAGIC,
                grid_size,
                metadata["voxel_size_mpc"],
            )
        )

        density.tofile(f)

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    print(f"Density volume: {OUTPUT_VOLUME}")
    print(f"Metadata: {OUTPUT_METADATA}")

    expected_bytes = (
        12 + density.size * 4
    )

    actual_bytes = OUTPUT_VOLUME.stat().st_size

    if actual_bytes != expected_bytes:
        raise IOError("Density export size verification failed.")

    print(
        f"Binary size: {actual_bytes / (1024 ** 2):.2f} MB"
    )


def export_density():

    print("\n================================")
    print("       DESI DENSITY EXPORT")
    print("================================")

    positions = load_positions()

    density, metadata = build_density_field(
        positions
    )

    export_density_data(
        density,
        metadata,
    )

    print("\nDensity export complete!")


if __name__ == "__main__":
    export_density()

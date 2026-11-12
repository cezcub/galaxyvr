import json

import numpy as np
import pyarrow.parquet as pq

from structure.config import CONFIG, INPUT_FILE, PROJECT_ROOT

METADATA_VERSION = 2

COORDINATE_COLUMNS = [
    "x_mpc",
    "y_mpc",
    "z_cart_mpc",
]

REDSHIFT_COLUMN = "Z"

BATCH_SIZE = 500_000

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "UnityProject"
    / "Assets"
    / "StreamingAssets"
    / "DESI"
)

def get_output_directory():

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    return OUTPUT_DIRECTORY

def validate_catalog_schema(parquet_file):

    schema = parquet_file.schema_arrow

    required = (
        COORDINATE_COLUMNS
        + [REDSHIFT_COLUMN]
    )

    missing = [
        column
        for column in required
        if column not in schema.names
    ]

    if missing:
        raise ValueError(
            f"Missing catalog columns: {missing}"
        )

def calculate_catalog_bounds():

    print("Scanning DESI catalog...")

    parquet_file = pq.ParquetFile(INPUT_FILE)

    validate_catalog_schema(parquet_file)

    selection = CONFIG["selection"]

    min_redshift = float(
        selection["min_redshift"]
    )

    max_redshift = float(
        selection["max_redshift"]
    )

    if max_redshift <= min_redshift:
        raise ValueError(
            "Invalid redshift selection."
        )

    spatial_min = np.full(
        3,
        np.inf,
        dtype=np.float64,
    )

    spatial_max = np.full(
        3,
        -np.inf,
        dtype=np.float64,
    )

    valid_count = 0
    rejected_count = 0

    columns = (
        COORDINATE_COLUMNS
        + [REDSHIFT_COLUMN]
    )

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=columns,
    ):

        positions = np.column_stack([
            batch.column(
                batch.schema.get_field_index(column)
            ).to_numpy(zero_copy_only=False)
            for column in COORDINATE_COLUMNS
        ]).astype(np.float64, copy=False)

        redshift = batch.column(
            batch.schema.get_field_index(REDSHIFT_COLUMN)
        ).to_numpy(zero_copy_only=False)

        valid = (
            np.isfinite(positions).all(axis=1)
            & np.isfinite(redshift)
            & (redshift > min_redshift)
            & (redshift <= max_redshift)
        )

        valid_positions = positions[valid]

        valid_count += len(valid_positions)
        rejected_count += len(positions) - len(valid_positions)

        if len(valid_positions) == 0:
            continue

        spatial_min = np.minimum(
            spatial_min,
            valid_positions.min(axis=0),
        )

        spatial_max = np.maximum(
            spatial_max,
            valid_positions.max(axis=0),
        )

    if valid_count == 0:
        raise RuntimeError(
            "No valid galaxies found."
        )

    print(f"Valid galaxies: {valid_count:,}")
    print(f"Rejected galaxies: {rejected_count:,}")

    return {
        "spatial_min": spatial_min,
        "spatial_max": spatial_max,
        "valid_count": valid_count,
        "rejected_count": rejected_count,
    }

def calculate_grid_geometry(
    spatial_min,
    spatial_max,
):

    grid_size = int(
        CONFIG["density"]["grid_size"]
    )

    if grid_size <= 0:
        raise ValueError(
            "Grid size must be positive."
        )

    extent = spatial_max - spatial_min

    box_size = float(
        extent.max()
    )

    if not np.isfinite(box_size) or box_size <= 0:
        raise ValueError(
            "Invalid spatial extent."
        )

    center = (
        spatial_min + spatial_max
    ) / 2.0

    # All exported spatial products use
    # the same cubic coordinate system.

    grid_min = center - box_size / 2.0
    grid_max = center + box_size / 2.0

    voxel_size = box_size / grid_size

    return {
        "grid_size": grid_size,
        "grid_shape": [
            grid_size,
            grid_size,
            grid_size,
        ],
        "box_size_mpc": box_size,
        "voxel_size_mpc": voxel_size,
        "center_mpc": center,
        "grid_min_mpc": grid_min,
        "grid_max_mpc": grid_max,
    }

def build_coordinate_metadata(
    center,
    grid_min,
    grid_max,
    scale,
):

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            "Unity scale must be positive and finite."
        )

    unity_min = (
        grid_min - center
    ) * scale

    unity_max = (
        grid_max - center
    ) * scale

    return {

        "source_unit": "Mpc",
        "unity_unit": "Unity units",

        "unity_units_per_mpc": scale,

        "origin_mpc": center.tolist(),

        "origin_definition": (
            "Center of the shared cubic spatial grid."
        ),

        "axis_mapping": {
            "unity_x": "x_mpc",
            "unity_y": "y_mpc",
            "unity_z": "z_cart_mpc",
        },

        "transform": (
            "unity_position = "
            "(catalog_position_mpc - origin_mpc) "
            "* unity_units_per_mpc"
        ),

        "grid_min_mpc": grid_min.tolist(),
        "grid_max_mpc": grid_max.tolist(),

        "grid_min_unity": unity_min.tolist(),
        "grid_max_unity": unity_max.tolist(),

        "unity_box_size": float(
            (grid_max[0] - grid_min[0]) * scale
        ),
    }

def build_metadata():

    print("\n=== BUILDING DESI VR METADATA ===\n")

    bounds = calculate_catalog_bounds()

    geometry = calculate_grid_geometry(
        bounds["spatial_min"],
        bounds["spatial_max"],
    )

    scale = float(
        CONFIG["visualization"]["unity_units_per_mpc"]
    )

    coordinates = build_coordinate_metadata(
        geometry["center_mpc"],
        geometry["grid_min_mpc"],
        geometry["grid_max_mpc"],
        scale,
    )

    density_settings = CONFIG["density"]

    selection = CONFIG["selection"]

    metadata = {


        "format": "DESI_VR_SPATIAL_METADATA",

        "version": METADATA_VERSION,

        "source": {

            "catalog": str(INPUT_FILE),

            "catalog_format": "Parquet",

            "valid_galaxies": bounds["valid_count"],

            "rejected_rows": bounds["rejected_count"],

        },

        "selection": {

            "minimum_redshift": float(
                selection["min_redshift"]
            ),

            "maximum_redshift": float(
                selection["max_redshift"]
            ),

            "redshift_column": REDSHIFT_COLUMN,

            "coordinate_columns": COORDINATE_COLUMNS,

            "redshift_interpretation": (
                "Observed catalog redshift."
            ),

        },

        "spatial_bounds": {

            "catalog_min_mpc": (
                bounds["spatial_min"].tolist()
            ),

            "catalog_max_mpc": (
                bounds["spatial_max"].tolist()
            ),

            "grid_min_mpc": (
                geometry["grid_min_mpc"].tolist()
            ),

            "grid_max_mpc": (
                geometry["grid_max_mpc"].tolist()
            ),

            "center_mpc": (
                geometry["center_mpc"].tolist()
            ),

            "box_size_mpc": (
                geometry["box_size_mpc"]
            ),

        },

        "grid": {

            "dimensions": geometry["grid_shape"],

            "resolution": geometry["grid_size"],

            "voxel_size_mpc": (
                geometry["voxel_size_mpc"]
            ),

            "smoothing_scale_mpc": float(
                density_settings["smoothing_scale_mpc"]
            ),

            "axis_order": ["x", "y", "z"],

            "flattening_order": "C",

        },

        "coordinates": coordinates,

        "cosmology": {

            "H0": float(
                CONFIG["cosmology"]["H0"]
            ),

            "omega_m": float(
                CONFIG["cosmology"]["omega_m"]
            ),

            "distance_type": (
                "Comoving distance"
            ),

        },

        "binary_conventions": {

            "endianness": "little",

            "floating_point": "float32",

            "integer_types": {

                "uint8": "8-bit unsigned integer",

                "uint32": "32-bit unsigned integer",

                "uint64": "64-bit unsigned integer",

            },

            "density_storage": "float32",

            "structure_storage": "uint8",

            "tweb_storage": "float32",

        },

        "export_products": {

            "galaxies": {

                "file": "galaxies.bin",

                "description": (
                    "Sampled DESI galaxy positions and redshifts."
                ),

            },

            "density": {

                "file": "density.bin",

                "description": (
                    "Smoothed normalized galaxy density field."
                ),

            },

            "tweb": {

                "file": "tweb.bin",

                "description": (
                    "Tidal tensor eigenvalues and "
                    "web classification."
                ),

            },

            "structures": {

                "file": "structures.bin",

                "description": (
                    "Discrete cosmic web structure classification."
                ),

            },

        },

    }

    return metadata

def save_metadata(metadata):

    output_dir = get_output_directory()

    output_file = (
        output_dir / "spatial_metadata.json"
    )

    temporary_file = output_file.with_suffix(
        ".json.tmp"
    )

    with open(
        temporary_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            allow_nan=False,
        )

    temporary_file.replace(
        output_file
    )

    return output_file

def load_metadata():

    output_file = (
        get_output_directory()
        / "spatial_metadata.json"
    )

    if not output_file.exists():
        raise FileNotFoundError(
            "Spatial metadata not found. "
            "Run metadata.py first."
        )

    with open(
        output_file,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    if metadata.get("version") != METADATA_VERSION:
        raise ValueError(
            "Unsupported metadata version."
        )

    return metadata

def export_metadata():

    metadata = build_metadata()

    output = save_metadata(metadata)

    print("\n=== METADATA EXPORT COMPLETE ===\n")

    print(
        f"Valid galaxies: "
        f"{metadata['source']['valid_galaxies']:,}"
    )

    print(
        f"Grid dimensions: "
        f"{metadata['grid']['dimensions']}"
    )

    print(
        f"Box size: "
        f"{metadata['spatial_bounds']['box_size_mpc']:.2f} Mpc"
    )

    print(
        f"Voxel size: "
        f"{metadata['grid']['voxel_size_mpc']:.2f} Mpc"
    )

    print(
        f"Unity scale: "
        f"{metadata['coordinates']['unity_units_per_mpc']} units/Mpc"
    )

    print(f"\nOutput: {output}")


if __name__ == "__main__":
    export_metadata()

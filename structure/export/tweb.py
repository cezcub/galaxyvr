import struct

import numpy as np
from pathlib import Path

from structure.config import CONFIG, PROJECT_ROOT
from structure.export.metadata import (
    load_metadata,
    get_output_directory,
)

TWEB_INPUT_FILE = (
    PROJECT_ROOT
    / CONFIG["paths"].get(
        "structure_map",
        "output/structure_map.npy",
    )
)

OUTPUT_FILENAME = "tweb.bin"

# Binary format identifier: TWEB
MAGIC = 0x54454257

FORMAT_VERSION = 1

CLASS_LABELS = {
    0: "Void",
    1: "Sheet",
    2: "Filament",
    3: "Knot",
}

def validate_tweb(tweb, metadata):

    if not isinstance(tweb, np.ndarray):
        raise TypeError(
            "T-web data must be a NumPy array."
        )

    if tweb.ndim != 3:
        raise ValueError(
            f"Expected a 3D T-web map, got {tweb.ndim}D."
        )

    expected_shape = tuple(
        metadata["grid"]["dimensions"]
    )

    if tweb.shape != expected_shape:
        raise ValueError(
            f"T-web dimensions {tweb.shape} do not match "
            f"metadata dimensions {expected_shape}."
        )

    if not np.issubdtype(
        tweb.dtype,
        np.integer,
    ):
        raise TypeError(
            "T-web classification must contain integers."
        )

    if not np.isfinite(tweb).all():
        raise ValueError(
            "T-web map contains invalid values."
        )

    unique_values = np.unique(tweb)

    invalid = [
        int(value)
        for value in unique_values
        if value not in CLASS_LABELS
    ]

    if invalid:
        raise ValueError(
            f"Unexpected T-web classification values: {invalid}"
        )

    print("T-web validation successful.")

    print(
        "Class distribution:"
    )

    for class_id, label in CLASS_LABELS.items():

        count = int(
            np.count_nonzero(tweb == class_id)
        )

        percentage = (
            100.0 * count / tweb.size
        )

        print(
            f"  {label:<12} "
            f"{count:>12,} "
            f"({percentage:.2f}%)"
        )

def write_tweb_binary(tweb, output_file):

    print("\nWriting T-web binary...")

    # Ensure contiguous, row-major uint8 data.
    tweb = np.ascontiguousarray(
        tweb,
        dtype=np.uint8,
    )

    nx, ny, nz = tweb.shape

    temporary_file = output_file.with_suffix(
        ".bin.tmp"
    )

    with open(temporary_file, "wb") as f:

        # Header:
        # uint32 magic
        # uint32 version
        # uint32 nx
        # uint32 ny
        # uint32 nz

        f.write(
            struct.pack(
                "<5I",
                MAGIC,
                FORMAT_VERSION,
                nx,
                ny,
                nz,
            )
        )

        # One uint8 classification per voxel.
        tweb.tofile(f)

    temporary_file.replace(output_file)

    return output_file

def export_tweb():

    print("\n=== DESI T-WEB EXPORT ===\n")\

    if not TWEB_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"T-web input not found: {TWEB_INPUT_FILE}\n"
            "Run the T-web calculation before exporting."
        )

    # Load shared spatial metadata.
    metadata = load_metadata()

    # Load existing T-web classification.
    print("Loading T-web classification...")

    tweb = np.load(
        TWEB_INPUT_FILE,
        mmap_mode="r",
    )

    print(f"Input shape: {tweb.shape}")
    print(f"Input dtype: {tweb.dtype}")

    # Validate against the shared grid.
    validate_tweb(
        tweb,
        metadata,
    )

    # Output directory shared by all exporters.
    output_dir = get_output_directory()

    output_file = (
        output_dir / OUTPUT_FILENAME
    )

    # Export classification grid.
    write_tweb_binary(
        tweb,
        output_file,
    )

    # Calculate output size.
    size_mb = (
        output_file.stat().st_size
        / (1024 ** 2)
    )

    print("\n=== T-WEB EXPORT COMPLETE ===")

    print(f"Grid dimensions: {tweb.shape}")
    print(f"Voxel count: {tweb.size:,}")
    print(f"Output size: {size_mb:.2f} MB")
    print(f"Output: {output_file}")

    print("\nClassification mapping:")

    for class_id, label in CLASS_LABELS.items():

        print(f"  {class_id}: {label}")

    return output_file

if __name__ == "__main__":
    export_tweb()
import json
import struct

import numpy as np

from structure.config import PROJECT_ROOT


INPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "UnityProject"
    / "Assets"
    / "StreamingAssets"
    / "DESI"
)

STRUCTURE_INPUT = INPUT_DIR / "structure_map.npy"

OUTPUT_STRUCTURE = OUTPUT_DIR / "structures.bin"
OUTPUT_METADATA = OUTPUT_DIR / "structures_metadata.json"

MAGIC = 0x44534953  # DSIS

STRUCTURE_NAMES = {
    0: "Void",
    1: "Sheet",
    2: "Filament",
    3: "Knot",
}


def load_structure_map():

    print("Loading T-Web structure map...")

    if not STRUCTURE_INPUT.exists():
        raise FileNotFoundError(
            f"Structure map not found: {STRUCTURE_INPUT}"
        )

    structure_map = np.load(
        STRUCTURE_INPUT,
        mmap_mode="r",
    )

    if structure_map.ndim != 3:
        raise ValueError(
            "Structure map must be a 3D array."
        )

    if not all(size == structure_map.shape[0]
               for size in structure_map.shape):
        raise ValueError(
            "Structure map must have cubic dimensions."
        )

    unique = np.unique(structure_map)

    if not np.all(np.isin(unique, [0, 1, 2, 3])):
        raise ValueError(
            f"Unexpected structure classifications: {unique}"
        )

    print(f"Grid dimensions: {structure_map.shape}")
    print(f"Classifications: {unique}")

    return structure_map


def calculate_statistics(structure_map):

    print("\nCalculating structure statistics...")

    total = structure_map.size

    statistics = {}

    for class_id, name in STRUCTURE_NAMES.items():

        count = int(
            np.count_nonzero(
                structure_map == class_id
            )
        )

        statistics[name.lower()] = {
            "class_id": class_id,
            "voxel_count": count,
            "volume_fraction": count / total,
        }

    return statistics


def export_structure_map(structure_map, statistics):

    print("\nExporting structure classifications...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid_size = structure_map.shape[0]

    # Preserve the original classification IDs.
    data = np.ascontiguousarray(
        structure_map,
        dtype=np.uint8,
    )

    # Binary header:
    # uint32 magic
    # uint32 grid size
    # uint8 classification data

    with open(OUTPUT_STRUCTURE, "wb") as f:

        f.write(
            struct.pack(
                "<II",
                MAGIC,
                grid_size,
            )
        )

        data.tofile(f)

    metadata = {
        "format": "DESI_STRUCTURES_V1",
        "grid_size": grid_size,
        "dimensions": list(data.shape),
        "axis_order": "XYZ",
        "data_type": "uint8",
        "classifications": STRUCTURE_NAMES,
        "statistics": statistics,
    }

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

    expected_bytes = 8 + data.size

    actual_bytes = OUTPUT_STRUCTURE.stat().st_size

    if actual_bytes != expected_bytes:
        raise IOError(
            "Structure export size verification failed."
        )

    print(f"Output: {OUTPUT_STRUCTURE}")
    print(
        f"Binary size: {actual_bytes / (1024 ** 2):.2f} MB"
    )


def export_structures():

    print("\n================================")
    print("      DESI STRUCTURE EXPORT")
    print("================================")

    structure_map = load_structure_map()

    statistics = calculate_statistics(
        structure_map
    )

    export_structure_map(
        structure_map,
        statistics,
    )

    print("\nStructure statistics:")

    for name, values in statistics.items():

        print(
            f"{name.capitalize():12s}: "
            f"{values['volume_fraction'] * 100:.2f}%"
        )

    print("\nStructure export complete!")


if __name__ == "__main__":
    export_structures()

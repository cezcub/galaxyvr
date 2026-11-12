import json
import struct

import numpy as np
import pyarrow.parquet as pq

from structure.config import CONFIG, INPUT_FILE, PROJECT_ROOT
from structure.export.metadata import load_metadata, get_output_directory

MAGIC = b"DSIG"
VERSION = 1

HEADER_FORMAT = "<4sII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

RECORD_DTYPE = np.dtype([
    ("x", "<f4"),
    ("y", "<f4"),
    ("z", "<f4"),
    ("redshift", "<f4"),
    ("targetid", "<u8"),
], align=False)

RECORD_SIZE = RECORD_DTYPE.itemsize

BATCH_SIZE = 500_000


def load_export_metadata():

    metadata = load_metadata()

    return {
        "origin": np.asarray(
            metadata["coordinates"]["origin_mpc"],
            dtype=np.float64,
        ),

        "scale": float(
            metadata["coordinates"]["unity_units_per_mpc"]
        ),

        "min_redshift": float(
            metadata["selection"]["minimum_redshift"]
        ),

        "max_redshift": float(
            metadata["selection"]["maximum_redshift"]
        ),
    }


def validate_schema(parquet_file):

    required = [
        "TARGETID",
        "Z",
        "x_mpc",
        "y_mpc",
        "z_cart_mpc",
    ]

    available = parquet_file.schema_arrow.names

    missing = [
        column for column in required
        if column not in available
    ]

    if missing:
        raise ValueError(
            f"Missing Parquet columns: {missing}. "
            "Update preprocessing to retain TARGETID."
        )


def collect_galaxies(metadata):

    print("Reading DESI galaxy catalog...")

    parquet_file = pq.ParquetFile(INPUT_FILE)

    validate_schema(parquet_file)

    columns = [
        "TARGETID",
        "Z",
        "x_mpc",
        "y_mpc",
        "z_cart_mpc",
    ]

    min_z = metadata["min_redshift"]
    max_z = metadata["max_redshift"]

    origin = metadata["origin"]
    scale = metadata["scale"]

    settings = CONFIG["visualization"]

    max_galaxies = settings.get("max_galaxies")
    seed = int(settings.get("random_seed", 42))

    rng = np.random.default_rng(seed)

    # Reservoir sampling allows deterministic sampling
    # without retaining the entire catalog in memory.
    reservoir = None
    seen = 0
    valid_count = 0

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=columns,
    ):

        table = batch.to_pydict()

        redshift = np.asarray(
            table["Z"],
            dtype=np.float64,
        )

        positions = np.column_stack([
            table["x_mpc"],
            table["y_mpc"],
            table["z_cart_mpc"],
        ]).astype(np.float64)

        targetid = np.asarray(
            table["TARGETID"],
            dtype=np.uint64,
        )

        valid = (
            np.isfinite(positions).all(axis=1)
            & np.isfinite(redshift)
            & (redshift > min_z)
            & (redshift <= max_z)
        )

        positions = positions[valid]
        redshift = redshift[valid]
        targetid = targetid[valid]

        if len(positions) == 0:
            continue

        # Transform into Unity coordinates.
        positions = (
            positions - origin
        ) * scale

        if not np.isfinite(positions).all():
            raise ValueError(
                "Invalid transformed galaxy coordinates."
            )

        records = np.empty(
            len(positions),
            dtype=RECORD_DTYPE,
        )

        records["x"] = positions[:, 0]
        records["y"] = positions[:, 1]
        records["z"] = positions[:, 2]
        records["redshift"] = redshift
        records["targetid"] = targetid

        if max_galaxies is None or max_galaxies <= 0:

            if reservoir is None:
                reservoir = records
            else:
                reservoir = np.concatenate(
                    (reservoir, records)
                )

        else:

            if reservoir is None:
                reservoir = np.empty(
                    max_galaxies,
                    dtype=RECORD_DTYPE,
                )

            for record in records:

                if seen < max_galaxies:
                    reservoir[seen] = record

                else:
                    index = rng.integers(
                        0,
                        seen + 1,
                    )

                    if index < max_galaxies:
                        reservoir[index] = record

                seen += 1

        valid_count += len(records)

        if max_galaxies is None or max_galaxies <= 0:
            seen = valid_count

        print(
            f"Processed {valid_count:,} valid galaxies",
            end="\r",
        )

    if reservoir is None or len(reservoir) == 0:
        raise RuntimeError(
            "No galaxies available for export."
        )

    if max_galaxies is not None and max_galaxies > 0:
        reservoir = reservoir[:min(valid_count, max_galaxies)]

    print()
    print(f"Valid galaxies: {valid_count:,}")
    print(f"Exported galaxies: {len(reservoir):,}")

    return reservoir, valid_count


def write_binary(records, output):

    count = len(records)

    with open(output, "wb") as f:

        f.write(
            struct.pack(
                HEADER_FORMAT,
                MAGIC,
                VERSION,
                count,
            )
        )

        records.tofile(f)

    expected_size = (
        HEADER_SIZE + count * RECORD_SIZE
    )

    if output.stat().st_size != expected_size:
        raise IOError(
            "Galaxy binary file size verification failed."
        )


def write_manifest(
    records,
    valid_count,
    output,
    binary_name,
):

    metadata = load_metadata()

    manifest = {

        "format": "DESI_GALAXIES",

        "version": VERSION,

        "binary_file": binary_name,

        "source": str(INPUT_FILE),

        "total_valid_galaxies": int(valid_count),

        "exported_galaxies": int(len(records)),

        "sampled": len(records) < valid_count,

        "record_size_bytes": RECORD_SIZE,

        "header_size_bytes": HEADER_SIZE,

        "endianness": "little",

        "fields": [
            {"name": "x", "type": "float32", "offset": 0},
            {"name": "y", "type": "float32", "offset": 4},
            {"name": "z", "type": "float32", "offset": 8},
            {"name": "redshift", "type": "float32", "offset": 12},
            {"name": "targetid", "type": "uint64", "offset": 16},
        ],

        "coordinate_system": metadata["coordinates"],

        "redshift_statistics": {
            "minimum": float(records["redshift"].min()),
            "maximum": float(records["redshift"].max()),
            "mean": float(records["redshift"].mean()),
        },
    }

    with open(output, "w") as f:
        json.dump(manifest, f, indent=4)


def export_galaxies():

    print("\n=== DESI GALAXY EXPORT ===\n")

    metadata = load_export_metadata()

    records, valid_count = collect_galaxies(
        metadata
    )

    output_dir = get_output_directory()

    binary_output = (
        output_dir / "galaxies.bin"
    )

    manifest_output = (
        output_dir / "galaxies_metadata.json"
    )

    print("Writing galaxy binary...")

    write_binary(
        records,
        binary_output,
    )

    write_manifest(
        records,
        valid_count,
        manifest_output,
        binary_output.name,
    )

    print("\n=== GALAXY EXPORT COMPLETE ===")

    print(f"Valid: {valid_count:,}")
    print(f"Exported: {len(records):,}")
    print(f"Record size: {RECORD_SIZE} bytes")

    print(
        f"Binary size: "
        f"{binary_output.stat().st_size / 1e6:.2f} MB"
    )

    print(f"Binary: {binary_output}")
    print(f"Manifest: {manifest_output}")


if __name__ == "__main__":
    export_galaxies()

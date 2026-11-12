from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits

from preprocessing.coord_conversion import add_3d_positions


INPUT_FILE = Path("data/DESIDR1.fits")
OUTPUT_FILE = Path("data/DESIDR1_processed.parquet")

CHUNK_SIZE = 500_000

H0 = 70.0
OMEGA_M = 0.3
SCALE_UNITS_PER_MPC = 0.001

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def preprocess_catalog():
    writer = None

    stats = {
        "input_rows": 0,
        "invalid_coordinates_or_redshift": 0,
        "outside_redshift_range": 0,
        "non_galaxy": 0,
        "redshift_warning": 0,
        "retained_rows": 0,
    }

    try:
        with fits.open(INPUT_FILE, memmap=True) as hdul:
            table = hdul["ZCATALOG"].data
            names = set(table.names)
            total_rows = len(table)

            ra_col = (
                "TARGET_RA" if "TARGET_RA" in names else "RA"
            )
            dec_col = (
                "TARGET_DEC" if "TARGET_DEC" in names else "DEC"
            )

            required = {ra_col, dec_col, "Z"}
            missing = required - names

            if missing:
                raise ValueError(
                    f"Missing required columns: {sorted(missing)}"
                )

            print(f"Input rows: {total_rows:,}")

            for start in range(0, total_rows, CHUNK_SIZE):
                stop = min(start + CHUNK_SIZE, total_rows)
                sl = slice(start, stop)

                # Convert FITS values to native numeric arrays.
                ra = np.asarray(
                    table[ra_col][sl], dtype=np.float64
                )
                dec = np.asarray(
                    table[dec_col][sl], dtype=np.float64
                )
                redshift = np.asarray(
                    table["Z"][sl], dtype=np.float64
                )

                stats["input_rows"] += len(ra)

                # Stage 1: reject invalid measurements.
                valid = (
                    np.isfinite(ra)
                    & np.isfinite(dec)
                    & np.isfinite(redshift)
                    & (ra >= 0.0)
                    & (ra < 360.0)
                    & (dec >= -90.0)
                    & (dec <= 90.0)
                    & (redshift > 0.0)
                )

                stats["invalid_coordinates_or_redshift"] += int(
                    np.count_nonzero(~valid)
                )

                # Stage 2: keep only primary observations

                if "ZCAT_PRIMARY" in names:
                    primary = np.asarray(
                        table["ZCAT_PRIMARY"][sl], dtype=bool
                    )
                    valid &= primary

                # Stage 3: keep galaxies if classification exists.
                spectype = None

                if "SPECTYPE" in names:
                    spectype = np.char.strip(
                        np.asarray(
                            table["SPECTYPE"][sl]
                        ).astype(str)
                    )

                    is_galaxy = spectype == "GALAXY"

                    stats["non_galaxy"] += int(
                        np.count_nonzero(valid & ~is_galaxy)
                    )

                    valid &= is_galaxy

                # Stage 4: keep rows with no redshift warning.
                zwarn = None

                if "ZWARN" in names:
                    zwarn = np.asarray(
                        table["ZWARN"][sl], dtype=np.int64
                    )

                    reliable = zwarn == 0

                    stats["redshift_warning"] += int(
                        np.count_nonzero(valid & ~reliable)
                    )

                    valid &= reliable

                indices = np.flatnonzero(valid)

                targetid = np.asarray(
                    table["TARGETID"][sl],
                    dtype=np.uint64,
                )

                if len(indices) == 0:
                    print(f"Processed {stop:,}/{total_rows:,}")
                    continue

                # Keep a compact set of useful source fields.
                chunk = pd.DataFrame({
                    "TARGETID": targetid[indices],
                    "RA": ra[indices],
                    "DEC": dec[indices],
                    "Z": redshift[indices]
                })

                if spectype is not None:
                    chunk["SPECTYPE"] = spectype[indices]

                if zwarn is not None:
                    chunk["ZWARN"] = zwarn[indices]

                # Use your existing cosmological conversion.
                chunk = add_3d_positions(
                    chunk,
                    H0=H0,
                    omega_m=OMEGA_M,
                    scale_units_per_mpc=SCALE_UNITS_PER_MPC,
                )

                # Write each processed chunk immediately.
                arrow_table = pa.Table.from_pandas(
                    chunk,
                    preserve_index=False,
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        OUTPUT_FILE,
                        arrow_table.schema,
                        compression="zstd",
                    )

                writer.write_table(arrow_table)

                stats["retained_rows"] += len(chunk)

                print(
                    f"Processed {stop:,}/{total_rows:,} rows; "
                    f"retained {stats['retained_rows']:,}"
                )

    finally:
        if writer is not None:
            writer.close()

    if stats["retained_rows"] == 0:
        raise RuntimeError(
            "No rows passed the filters; no processed catalog was written."
        )

    print("\nPreprocessing complete")
    print(f"Output: {OUTPUT_FILE}")
    for key, value in stats.items():
        print(f"{key}: {value:,}")


if __name__ == "__main__":
    preprocess_catalog()

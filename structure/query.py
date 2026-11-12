from astropy.io import fits
from collections import Counter
import numpy as np
import sys

FITS_FILE = "data/desiDR1.fits"


def clean_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.bytes_):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def main():
    try:
        with fits.open(FITS_FILE, memmap=True) as hdul:

            print("=" * 65)
            print("DESI FITS DATASET SUMMARY")
            print("=" * 65)

            # Display all FITS extensions.
            hdul.info()

            # Find tables containing an object-classification column.
            candidates = []

            for i, hdu in enumerate(hdul):
                if hdu.data is None:
                    continue

                if not hasattr(hdu, "columns"):
                    continue

                columns = list(hdu.columns.names or [])
                upper_columns = {c.upper(): c for c in columns}

                if "SPECTYPE" in upper_columns:
                    candidates.append(
                        (i, hdu, upper_columns)
                    )

            if not candidates:
                print("\nERROR: No table with a SPECTYPE column found.")
                print("Available table columns:")

                for i, hdu in enumerate(hdul):
                    if getattr(hdu, "columns", None) is not None:
                        if hdu.data is not None:
                            print(
                                f"\nExtension {i} ({hdu.name}):"
                            )
                            print(hdu.columns.names)

                print(
                    "\nCheck the output above to identify the "
                    "appropriate classification column."
                )
                return

            # Prefer a table with TARGETID and SPECTYPE.
            selected = next(
                (
                    item for item in candidates
                    if "TARGETID" in item[2]
                ),
                candidates[0]
            )

            ext_index, hdu, upper_columns = selected
            data = hdu.data

            print(f"\nSelected extension: {ext_index}")
            print(f"Extension name: {hdu.name}")
            print(f"Total rows: {len(data):,}")
            print("\nAvailable columns:")
            print(list(hdu.columns.names))

            # Count all classifications.
            spectype_col = upper_columns["SPECTYPE"]

            classifications = [
                clean_value(x).upper()
                for x in data[spectype_col]
            ]

            counts = Counter(classifications)

            print("\n" + "=" * 65)
            print("OBJECT CLASSIFICATION COUNTS")
            print("=" * 65)

            for obj_type, count in counts.most_common():
                print(f"{obj_type or '(blank)':20s} {count:12,}")

            # Print common classes individually.
            print("\n" + "=" * 65)
            print("MAIN OBJECT COUNTS")
            print("=" * 65)

            for label, aliases in [
                ("Galaxies", {"GALAXY"}),
                ("Quasars", {"QSO"}),
                ("Stars", {"STAR"}),
                ("White dwarfs", {"WD"}),
            ]:
                count = sum(counts.get(alias, 0) for alias in aliases)
                print(f"{label:20s} {count:12,}")

            classified = sum(
                count for obj_type, count in counts.items()
                if obj_type and obj_type not in {
                    "UNKNOWN", "NAN", "NONE", ""
                }
            )

            print(f"\nRows with a non-unknown classification: "
                  f"{classified:,}")

            # Count distinct targets, if TARGETID exists.
            if "TARGETID" in upper_columns:
                target_col = upper_columns["TARGETID"]
                target_ids = data[target_col]
                unique_targets = len(np.unique(target_ids))

                print(f"Unique TARGETID values: {unique_targets:,}")

                print(
                    "\nNote: the classification counts above count "
                    "rows, not necessarily distinct targets."
                )

            # Count quality-flagged rows if ZWARN exists.
            if "ZWARN" in upper_columns:
                zwarn_col = upper_columns["ZWARN"]
                good = data[zwarn_col] == 0

                if "ZCAT_PRIMARY" in upper_columns:
                    primary_col = upper_columns["ZCAT_PRIMARY"]
                    good &= np.asarray(data[primary_col], dtype=bool)

                print("\n" + "=" * 65)
                print("QUALITY-FILTERED COUNTS")
                print("=" * 65)
                print(f"Rows with ZWARN == 0: {np.count_nonzero(good):,}")

                good_counts = Counter(
                    classifications[i]
                    for i in range(len(classifications))
                    if good[i]
                )

                for obj_type, count in good_counts.most_common():
                    print(f"{obj_type or '(blank)':20s} {count:12,}")

            print("\nFinished successfully.")

    except FileNotFoundError:
        print(f"ERROR: File not found: {FITS_FILE}")
        print("Update FITS_FILE with the correct path.")

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

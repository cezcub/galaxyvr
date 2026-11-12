from structure.export.galaxies import export_galaxies

from structure.export.density import export_density

from structure.export.structures import export_structures

from structure.export.tweb import export_tweb

from structure.export.metadata import export_metadata


def run_pipeline():

    print("\n========================================")
    print("       DESI VR EXPORT PIPELINE")
    print("========================================")

    exporters = [
        ("Spatial Metadata", export_metadata),
        ("Galaxy Catalog", export_galaxies),
        ("Density Field", export_density),
        ("Structure Map", export_structures),
        ("T-Web Classification", export_tweb),
    ]

    for name, exporter in exporters:

        print("\n----------------------------------------")
        print(f" EXPORTING: {name.upper()}")
        print("----------------------------------------")

        try:
            exporter()

        except Exception as error:

            print(
                f"\nPIPELINE FAILED during {name} export."
            )

            raise RuntimeError(
                f"Export failed: {name}"
            ) from error

        print(f"\n{name} export successful.")

    print("\n========================================")
    print("       EXPORT PIPELINE COMPLETE")
    print("========================================")

    print(
        "\nAll Unity export products have been generated."
    )


if __name__ == "__main__":
    run_pipeline()

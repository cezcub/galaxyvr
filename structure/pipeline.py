import numpy as np
import json

from structure.catalog import (
    load_galaxies,
    get_positions,
)

from structure.density import build_density_field, calculate_density_contrast

from structure.structures import classify_density

from structure.config import OUTPUT_DIR, CONFIG

from structure.tweb import build_tweb


def run_pipeline():

    print("\n=== COSMIC WEB PIPELINE ===\n")

    # 1. Load catalog.
    galaxies = load_galaxies()

    positions = get_positions(galaxies)

    # Build density
    density, metadata = build_density_field(positions)

    # Calculate provisional contrast
    contrast = calculate_density_contrast(density)

    # Save density
    np.save(
        OUTPUT_DIR / "density_field.npy",
        density,
        )

    np.save(
        OUTPUT_DIR / "density_contrast.npy",
        contrast,
    )

    np.save(
        OUTPUT_DIR / "galaxy_positions.npy",
        positions.astype(np.float32),
    )

    # 4. T-web classification
    structures, eigenvalues, tweb_metadata = build_tweb(
        density=density,
        voxel_size_mpc=metadata["voxel_size_mpc"],
        eigenvalue_threshold=CONFIG["tweb"]["eigenvalue_threshold"],
    )

    # 5. Save T-web structure map
    np.save(
        OUTPUT_DIR / "structure_map.npy",
        structures,
    )

    # Save eigenvalues for scientific analysis
    if CONFIG["tweb"]["save_eigenvalues"]:
        np.save(
            OUTPUT_DIR / "tweb_eigenvalues.npy",
            eigenvalues,
        )

    # Save T-web metadata
    with open(
        OUTPUT_DIR / "tweb_metadata.json", "w") as f:
        json.dump(tweb_metadata, f, indent=4)

    # 6. Save metadata.
    with open(
        OUTPUT_DIR / "density_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n=== PIPELINE COMPLETE ===")
    print(f"Galaxies: {len(galaxies):,}")
    print(f"Density shape: {density.shape}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_pipeline()

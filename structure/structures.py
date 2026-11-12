import numpy as np
from structure.config import CONFIG

def classify_density(density):

    settings = CONFIG["structures"]

    valid = density > 0

    if not np.any(valid):
        raise ValueError("Density field is empty.")

    log_density = np.log10(
        np.maximum(density, 1e-6)
    )

    low_threshold = np.percentile(
        log_density[valid],
        settings["low_density_percentile"],
    )

    high_threshold = np.percentile(
        log_density[valid],
        settings["high_density_percentile"],
    )

    structure_map = np.zeros(
        density.shape,
        dtype=np.uint8,
    )

    structure_map[
        (log_density > low_threshold)
        & (log_density <= high_threshold)
    ] = 1

    structure_map[
        log_density > high_threshold
    ] = 2

    return structure_map

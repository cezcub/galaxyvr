import numpy as np


def calculate_density_contrast(density):
    """
    Calculate dimensionless density contrast.

    Uses the mean of the complete computational grid.
    This is a provisional normalization, not a
    survey-selection-corrected matter density.
    """

    density = np.asarray(density, dtype=np.float32)

    if density.ndim != 3:
        raise ValueError("Density must be a 3D array.")

    if not np.all(np.isfinite(density)):
        raise ValueError("Density contains non-finite values.")

    mean_density = float(density.mean())

    if mean_density <= 0:
        raise ValueError("Mean density must be positive.")

    return (density / mean_density - 1.0).astype(np.float32)


def calculate_tidal_tensor(delta, voxel_size_mpc):
    """
    Calculate the six independent components of the
    symmetric T-web tidal tensor using Fourier methods.

    Convention:
        Laplacian(Phi) = delta
        T_ij = d2(Phi) / dx_i dx_j

    FFTs imply periodic boundary conditions.
    """

    delta = np.asarray(delta, dtype=np.float32)

    if delta.ndim != 3:
        raise ValueError("Density contrast must be 3D.")

    if voxel_size_mpc <= 0:
        raise ValueError("Voxel size must be positive.")

    nx, ny, nz = delta.shape

    # Fourier transform of density contrast.
    delta_k = np.fft.fftn(delta)

    # Wave numbers in inverse Mpc.
    kx = 2.0 * np.pi * np.fft.fftfreq(
        nx, d=voxel_size_mpc
    )

    ky = 2.0 * np.pi * np.fft.fftfreq(
        ny, d=voxel_size_mpc
    )

    kz = 2.0 * np.pi * np.fft.fftfreq(
        nz, d=voxel_size_mpc
    )

    kx = kx[:, None, None]
    ky = ky[None, :, None]
    kz = kz[None, None, :]

    k2 = kx**2 + ky**2 + kz**2

    # Avoid division by zero at k = 0.
    safe_k2 = np.where(k2 == 0, 1.0, k2)

    components = {}

    pairs = {
        "xx": (kx, kx),
        "yy": (ky, ky),
        "zz": (kz, kz),
        "xy": (kx, ky),
        "xz": (kx, kz),
        "yz": (ky, kz),
    }

    for name, (ki, kj) in pairs.items():

        factor = (ki * kj) / safe_k2

        tensor_k = delta_k * factor

        # The zero-frequency mode has no contribution.
        tensor_k[0, 0, 0] = 0.0

        tensor_component = np.fft.ifftn(
            tensor_k
        ).real

        components[name] = tensor_component.astype(
            np.float32
        )

    return components


def classify_tweb(
    components,
    eigenvalue_threshold=0.2,
    slice_depth=16,
):
    """
    Classify cosmic-web environments using the
    number of tidal-tensor eigenvalues above threshold.

    Labels:
        0 = Void
        1 = Sheet
        2 = Filament
        3 = Knot
    """

    xx = components["xx"]
    yy = components["yy"]
    zz = components["zz"]

    xy = components["xy"]
    xz = components["xz"]
    yz = components["yz"]

    nx, ny, nz = xx.shape

    structure_map = np.empty(
        (nx, ny, nz),
        dtype=np.uint8,
    )

    eigenvalues = np.empty(
        (nx, ny, nz, 3),
        dtype=np.float32,
    )

    for start in range(0, nz, slice_depth):

        stop = min(start + slice_depth, nz)

        current_depth = stop - start

        tensor = np.empty(
            (nx, ny, current_depth, 3, 3),
            dtype=np.float32,
        )

        # Diagonal components.
        tensor[..., 0, 0] = xx[:, :, start:stop]
        tensor[..., 1, 1] = yy[:, :, start:stop]
        tensor[..., 2, 2] = zz[:, :, start:stop]

        # Off-diagonal components.
        tensor[..., 0, 1] = xy[:, :, start:stop]
        tensor[..., 1, 0] = xy[:, :, start:stop]

        tensor[..., 0, 2] = xz[:, :, start:stop]
        tensor[..., 2, 0] = xz[:, :, start:stop]

        tensor[..., 1, 2] = yz[:, :, start:stop]
        tensor[..., 2, 1] = yz[:, :, start:stop]

        # Symmetric eigensolver.
        values = np.linalg.eigvalsh(tensor)

        eigenvalues[:, :, start:stop, :] = values

        # Count eigenvalues exceeding threshold.
        collapsed = np.sum(
            values > eigenvalue_threshold,
            axis=-1,
        )

        structure_map[:, :, start:stop] = (
            collapsed.astype(np.uint8)
        )

        print(
            f"T-web classification: {stop}/{nz} slices"
        )

    return structure_map, eigenvalues


def build_tweb(
    density,
    voxel_size_mpc,
    eigenvalue_threshold=0.2,
):
    """
    Complete T-web reconstruction.
    """

    print("\n=== T-WEB CLASSIFICATION ===")

    # 1. Density contrast.
    print("Calculating density contrast...")

    delta = calculate_density_contrast(density)

    # 2. Tidal tensor.
    print("Calculating tidal tensor...")

    components = calculate_tidal_tensor(
        delta,
        voxel_size_mpc,
    )

    # 3. Eigenvalue classification.
    print("Classifying cosmic-web structures...")

    structure_map, eigenvalues = classify_tweb(
        components,
        eigenvalue_threshold=eigenvalue_threshold,
    )

    # 4. Classification statistics.
    labels = {
        0: "void",
        1: "sheet",
        2: "filament",
        3: "knot",
    }

    total = structure_map.size

    statistics = {}

    for label, name in labels.items():

        count = int(
            np.count_nonzero(structure_map == label)
        )

        statistics[name] = {
            "voxels": count,
            "percentage": 100.0 * count / total,
        }

    metadata = {
        "method": "T-web",
        "eigenvalue_threshold": float(
            eigenvalue_threshold
        ),
        "boundary_conditions": "periodic",
        "coordinate_unit": "Mpc",
        "voxel_size_mpc": float(voxel_size_mpc),
        "density_contrast_normalization": (
            "complete_grid_mean"
        ),
        "classes": labels,
        "statistics": statistics,
        "scientific_status": (
            "Exploratory observed-density classification; "
            "not corrected for DESI survey selection."
        ),
    }

    print("\n=== T-WEB RESULTS ===")

    for name, values in statistics.items():

        print(
            f"{name.capitalize():12s}: "
            f"{values['percentage']:.2f}%"
        )

    return structure_map, eigenvalues, metadata
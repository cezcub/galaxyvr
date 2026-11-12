
import numpy as np
import traceback

from structure.tweb import (
    calculate_density_contrast,
    calculate_tidal_tensor,
    classify_tweb,
)


GRID = 32
VOXEL_SIZE = 5.0
THRESHOLD = 0.2

passed = 0
failed = 0


def run_test(name, test_function):

    global passed, failed

    print(f"\nRunning: {name}")

    try:
        test_function()

        print("[PASS]", name)
        passed += 1

    except Exception as error:

        print("[FAIL]", name)
        print(f"Reason: {error}")

        traceback.print_exc()
        failed += 1


# --------------------------------------------------
# TEST 1: Uniform density
# --------------------------------------------------

def test_uniform_density():

    density = np.ones(
        (GRID, GRID, GRID),
        dtype=np.float32,
    )

    delta = calculate_density_contrast(density)

    assert np.allclose(delta, 0.0), \
        "Uniform density should produce zero contrast."

    tensor = calculate_tidal_tensor(
        delta,
        VOXEL_SIZE,
    )

    for component in tensor.values():

        assert np.allclose(
            component,
            0.0,
            atol=1e-6,
        ), "Uniform density should produce zero tidal tensor."


# --------------------------------------------------
# TEST 2: Sinusoidal density wave
# --------------------------------------------------

def test_sinusoidal_wave():

    x = np.arange(GRID)

    wave = 0.5 * np.cos(
        2.0 * np.pi * x / GRID
    )

    delta = np.broadcast_to(
        wave[:, None, None],
        (GRID, GRID, GRID),
    ).copy().astype(np.float32)

    tensor = calculate_tidal_tensor(
        delta,
        VOXEL_SIZE,
    )

    trace = (
        tensor["xx"]
        + tensor["yy"]
        + tensor["zz"]
    )

    assert np.allclose(
        trace,
        delta,
        atol=1e-5,
    ), "Tidal tensor trace does not match density contrast."

    assert np.allclose(
        tensor["yy"],
        0.0,
        atol=1e-5,
    ), "Unexpected y-direction tidal component."

    assert np.allclose(
        tensor["zz"],
        0.0,
        atol=1e-5,
    ), "Unexpected z-direction tidal component."

    structure_map, eigenvalues = classify_tweb(
        tensor,
        eigenvalue_threshold=THRESHOLD,
    )

    assert np.all(np.isfinite(eigenvalues)), \
        "Non-finite eigenvalues detected."

    assert np.any(structure_map == 1), \
        "Expected sheet structures were not detected."

    assert np.any(structure_map == 0), \
        "Expected void regions were not detected."


# --------------------------------------------------
# TEST 3: Random density field
# --------------------------------------------------

def test_random_field():

    rng = np.random.default_rng(42)

    delta = rng.normal(
        0.0,
        0.1,
        size=(GRID, GRID, GRID),
    ).astype(np.float32)

    delta -= delta.mean()

    tensor = calculate_tidal_tensor(
        delta,
        VOXEL_SIZE,
    )

    trace = (
        tensor["xx"]
        + tensor["yy"]
        + tensor["zz"]
    )

    assert np.allclose(
        trace,
        delta,
        atol=1e-5,
    ), "Random field trace consistency failed."

    structure_map, eigenvalues = classify_tweb(
        tensor,
        eigenvalue_threshold=THRESHOLD,
    )

    assert np.all(np.isfinite(eigenvalues)), \
        "Random field produced invalid eigenvalues."

    assert np.all(
        np.isin(structure_map, [0, 1, 2, 3])
    ), "Invalid T-web classification values."


# --------------------------------------------------
# TEST 4: Invalid density input
# --------------------------------------------------

def test_invalid_density():

    density = np.ones(
        (GRID, GRID),
        dtype=np.float32,
    )

    try:

        calculate_density_contrast(density)

    except ValueError:
        return

    raise AssertionError(
        "Invalid density dimensions should raise ValueError."
    )


# --------------------------------------------------
# TEST 5: Empty density field
# --------------------------------------------------

def test_empty_density():

    density = np.zeros(
        (GRID, GRID, GRID),
        dtype=np.float32,
    )

    try:

        calculate_density_contrast(density)

    except ValueError:
        return

    raise AssertionError(
        "Empty density field should raise ValueError."
    )


# --------------------------------------------------
# TEST 6: Classification consistency
# --------------------------------------------------

def test_classification_consistency():

    rng = np.random.default_rng(123)

    delta = rng.normal(
        0.0,
        0.2,
        size=(GRID, GRID, GRID),
    ).astype(np.float32)

    delta -= delta.mean()

    tensor = calculate_tidal_tensor(
        delta,
        VOXEL_SIZE,
    )

    structure_map, eigenvalues = classify_tweb(
        tensor,
        eigenvalue_threshold=THRESHOLD,
    )

    assert structure_map.shape == delta.shape

    assert eigenvalues.shape == (
        GRID,
        GRID,
        GRID,
        3,
    )

    assert np.all(
        np.diff(eigenvalues, axis=-1) >= -1e-6
    ), "Eigenvalues are not sorted in ascending order."

# --------------------------------------------------
# RUN ALL TESTS
# --------------------------------------------------

if __name__ == "__main__":

    print("\n======================================")
    print("       T-WEB VALIDATION SUITE")
    print("======================================")

    run_test(
        "Uniform density",
        test_uniform_density,
    )

    run_test(
        "Sinusoidal density wave",
        test_sinusoidal_wave,
    )

    run_test(
        "Random density field",
        test_random_field,
    )

    run_test(
        "Invalid density input",
        test_invalid_density,
    )

    run_test(
        "Empty density field",
        test_empty_density,
    )

    run_test(
        "Classification consistency",
        test_classification_consistency,
    )

    print("\n======================================")
    print("          VALIDATION RESULTS")
    print("======================================")

    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    if failed == 0:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")

    print("======================================\n")

    if failed > 0:
        raise SystemExit(1)

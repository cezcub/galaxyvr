from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from structure.config import CONFIG, INPUT_FILE, PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "output"
DENSITY_FILE = OUTPUT_DIR / "density_field.npy"
STRUCTURE_FILE = OUTPUT_DIR / "structure_map.npy"
METADATA_FILE = OUTPUT_DIR / "density_metadata.json"


def load_catalog():

    print("\nLoading DESI DR1 catalog...")

    df = pd.read_parquet(INPUT_FILE)

    required = [
        "Z",
        "x_mpc",
        "y_mpc",
        "z_cart_mpc",
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    valid = (
        np.isfinite(df["Z"])
        & np.isfinite(df["x_mpc"])
        & np.isfinite(df["y_mpc"])
        & np.isfinite(df["z_cart_mpc"])
        & (df["Z"] > 0)
    )

    df = df.loc[valid].copy()

    print(f"Available galaxies: {len(df):,}")

    return df


def load_structure_data():

    print("\nLoading density and T-web data...")

    required_files = [
        DENSITY_FILE,
        STRUCTURE_FILE,
        METADATA_FILE,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required pipeline output not found: {path}"
            )

    density = np.load(
        DENSITY_FILE,
        mmap_mode="r",
    )

    structure_map = np.load(
        STRUCTURE_FILE,
        mmap_mode="r",
    )

    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    if density.shape != structure_map.shape:
        raise ValueError(
            "Density and structure maps have different dimensions."
        )

    grid_size = metadata["grid_size"]

    if density.shape != (grid_size,) * 3:
        raise ValueError(
            "Density dimensions do not match metadata."
        )

    grid_min = np.asarray(
        metadata["grid_minimum_mpc"],
        dtype=np.float64,
    )

    voxel_size = float(
        metadata["voxel_size_mpc"]
    )

    axes = [
        grid_min[i]
        + (
            np.arange(grid_size)
            + 0.5
        ) * voxel_size

        for i in range(3)
    ]

    print(f"Density shape: {density.shape}")
    print(f"Voxel size: {voxel_size:.3f} Mpc")

    return density, structure_map, axes, metadata


def calculate_structure_statistics(structure_map):

    labels = {
        0: "Void",
        1: "Sheet",
        2: "Filament",
        3: "Knot",
    }

    total = structure_map.size

    print("\nT-WEB CLASSIFICATION")
    print("-" * 40)

    statistics = {}

    for class_id, name in labels.items():

        count = int(
            np.count_nonzero(
                structure_map == class_id
            )
        )

        percentage = 100.0 * count / total

        statistics[name] = percentage

        print(
            f"{name:12s}: "
            f"{count:,} voxels "
            f"({percentage:.2f}%)"
        )

    invalid = np.count_nonzero(
        ~np.isin(structure_map, list(labels.keys()))
    )

    if invalid:
        raise ValueError(
            f"Found {invalid:,} invalid structure labels."
        )

    return statistics


def add_density_surface(fig, density, axes, settings):

    print("Preparing density visualization...")

    density = np.asarray(
        density,
        dtype=np.float32,
    )

    positive = density[density > 0]

    if positive.size == 0:
        raise ValueError("Density field contains no positive values.")

    low = np.percentile(
        positive,
        settings["density_percentile_min"],
    )

    high = np.percentile(
        positive,
        settings["density_percentile_max"],
    )

    # Downsample for Plotly.
    # This is only for visualization, not scientific processing.

    step = max(
        1,
        int(np.ceil(density.shape[0] / 96)),
    )

    density_display = density[
        ::step,
        ::step,
        ::step,
    ]

    x, y, z = np.meshgrid(
        axes[0][::step],
        axes[1][::step],
        axes[2][::step],
        indexing="ij",
    )

    fig.add_trace(
        go.Isosurface(

            x=x.ravel(),
            y=y.ravel(),
            z=z.ravel(),

            value=density_display.ravel(),

            isomin=float(low),
            isomax=float(high),

            surface_count=settings["density_surface_count"],

            colorscale=[
                [0.0, "#1746A2"],
                [0.4, "#00B4D8"],
                [0.75, "#F9C74F"],
                [1.0, "#F94144"],
            ],

            opacity=settings["density_opacity"],

            caps=dict(
                x_show=False,
                y_show=False,
                z_show=False,
            ),

            showscale=True,

            colorbar=dict(
                title="Density contrast",
                x=1.02,
            ),

            name="Density field",

            visible=True,

            hoverinfo="skip",
        )
    )


def add_tweb_visualization(fig, structure_map, axes, settings):

    print("Preparing T-web visualization...")

    colors = {
        0: "#10182D",
        1: "#2878B5",
        2: "#F4A340",
        3: "#F04444",
    }

    labels = {
        0: "Void",
        1: "Sheet",
        2: "Filament",
        3: "Knot",
    }

    # Downsample the classification map.
    step = max(
        1,
        int(np.ceil(structure_map.shape[0] / 96)),
    )

    structure_display = np.asarray(
        structure_map[::step, ::step, ::step]
    )

    x, y, z = np.meshgrid(
        axes[0][::step],
        axes[1][::step],
        axes[2][::step],
        indexing="ij",
    )

    flat_structure = structure_display.ravel()

    for class_id in [1, 2, 3]:

        mask = flat_structure == class_id

        if not np.any(mask):
            continue

        fig.add_trace(
            go.Scatter3d(

                x=x.ravel()[mask],
                y=y.ravel()[mask],
                z=z.ravel()[mask],

                mode="markers",

                marker=dict(
                    size=2,
                    color=colors[class_id],
                    opacity=0.35,
                ),

                name=labels[class_id],

                visible=True,

                hovertemplate=(
                    f"{labels[class_id]}"
                    "<extra></extra>"
                ),
            )
        )

    print("T-web visualization prepared.")


def add_galaxies(fig, df, settings):

    print("Sampling galaxies...")

    max_galaxies = settings["max_galaxies"]

    if len(df) > max_galaxies:

        galaxies = df.sample(
            n=max_galaxies,
            random_state=settings["random_seed"],
        )

    else:
        galaxies = df

    fig.add_trace(
        go.Scatter3d(

            x=galaxies["x_mpc"],
            y=galaxies["y_mpc"],
            z=galaxies["z_cart_mpc"],

            mode="markers",

            marker=dict(
                size=settings["galaxy_point_size"],
                opacity=settings["galaxy_opacity"],
                color=galaxies["Z"],
                colorscale="Plasma",
                showscale=True,

                colorbar=dict(
                    title="Redshift",
                    x=1.15,
                ),
            ),

            name="DESI galaxies",

            visible=True,

            hovertemplate=(
                "Redshift: %{marker.color:.3f}"
                "<extra></extra>"
            ),
        )
    )


def create_visualization(
    df,
    density,
    structure_map,
    axes,
    settings,
):

    print("\nCreating 3D cosmic web visualization...")

    fig = go.Figure()

    add_density_surface(
        fig,
        density,
        axes,
        settings,
    )

    add_tweb_visualization(
        fig,
        structure_map,
        axes,
        settings,
    )

    add_galaxies(
        fig,
        df,
        settings,
    )

    background = settings["background"]

    fig.update_layout(

        title=dict(
            text=(
                "DESI DR1 | 3D Cosmic Web"
                "<br><sup>Galaxy distribution, density and T-web classification</sup>"
            ),

            font=dict(
                size=22,
                color="white",
            ),
        ),

        paper_bgcolor=background,
        plot_bgcolor=background,

        font=dict(color="white"),

        scene=dict(

            bgcolor=background,

            xaxis=dict(
                title="X (Mpc)",
                backgroundcolor=background,
                gridcolor="#252B40",
            ),

            yaxis=dict(
                title="Y (Mpc)",
                backgroundcolor=background,
                gridcolor="#252B40",
            ),

            zaxis=dict(
                title="Z (Mpc)",
                backgroundcolor=background,
                gridcolor="#252B40",
            ),

            aspectmode="data",
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)",
        ),

        margin=dict(
            l=0,
            r=130,
            b=0,
            t=90,
        ),

    )

    return fig


def main():

    settings = CONFIG["cosmic_web_preview"]

    df = load_catalog()

    density, structure_map, axes, metadata = (
        load_structure_data()
    )

    statistics = calculate_structure_statistics(
        structure_map
    )

    fig = create_visualization(
        df,
        density,
        structure_map,
        axes,
        settings,
    )

    output = PROJECT_ROOT / settings["output_file"]

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.write_html(
        str(output),
        include_plotlyjs=True,
    )

    print("\n======================================")
    print("COSMIC WEB PREVIEW COMPLETE")
    print("======================================")

    print(f"Galaxies available: {len(df):,}")
    print(f"Density grid: {density.shape}")
    print(f"Voxel size: {metadata['voxel_size_mpc']:.3f} Mpc")

    print("\nOutput:")
    print(output)

    print("\nStructure percentages:")

    for name, percentage in statistics.items():
        print(f"{name}: {percentage:.2f}%")

    print("======================================\n")

    fig.show()


if __name__ == "__main__":
    main()

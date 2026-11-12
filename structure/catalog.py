import numpy as np
import pandas as pd
from structure.config import CONFIG, INPUT_FILE

def load_galaxies():

    print(f"Loading catalog: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    selection = CONFIG["selection"]

    min_z = selection["min_redshift"]
    max_z = selection["max_redshift"]

    valid = (
        np.isfinite(df["RA"])
        & np.isfinite(df["DEC"])
        & np.isfinite(df["Z"])
        & np.isfinite(df["x_mpc"])
        & np.isfinite(df["y_mpc"])
        & np.isfinite(df["z_cart_mpc"])
        & (df["Z"] > min_z)
        & (df["Z"] <= max_z)
    )

    df = df.loc[valid].copy()

    print(f"Retained galaxies: {len(df):,}")

    return df


def get_positions(df):

    return df[
        ["x_mpc", "y_mpc", "z_cart_mpc"]
    ].to_numpy(dtype=np.float32)


def get_vr_positions(df):

    return df[
        ["x_vr", "y_vr", "z_vr"]
    ].to_numpy(dtype=np.float32)

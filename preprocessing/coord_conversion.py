import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM


def add_3d_positions(catalog, H0=70.0, omega_m=0.3, scale_units_per_mpc=0.001):

    required = {"RA", "DEC", "Z"}
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    ra = catalog["RA"].to_numpy(dtype=np.float64)
    dec = catalog["DEC"].to_numpy(dtype=np.float64)
    redshift = catalog["Z"].to_numpy(dtype=np.float64)

    valid = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(redshift)
        & (ra >= 0.0)
        & (ra < 360.0)
        & (dec >= -90.0)
        & (dec <= 90.0)
        & (redshift >= 0.0)
    )

    if not np.all(valid):
        raise ValueError(
            f"Input contains {(~valid).sum()} invalid coordinate/redshift rows.")

    cosmology = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=omega_m)

    distance_mpc = cosmology.comoving_distance(redshift).to_value(u.Mpc)

    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    cos_dec = np.cos(dec_rad)

    x = distance_mpc * cos_dec * np.cos(ra_rad)
    y = distance_mpc * cos_dec * np.sin(ra_rad)
    z_cart = distance_mpc * np.sin(dec_rad)

    result = catalog.copy()

    result["comoving_distance_mpc"] = distance_mpc
    result["x_mpc"] = x
    result["y_mpc"] = y
    result["z_cart_mpc"] = z_cart

    result["x_vr"] = (x * scale_units_per_mpc).astype(np.float32)
    result["y_vr"] = (y * scale_units_per_mpc).astype(np.float32)
    result["z_vr"] = (z_cart * scale_units_per_mpc).astype(np.float32)

    return result

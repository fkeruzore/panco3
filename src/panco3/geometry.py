"""Static radial grids for the forward model.

Ports ``panco2.PressureProfileFitter.define_model``
(``panco2/panco2/panco2.py:399-436``). These grids depend only on the map
size, pixel size and angular diameter distance, so they are computed once
at setup in NumPy and held as constants. The model converts them to JAX
arrays where needed.
"""

from __future__ import annotations

import numpy as np
import astropy.units as u

_ARCSEC_TO_RAD = float(u.arcsec.to("rad"))


def make_radii(
    npix: int, pix_size: float, d_a: float
) -> dict[str, np.ndarray]:
    """Build the radii grids used by the forward model.

    Parameters
    ----------
    npix : int
        Map side length in pixels (square map).
    pix_size : float
        Pixel size [arcsec].
    d_a : float
        Angular diameter distance [kpc].

    Returns
    -------
    dict with keys
        ``theta_x`` : (npix//2 + 1,) 1-D on-sky angle [rad], half axis.
        ``r_x``     : (npix//2 + 1,) 1-D physical radius [kpc], half axis.
        ``r_xy``    : (npix, npix) 2-D physical radius from center [kpc].
    """
    theta_x = np.arange(0, int(npix / 2) + 1) * pix_size * _ARCSEC_TO_RAD
    r_x = d_a * np.tan(theta_x)  # kpc

    # Full symmetric axis: [-r_x[-1..1], r_x[0..]] then 2-D radius map.
    axis = np.concatenate((-np.flip(r_x[1:]), r_x))
    r_xy = np.hypot(*np.meshgrid(axis, axis))

    return {"theta_x": theta_x, "r_x": r_x, "r_xy": r_xy}

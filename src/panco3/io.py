"""Data I/O: load a tSZ map + RMS from FITS (NumPy/astropy).

Ports the data-loading logic of ``panco2.PressureProfileFitter.__init__``
(``panco2/panco2/panco2.py:64-130``). This is plain NumPy/astropy -- it runs
once at setup and is not in the gradient path. The fitter converts the returned
map/RMS to JAX arrays at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord


@dataclass
class SZData:
    sz_map: np.ndarray  # (npix, npix)
    sz_rms: np.ndarray  # (npix, npix)
    wcs: WCS
    pix_size: float  # arcsec
    map_size: float  # arcmin
    coords_center: SkyCoord


def load_sz_fits(
    sz_map_file: str,
    hdu_data: int,
    hdu_rms: int,
    coords_center: SkyCoord | None = None,
    map_size: float | None = None,
) -> SZData:
    """Load (and optionally crop) a tSZ map and its noise RMS from FITS."""
    with fits.open(sz_map_file) as hdulist:
        head = hdulist[hdu_data].header
        sz_map = hdulist[hdu_data].data
        sz_rms = hdulist[hdu_rms].data

        wcs = WCS(head)
        pix_size = np.abs(proj_plane_pixel_scales(wcs) * 3600)
        assert pix_size.size == 2, "Can't process the header, is it not 2d?"
        # panco2 used an exact ``==`` here, which fails on float-epsilon
        # differences from the projection; use a tolerant comparison instead.
        assert np.isclose(pix_size[0], pix_size[1], rtol=1e-6), (
            "Can't process map with different pixel sizes in RA and dec"
        )
        pix_size = float(pix_size[0])

        if coords_center is None:
            pix_center = np.array(sz_map.shape) // 2
            ra, dec = np.squeeze(wcs.all_pix2world(*pix_center, 0))
            coords_center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

        if map_size is not None:
            # Ensure an odd number of pixels after cropping.
            new_npix = int(map_size * 60 / pix_size)
            if new_npix % 2 == 0:
                map_size += pix_size / 60.0
                new_npix += 1
            cropped_map = Cutout2D(
                sz_map, coords_center, new_npix, wcs=wcs, mode="strict"
            )
            cropped_rms = Cutout2D(sz_rms, coords_center, new_npix, wcs=wcs)
            out_map, out_rms, out_wcs = (
                cropped_map.data,
                cropped_rms.data,
                cropped_map.wcs,
            )
        else:
            out_map, out_rms, out_wcs = sz_map, sz_rms, wcs
            map_size = sz_map.shape[0] * pix_size / 60

    sz_shape, rms_shape = out_map.shape, out_rms.shape
    assert sz_shape == rms_shape, (
        f"SZ map and RMS have incompatible shapes: {sz_shape, rms_shape}"
    )
    assert np.all(np.array(sz_shape) % 2 == 1), (
        f"SZ map has an even number of pixels: {sz_shape}"
    )

    return SZData(
        sz_map=np.asarray(out_map, dtype=float),
        sz_rms=np.asarray(out_rms, dtype=float),
        wcs=out_wcs,
        pix_size=pix_size,
        map_size=float(map_size),
        coords_center=coords_center,
    )

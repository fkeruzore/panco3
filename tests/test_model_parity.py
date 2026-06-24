"""Forward-model parity: panco3 (JAX) vs panco2 (NumPy) ModelBinned.

Compares the full pipeline -- Compton-y profile, 2-D projection, beam +
transfer-function filtering, conv/zero scaling, and point sources -- on
identical inputs. panco3's differentiable LOS quadrature should reproduce
panco2's analytic ``shell_pl`` (truncated at ``r_bins[-1]``) to ~1e-7.
"""

import numpy as np
import jax.numpy as jnp
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

import panco3
from panco3 import geometry, utils
from panco3.model import ModelBinned
from panco3.filtering import Filter1d
from conftest import load_panco2_package_module


NPIX = 101
PIX_SIZE = 3.0  # arcsec
Z, M500 = 0.5, 6e14


@pytest.fixture(scope="module")
def setup():
    cluster = panco3.Cluster(
        Z, M500, cosmo=panco3.cosmo_from_flat_lcdm(70.0, 0.3)
    )
    radii = geometry.make_radii(NPIX, PIX_SIZE, cluster.d_a)

    # Radial binning roughly like the C2_NIKA2 example.
    half_kpc = cluster.arcsec2kpc(NPIX * PIX_SIZE / 2)
    r_bins = np.concatenate(
        [
            [cluster.arcsec2kpc(PIX_SIZE)],
            np.logspace(np.log10(50.0), np.log10(1.1 * half_kpc), 4),
        ]
    )
    P_bins = np.asarray(utils.gNFW_from_params(r_bins, cluster.A10_params))
    return cluster, radii, r_bins, P_bins


def _par_vec(P_bins, conv=-12.0, zero=1e-4):
    return jnp.asarray(np.concatenate([P_bins, [conv, zero]]))


def test_compton_map_parity(setup):
    _, radii, r_bins, P_bins = setup
    p2model = load_panco2_package_module("model")

    m3 = ModelBinned(r_bins, radii, zero_level=True, n_nodes=64)
    m2 = p2model.ModelBinned(r_bins, radii, zero_level=True)

    par = _par_vec(P_bins)
    par_np = np.asarray(par)

    map3 = np.asarray(m3.compton_map(par))
    map2 = m2.compton_map(par_np)

    assert map3.shape == map2.shape == (NPIX, NPIX)
    # Relative agreement on the (nonzero) signal.
    mask = np.abs(map2) > np.abs(map2).max() * 1e-6
    rel = np.abs(map3[mask] / map2[mask] - 1.0)
    assert np.max(rel) < 1e-6


def test_sz_map_parity_with_filtering(setup):
    cluster, radii, r_bins, P_bins = setup
    p2model = load_panco2_package_module("model")

    # Identical beam + 1-D transfer function for both models.
    ell = np.linspace(0, 1e5, 500)
    tf = 1.0 - np.exp(-((ell / 2e4) ** 2))  # high-pass-ish mock TF
    beam_sigma_pix = (18.0 / (2 * np.sqrt(2 * np.log(2)))) / PIX_SIZE
    k_1d = (
        ell / (2 * np.pi) / 206265.0
    )  # ell -> 1/arcsec (rough, identical both)

    m3 = ModelBinned(r_bins, radii, zero_level=True, n_nodes=64)
    m2 = p2model.ModelBinned(r_bins, radii, zero_level=True)
    m3.filter = Filter1d(NPIX, PIX_SIZE, k_1d, tf, beam_sigma_pix, pad_pix=20)
    p2filt = load_panco2_package_module("filtering")
    m2.filter = p2filt.Filter1d(
        NPIX, PIX_SIZE, k_1d, tf, beam_sigma_pix, pad_pix=20
    )

    par = _par_vec(P_bins)
    sz3 = np.asarray(m3.sz_map(par))
    sz2 = m2.sz_map(np.asarray(par))

    assert sz3.shape == sz2.shape
    # Absolute agreement (map has a zero level); compare to signal scale.
    scale = np.abs(sz2 - sz2.mean()).max()
    assert np.max(np.abs(sz3 - sz2)) < 1e-6 * scale + 1e-12


def test_point_source_parity(setup):
    cluster, radii, r_bins, P_bins = setup
    p2model = load_panco2_package_module("model")

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [NPIX / 2, NPIX / 2]
    wcs.wcs.cdelt = [-PIX_SIZE / 3600.0, PIX_SIZE / 3600.0]
    wcs.wcs.crval = [180.0, 0.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.array_shape = (NPIX, NPIX)

    coords = [
        SkyCoord("12h00m02s +00d00m20s"),
        SkyCoord("11h59m58s -00d00m10s"),
    ]
    beam_sigma_pix = (18.0 / (2 * np.sqrt(2 * np.log(2)))) / PIX_SIZE

    m3 = ModelBinned(r_bins, radii, zero_level=True, n_nodes=64)
    m2 = p2model.ModelBinned(r_bins, radii, zero_level=True)
    m3.add_point_sources(coords, wcs, beam_sigma_pix)
    m2.add_point_sources(coords, wcs, beam_sigma_pix)

    par = jnp.asarray(np.concatenate([P_bins, [-12.0, 1e-4, 5.0, -3.0]]))
    ps3 = np.asarray(m3.ps_map(par))
    ps2 = m2.ps_map(np.asarray(par))
    assert np.allclose(ps3, ps2, rtol=1e-10, atol=1e-12)


def test_sz_map_differentiable(setup):
    import jax

    _, radii, r_bins, P_bins = setup
    m3 = ModelBinned(r_bins, radii, zero_level=True, n_nodes=32)

    def loss(logP, conv, zero):
        par = jnp.concatenate([jnp.exp(logP), jnp.array([conv, zero])])
        return jnp.sum(m3.sz_map(par) ** 2)

    g = jax.grad(loss, argnums=(0, 1, 2))(
        jnp.log(jnp.asarray(P_bins)), -12.0, 1e-4
    )
    assert all(np.all(np.isfinite(np.asarray(gi))) for gi in g)

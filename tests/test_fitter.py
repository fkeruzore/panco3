"""I/O + fitter smoke tests: load FITS, build model, evaluate, serialize."""

import os

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from astropy.io import fits

from panco3.fitter import PressureProfileFitter
from panco3 import utils

EXAMPLE_FITS = os.path.join(
    os.path.dirname(__file__),
    "..",
    "panco2",
    "examples",
    "C2_NIKA2",
    "C2_nk2.fits",
)


def _make_synthetic_fits(path, npix=65, pix_size=3.0):
    """Write a 2-HDU FITS (data + rms) with a valid TAN WCS, odd npix."""
    hdr = fits.Header()
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = npix // 2 + 1
    hdr["CRPIX2"] = npix // 2 + 1
    hdr["CRVAL1"] = 180.0
    hdr["CRVAL2"] = 0.0
    hdr["CDELT1"] = -pix_size / 3600.0
    hdr["CDELT2"] = pix_size / 3600.0
    rng = np.random.default_rng(0)
    data = rng.normal(size=(npix, npix)) * 1e-4
    rms = np.full((npix, npix), 1e-4)
    hdus = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=data, header=hdr, name="DATA"),
            fits.ImageHDU(data=rms, header=hdr, name="RMS"),
        ]
    )
    hdus.writeto(path, overwrite=True)


@pytest.fixture(scope="module")
def synth_fits(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "synth.fits")
    _make_synthetic_fits(path)
    return path


def test_load_and_model_map(synth_fits):
    ppf = PressureProfileFitter(synth_fits, 1, 2, z=0.5, M_500=6e14)
    assert ppf.sz_map.shape == ppf.sz_rms.shape
    assert ppf.sz_map.shape[0] % 2 == 1

    r_bins = np.logspace(np.log10(50.0), np.log10(2000.0), 5)
    ppf.define_model(r_bins, n_nodes=32)
    P_bins = np.asarray(utils.gNFW_from_params(r_bins, ppf.cluster.A10_params))
    par = jnp.asarray(np.concatenate([P_bins, [-12.0, 1e-4]]))

    m = np.asarray(ppf.model_map(par))
    assert m.shape == ppf.sz_map.shape
    assert np.all(np.isfinite(m))


def test_filtering_and_grad(synth_fits):
    ppf = PressureProfileFitter(synth_fits, 1, 2, z=0.5, M_500=6e14)
    r_bins = np.logspace(np.log10(50.0), np.log10(2000.0), 5)
    ppf.define_model(r_bins, n_nodes=32)

    ell = np.linspace(0, 1e5, 400)
    tf = 1.0 - np.exp(-((ell / 2e4) ** 2))
    ppf.add_filtering(beam_fwhm=18.0, ell=ell, tf=tf, pad=10)

    P_bins = np.asarray(utils.gNFW_from_params(r_bins, ppf.cluster.A10_params))

    def loss(logP):
        par = jnp.concatenate([jnp.exp(logP), jnp.array([-12.0, 1e-4])])
        m = ppf.model_map(par)
        return jnp.sum(((ppf.sz_map - m) / ppf.sz_rms) ** 2)

    g = jax.grad(loss)(jnp.log(jnp.asarray(P_bins)))
    assert np.all(np.isfinite(np.asarray(g)))


def test_dump_load_roundtrip(synth_fits, tmp_path):
    ppf = PressureProfileFitter(synth_fits, 1, 2, z=0.5, M_500=6e14)
    r_bins = np.logspace(np.log10(50.0), np.log10(2000.0), 5)
    ppf.define_model(r_bins)
    ppf.define_priors(conv=(-12.0, 1.0), zero=(0.0, 1e-4))

    f = str(tmp_path / "ppf.dill")
    ppf.dump_to_file(f)
    ppf2 = PressureProfileFitter.load_from_file(f)
    assert ppf2.priors["conv"] == (-12.0, 1.0)
    assert np.allclose(np.asarray(ppf2.sz_map), np.asarray(ppf.sz_map))


@pytest.mark.skipif(
    not os.path.exists(EXAMPLE_FITS), reason="example FITS absent"
)
def test_load_example_fits():
    ppf = PressureProfileFitter(
        EXAMPLE_FITS, 1, 5, z=0.5, M_500=6e14, map_size=4.0
    )
    assert ppf.sz_map.shape[0] % 2 == 1
    r_bins = np.logspace(np.log10(50.0), np.log10(1500.0), 5)
    ppf.define_model(r_bins)
    P_bins = np.asarray(utils.gNFW_from_params(r_bins, ppf.cluster.A10_params))
    par = jnp.asarray(np.concatenate([P_bins, [-12.0, 0.0]]))
    assert np.all(np.isfinite(np.asarray(ppf.model_map(par))))

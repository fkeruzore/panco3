"""Inference tests: differentiable log-posterior + BlackJAX NUTS recovery.

Simulates a tSZ map from a known pressure profile, then checks that NUTS
recovers the truth (the end-to-end regression for the whole differentiable
stack).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from astropy.io import fits

import panco3
from panco3.fitter import PressureProfileFitter
from panco3 import utils, priors, posterior, inference


def _make_fits(path, npix=33, pix=3.0):
    hdr = fits.Header()
    hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"
    hdr["CRPIX1"] = hdr["CRPIX2"] = npix // 2 + 1
    hdr["CRVAL1"], hdr["CRVAL2"] = 180.0, 0.0
    hdr["CDELT1"], hdr["CDELT2"] = -pix / 3600, pix / 3600
    z = np.zeros((npix, npix))
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(z, hdr, name="DATA"),
            fits.ImageHDU(z + 1, hdr, name="RMS"),
        ]
    ).writeto(path, overwrite=True)


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    """A fitter whose data is a known model map + Gaussian noise."""
    path = str(tmp_path_factory.mktemp("sim") / "sim.fits")
    _make_fits(path)
    ppf = PressureProfileFitter(
        path,
        1,
        2,
        z=0.5,
        M_500=6e14,
        cosmo=panco3.cosmo_from_flat_lcdm(70, 0.3),
    )
    r_bins = np.logspace(np.log10(60.0), np.log10(1200.0), 4)
    ppf.define_model(r_bins, n_nodes=16)
    ppf.add_filtering(beam_fwhm=18.0)

    P_true = np.asarray(utils.gNFW_from_params(r_bins, ppf.cluster.A10_params))
    par_true = jnp.asarray(np.concatenate([P_true, [1.0, 0.0]]))
    m_true = np.asarray(ppf.model.model_map(par_true))

    rms = 0.02 * np.abs(m_true).max()
    rng = np.random.default_rng(1)
    ppf.sz_map = jnp.asarray(m_true + rng.normal(size=m_true.shape) * rms)
    ppf.sz_rms = jnp.asarray(np.full(m_true.shape, rms))

    ppf.define_priors(
        P_bins=[priors.LogNormal(np.log(P), 2.0) for P in P_true],
        conv=priors.Normal(1.0, 0.5),
        zero=priors.Normal(0.0, 0.5 * rms),
    )
    return ppf, P_true


def test_log_posterior_grad_and_jit(simulated):
    ppf, _ = simulated
    logpost, plist, init_z, constrain = posterior.make_log_posterior(ppf)
    assert np.isfinite(float(logpost(init_z)))
    g = jax.grad(logpost)(init_z)
    assert np.all(np.isfinite(np.asarray(g)))
    # jit compiles and matches eager.
    jlp = jax.jit(logpost)
    assert np.isclose(float(jlp(init_z)), float(logpost(init_z)), rtol=1e-10)


def test_priors_pushforward():
    """LogUniform pushforward is flat in log(theta); LogNormal is unbounded."""
    lu = priors.LogUniform(1e-3, 1e1)
    zs = jnp.linspace(-5, 5, 11)
    thetas = jax.vmap(lu.constrain)(zs)
    assert np.all(np.asarray(thetas) > 1e-3 - 1e-9)
    assert np.all(np.asarray(thetas) < 1e1 + 1e-6)
    # log_prob finite and differentiable everywhere (no hard walls).
    lp = jax.vmap(jax.grad(lu.log_prob))(zs)
    assert np.all(np.isfinite(np.asarray(lp)))


@pytest.mark.slow
def test_nuts_recovers_profile(simulated):
    ppf, P_true = simulated
    logpost, plist, init_z, constrain = posterior.make_log_posterior(ppf)

    # target_acceptance_rate=0.9: the conv * amplitude degeneracy here (broad
    # conv prior) is a mild funnel; smaller steps keep divergences low.
    res = inference.run_nuts(
        logpost,
        init_z,
        num_warmup=600,
        num_samples=600,
        num_chains=2,
        target_acceptance_rate=0.9,
        rng_key=jax.random.PRNGKey(0),
    )
    n_draws = res["samples"].shape[0] * res["samples"].shape[1]
    assert res["acceptance_rate"].mean() > 0.5
    # A few divergences are expected at the funnel neck; require << 5%.
    assert res["divergences"].sum() < 0.05 * n_draws

    cs = inference.constrained_samples(
        res, constrain
    )  # (chains, draws, nparams)
    flat = cs.reshape(-1, cs.shape[-1])

    # Bayesian recovery: the truth must fall inside the 95% credible interval
    # for every parameter. (A median-point tolerance would be fragile for the
    # weakly-constrained outer bin, whose posterior is genuinely broad -- the
    # whitened sampler now explores it instead of staying pinned near init.)
    truth = np.concatenate([P_true, [1.0, 0.0]])
    lo, hi = np.percentile(flat, [2.5, 97.5], axis=0)
    for i, name in enumerate(ppf.model.params):
        assert lo[i] <= truth[i] <= hi[i], (
            f"{name}: truth {truth[i]:.3e} outside 95% CI "
            f"[{lo[i]:.3e}, {hi[i]:.3e}]"
        )

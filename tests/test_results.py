"""Post-processing helpers: profile CIs, named dicts, and plotting smoke
tests."""

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import pytest

import panco3
from panco3 import geometry, utils, results
from panco3.model import ModelBinned

NPIX, PIX = 41, 3.0


@pytest.fixture(scope="module")
def model_and_samples():
    cluster = panco3.Cluster(
        0.5, 6e14, cosmo=panco3.cosmo_from_flat_lcdm(70, 0.3)
    )
    radii = geometry.make_radii(NPIX, PIX, cluster.d_a)
    r_bins = np.logspace(np.log10(60.0), np.log10(1200.0), 4)
    m = ModelBinned(r_bins, radii, zero_level=True, n_nodes=16)

    P_true = np.asarray(utils.gNFW_from_params(r_bins, cluster.A10_params))
    # Fake posterior: truth + small log-normal scatter; (chains, draws,
    # n_params).
    rng = np.random.default_rng(0)
    n_chain, n_draw = 2, 200
    P = P_true[None, None, :] * np.exp(
        rng.normal(0, 0.05, (n_chain, n_draw, 4))
    )
    conv = rng.normal(1.0, 0.02, (n_chain, n_draw, 1))
    zero = rng.normal(0.0, 1e-5, (n_chain, n_draw, 1))
    samples = np.concatenate([P, conv, zero], axis=-1)
    return m, samples, P_true


def test_constrained_to_dict(model_and_samples):
    m, samples, _ = model_and_samples
    d = results.constrained_to_dict(samples, m)
    assert set(d.keys()) == set(m.params)
    assert d["P_0"].shape == samples.shape[:2]


def test_median_par_vec(model_and_samples):
    m, samples, P_true = model_and_samples
    med = results.median_par_vec(samples)
    assert med.shape == (m.n_params,)
    assert np.allclose(med[: m.n_bins], P_true, rtol=0.05)


def test_pressure_profile_ci(model_and_samples):
    m, samples, P_true = model_and_samples
    r = np.logspace(np.log10(60), np.log10(1200), 30)
    lo, mid, hi = results.pressure_profile_ci(m, samples, r)
    assert lo.shape == mid.shape == hi.shape == (30,)
    assert np.all(lo <= mid) and np.all(mid <= hi)
    assert np.all(mid > 0)


def test_plots_run(model_and_samples, tmp_path):
    import matplotlib.pyplot as plt

    m, samples, P_true = model_and_samples
    r = np.logspace(np.log10(60), np.log10(1200), 30)
    ax = results.plot_pressure_profile(m, samples, r, truth=P_true)
    ax.figure.savefig(tmp_path / "prof.png")
    plt.close("all")


def test_to_inference_data(model_and_samples):
    m, samples, _ = model_and_samples
    idata = results.to_inference_data(samples, m)
    assert set(idata.posterior.data_vars) == set(m.params)
    # Constrained (physical) space: pressures are positive, not log values.
    assert float(idata.posterior["P_0"].mean()) > 0


def test_trace_and_corner_run(model_and_samples, tmp_path):
    import matplotlib.pyplot as plt
    from panco3 import priors

    m, samples, P_true = model_and_samples
    truth = {f"P_{i}": P for i, P in enumerate(P_true)}
    truth["conv"] = 1.0
    truth["zero"] = 0.0

    tr = results.plot_trace(samples, m, truth=truth)
    tr.ravel()[0].figure.savefig(tmp_path / "trace.png")

    # Corner: lower=contours, upper=cloud, diagonal truth line + prior. Use an
    # ordered prior list (as returned by make_log_posterior) and truth dict.
    plist = [priors.LogNormal(np.log(P), 0.4) for P in P_true] + [
        priors.Normal(1.0, 0.02),
        priors.Normal(0.0, 1e-5),
    ]
    co = results.plot_corner(samples, m, truth=truth, priors=plist)
    assert co.shape == (m.n_params, m.n_params)
    co.ravel()[0].figure.savefig(tmp_path / "corner.png")

    # truth as a full par_vec array should also work (no priors).
    co2 = results.plot_corner(
        samples, m, truth=results.median_par_vec(samples)
    )
    co2.ravel()[0].figure.savefig(tmp_path / "corner2.png")
    plt.close("all")


def test_prior_pdf_normalization():
    from panco3 import priors
    from scipy.integrate import quad

    # Each prior's natural-space pdf should integrate to ~1.
    n = priors.Normal(0.5, 0.2)
    assert abs(quad(n.pdf, -5, 6)[0] - 1.0) < 1e-6

    ln = priors.LogNormal(np.log(1e-2), 0.5)
    assert abs(quad(ln.pdf, 0, 1.0)[0] - 1.0) < 1e-4

    lu = priors.LogUniform(1e-3, 1e-1)
    assert abs(quad(lu.pdf, 0, 0.2)[0] - 1.0) < 1e-6

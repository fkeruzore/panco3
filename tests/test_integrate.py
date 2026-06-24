"""Validate the differentiable LOS quadrature against the analytic oracle.

* ``interp.interp_powerlaw`` / ``interp.prof2map`` vs panco2's ``utils``.
* ``integrate.compton_y_profile`` vs the analytic per-shell Abel transform
  (``abell_ref``), reproducing panco2's truncation at ``r_bins[-1]``.
* Gauss-Legendre node-count (Richardson) convergence.
* ``jax.grad`` of the integral vs central finite differences (the regression
  guard that the analytic/betainc path would have failed).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.integrate import quad

import panco3  # noqa: F401  (enables x64)
from panco3 import abell_ref as ar
from panco3 import interp, integrate
from conftest import load_panco2_module

SZ_FACT = panco3.SZ_FACT


# --------------------------------------------------------------------------- #
# Helpers: analytic oracle reproducing panco2.model.compton_prof
# --------------------------------------------------------------------------- #
def compute_slopes(P_i, r_bins):
    lr, lp = np.log(r_bins), np.log(P_i)
    a = -np.ediff1d(lp) / np.ediff1d(lr)
    return np.concatenate(([a[0]], a, [a[-1]]))


def analytic_y(P_i, r_bins, R, sz_fact):
    """panco2's truncated compton_prof: shells [0,r0]..[r_{n-2},r_{n-1}]."""
    alphas = compute_slopes(P_i, r_bins)
    r_edges = np.concatenate(([0.0], r_bins))
    total = np.zeros_like(R, dtype=float)
    for i in range(len(P_i)):
        total += ar.projected_power_law(
            P_i[i], alphas[i], r_edges[i], r_edges[i + 1], R
        )
    return total * sz_fact


def _P_loglog(r, r_bins, P_i):
    """NumPy log-log interp/extrap of the bins (matches
    interp.interp_powerlaw)."""
    lr, lp = np.log(r_bins), np.log(P_i)
    x = np.log(r)
    i = np.clip(np.searchsorted(lr, x, side="right") - 1, 0, len(lr) - 2)
    sl = (lp[i + 1] - lp[i]) / (lr[i + 1] - lr[i])
    return np.exp(lp[i] + sl * (x - lr[i]))


def brute_y(P_i, r_bins, R, sz_fact):
    """Ground-truth LOS integral via scipy.quad, truncated at r_bins[-1].

    Well-defined for *all* slopes (including alpha=1, where the analytic Abel
    form hits a removable Gamma pole and is unusable).
    """
    r_max = r_bins[-1]
    out = np.zeros_like(R, dtype=float)
    for k, RR in enumerate(R):
        if RR >= r_max:
            continue
        ll_max = np.sqrt(r_max**2 - RR**2)
        val, _ = quad(
            lambda ll, RR_val=RR: _P_loglog(
                np.sqrt(RR_val**2 + ll**2), r_bins, P_i
            ),
            0.0,
            ll_max,
            epsabs=1e-15,
            epsrel=1e-12,
            limit=400,
        )
        out[k] = 2.0 * val * sz_fact
    return out


def gnfw_bins(n=5, rmin=20.0, rmax=2000.0):
    r_bins = np.logspace(np.log10(rmin), np.log10(rmax), n)
    P_i = 1e-2 * (r_bins / r_bins[0]) ** (-1.2) / (1 + (r_bins / 700) ** 3)
    return r_bins, P_i


# --------------------------------------------------------------------------- #
# Interpolation parity
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def p2_utils():
    return load_panco2_module("utils")


def test_interp_powerlaw_matches_panco2(p2_utils):
    r_bins, P_i = gnfw_bins()
    # Interior, plus extrapolation below r_bins[0] and above r_bins[-1].
    r_new = np.logspace(np.log10(2.0), np.log10(6000.0), 50)
    ref = p2_utils.interp_powerlaw(r_bins, P_i, r_new)
    got = np.asarray(
        interp.interp_powerlaw(
            jnp.asarray(r_bins), jnp.asarray(P_i), jnp.asarray(r_new)
        )
    )
    assert np.allclose(got, ref, rtol=1e-12, atol=0)


def test_prof2map_matches_panco2(p2_utils):
    r_x = np.logspace(np.log10(5.0), np.log10(2000.0), 40)
    prof = 1e-3 * (r_x / 100.0) ** (-1.5)
    r_xy = np.linspace(1.0, 3000.0, 200)
    ref = p2_utils.prof2map(prof, r_x, r_xy)
    got = np.asarray(
        interp.prof2map(jnp.asarray(prof), jnp.asarray(r_x), jnp.asarray(r_xy))
    )
    assert np.allclose(got, ref, rtol=1e-12, atol=0)


# --------------------------------------------------------------------------- #
# Quadrature vs analytic oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "alpha", [0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
)
def test_single_powerlaw_vs_bruteforce(alpha):
    """Two-bin (single power-law) profile, slope swept incl. integers.

    Compared against a brute-force quad (ground truth), which - unlike the
    analytic Abel oracle - is well-defined at alpha=1 (the Gamma pole).
    """
    r_bins = np.array([50.0, 1500.0])
    P0 = 1e-2
    P_i = np.array([P0, P0 * (r_bins[1] / r_bins[0]) ** (-alpha)])
    R = np.logspace(np.log10(5.0), np.log10(1400.0), 16)

    ref = brute_y(P_i, r_bins, R, SZ_FACT)
    got = np.asarray(
        integrate.compton_y_profile(
            jnp.asarray(P_i),
            jnp.asarray(r_bins),
            jnp.asarray(R),
            SZ_FACT,
            n_nodes=24,
        )
    )
    assert np.max(np.abs(got / ref - 1.0)) < 1e-8


def test_multibin_vs_analytic():
    r_bins, P_i = gnfw_bins()
    R = np.logspace(np.log10(5.0), np.log10(1900.0), 24)
    ref = analytic_y(P_i, r_bins, R, SZ_FACT)
    got = np.asarray(
        integrate.compton_y_profile(
            jnp.asarray(P_i),
            jnp.asarray(r_bins),
            jnp.asarray(R),
            SZ_FACT,
            n_nodes=24,
        )
    )
    # Segmented quadrature: each segment is a single power law -> spectral GL.
    rel = np.abs(got / ref - 1.0)
    assert np.max(rel) < 1e-9


def test_segmentation_accurate_at_low_node_count():
    """The point of segmentation: high accuracy with few nodes per segment."""
    r_bins, P_i = gnfw_bins()
    R = np.logspace(np.log10(5.0), np.log10(1900.0), 24)
    ref = analytic_y(P_i, r_bins, R, SZ_FACT)

    def err(n):
        got = np.asarray(
            integrate.compton_y_profile(
                jnp.asarray(P_i),
                jnp.asarray(r_bins),
                jnp.asarray(R),
                SZ_FACT,
                n_nodes=n,
            )
        )
        return np.max(np.abs(got / ref - 1.0))

    # Even 8 nodes/segment is already excellent; 24 reaches ~machine precision.
    assert err(8) < 1e-6
    assert err(24) < 1e-10


def test_outer_truncation_zero_beyond_rmax():
    r_bins, P_i = gnfw_bins()
    R = np.array([r_bins[-1] * 1.01, r_bins[-1] * 2.0])
    got = np.asarray(
        integrate.compton_y_profile(
            jnp.asarray(P_i), jnp.asarray(r_bins), jnp.asarray(R), SZ_FACT
        )
    )
    assert np.allclose(got, 0.0)


# --------------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------------- #
def test_gradient_finite_difference():
    r_bins, P_i = gnfw_bins()
    R = np.logspace(np.log10(10.0), np.log10(1500.0), 8)
    r_bins_j, R_j = jnp.asarray(r_bins), jnp.asarray(R)

    def scalar_out(logP):
        P = jnp.exp(logP)
        y = integrate.compton_y_profile(P, r_bins_j, R_j, SZ_FACT, n_nodes=128)
        return jnp.sum(y)

    logP0 = jnp.log(jnp.asarray(P_i))
    grad = np.asarray(jax.grad(scalar_out)(logP0))
    assert np.all(np.isfinite(grad))

    # Central finite differences w.r.t. each log-pressure.
    eps = 1e-6
    fd = np.zeros_like(grad)
    for k in range(len(P_i)):
        dp = jnp.zeros_like(logP0).at[k].set(eps)
        fp = float(scalar_out(logP0 + dp))
        fm = float(scalar_out(logP0 - dp))
        fd[k] = (fp - fm) / (2 * eps)
    assert np.allclose(grad, fd, rtol=1e-5, atol=1e-12)


def test_jit_and_grad_compile():
    r_bins, P_i = gnfw_bins()
    R = np.logspace(np.log10(10.0), np.log10(1500.0), 8)
    f = jax.jit(
        lambda P: integrate.compton_y_profile(
            P, jnp.asarray(r_bins), jnp.asarray(R), SZ_FACT
        )
    )
    out = f(jnp.asarray(P_i))
    assert out.shape == R.shape
    assert np.all(np.isfinite(np.asarray(out)))

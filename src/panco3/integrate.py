"""Differentiable line-of-sight (Compton-y) integral by numerical quadrature.

This replaces panco2's analytic per-shell Abel transform
(``shell_pl``/``abell``). The analytic form reduces to the incomplete Beta
function, whose JAX implementation has **no gradient w.r.t. its shape
parameters** -- and those parameters are set by the pressure-profile
slope, which depends on the sampled bins. So the analytic path is not
autodiff-able for HMC. Instead we evaluate the physically primitive
integral

    y(R) = 2 * sz_fact * \\int_0^{l_max} P(sqrt(R^2 + l^2)) dl,

where ``P(r)`` is the (already differentiable) log-log interpolation of the
pressure bins. With the substitution ``l = R sinh(t)`` (so ``r = R cosh(t)``,
``dl = R cosh(t) dt``) the integrand's projection singularity is removed.

**Segmented quadrature.** ``P(r)`` is only C0 (kinks at the bin radii),
which would limit plain Gauss-Legendre to algebraic convergence. We instead
split each line of sight at the radii where ``r = R cosh(t)`` crosses a
bin edge, so every segment lies within a single power-law bin where the
integrand is smooth and GL converges spectrally. The split points depend
only on the (fixed) ``r_bins`` and ``R``, not on the sampled ``P_i``, so
differentiability is preserved.

**Outer truncation.** panco2's ``model.compton_prof`` sums shells only
out to ``r_bins[-1]`` (the outer tail beyond the last bin is dropped;
verified to ~1e-9 against a brute-force integral). We reproduce that by
default: ``P`` is treated as zero beyond ``r_max_integ`` (default
``r_bins[-1]``). This strongly affects
the predicted signal near the outer radius. Pass a larger ``r_max_integ`` to
include more of the tail.
"""

from __future__ import annotations

import functools

import numpy as np
import jax.numpy as jnp
from jax import Array

from .interp import interp_powerlaw


@functools.lru_cache(maxsize=8)
def gauss_legendre(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights on ``[-1, 1]`` as float64 numpy arrays.

    Cached: the nodes are static constants reused across every evaluation.
    We return *numpy* (not jax) arrays on purpose: a cached jax array created
    inside the first ``jit`` trace would leak that trace's context and raise
    ``UnexpectedTracerError`` on a later, independent ``jit``. NumPy arrays are
    instead folded in as concrete constants in each trace.
    """
    x, w = np.polynomial.legendre.leggauss(int(n_nodes))
    return x.astype(np.float64), w.astype(np.float64)


def compton_y_profile(
    P_i: Array,
    r_bins: Array,
    R: Array,
    sz_fact: float,
    *,
    r_max_integ: float | None = None,
    n_nodes: int = 32,
) -> Array:
    """Compton-y profile ``y(R)`` from binned pressure, by segmented LOS
    quadrature.

    Parameters
    ----------
    P_i : (n_bins,) array
        Pressure values at ``r_bins`` [keV cm-3].
    r_bins : (n_bins,) array
        Radial bin edges [kpc], strictly increasing.
    R : (n_R,) array
        Projected radii at which to evaluate ``y`` [kpc], strictly positive.
    sz_fact : float
        ``sigma_T / (m_e c^2)`` in [cm3 keV-1 kpc-1] (see ``cluster.SZ_FACT``).
    r_max_integ : float, optional
        Outer 3-D radius to integrate to [kpc]. Defaults to ``r_bins[-1]`` to
        match panco2 (pressure treated as zero beyond it).
    n_nodes : int
        Gauss-Legendre nodes **per segment** (one segment per bin).

    Returns
    -------
    (n_R,) array
        The Compton-y profile, ``0`` wherever ``R >= r_max_integ``.
    """
    R = jnp.asarray(R)
    r_bins = jnp.asarray(r_bins)
    if r_max_integ is None:
        r_max_integ = r_bins[-1]
    r_max_integ = jnp.asarray(r_max_integ)

    gl_x, gl_w = gauss_legendre(n_nodes)

    # Segment boundaries in t for each projected radius. A line of sight at R
    # crosses bin edge r_bins[k] at t = arccosh(r_bins[k] / R). Clipping the
    # ratio to >= 1 collapses crossings at r_bins[k] <= R to t = 0 (zero-width
    # leading segments contributing nothing), and yields t_max at the outer
    # edge r_bins[-1] = r_max_integ. Prepending 0 gives n_bins segments, each
    # spanning a single power-law bin.
    tc = jnp.arccosh(
        jnp.maximum(r_bins[None, :] / R[:, None], 1.0)
    )  # (n_R, n_bins)
    bounds = jnp.concatenate(
        [jnp.zeros((R.shape[0], 1)), tc], axis=1
    )  # (n_R, n_bins+1)
    t_lo = bounds[:, :-1]  # (n_R, n_bins)
    t_hi = bounds[:, 1:]  # (n_R, n_bins)

    # GL nodes mapped into every (R, segment): shape (n_R, n_bins, n_nodes).
    half = 0.5 * (t_hi - t_lo)
    mid = 0.5 * (t_hi + t_lo)
    t = mid[..., None] + half[..., None] * gl_x[None, None, :]
    w = half[..., None] * gl_w[None, None, :]

    cosh_t = jnp.cosh(t)
    r3d = R[:, None, None] * cosh_t  # in [R, r_max_integ], within one bin
    P_vals = interp_powerlaw(r_bins, P_i, r3d)
    integrand = P_vals * R[:, None, None] * cosh_t  # dl/dt Jacobian included

    y = 2.0 * sz_fact * jnp.sum(integrand * w, axis=(1, 2))
    return y

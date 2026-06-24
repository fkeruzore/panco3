"""Differentiable 1-D interpolation with linear extrapolation.

panco2 uses ``scipy.interpolate.interp1d(..., fill_value="extrapolate")``
in two distinct coordinate systems, both reproduced here in JAX:

* ``interp_powerlaw`` -- **log-log** linear interp (logs *both* axes),
  used to evaluate the piecewise-power-law pressure profile ``P(r)`` from
  the bins. Mirrors ``panco2.utils.interp_powerlaw``
  (``panco2/panco2/utils.py:123``).
* ``prof2map`` -- **linear-x / log-y** interp, used to project a 1-D
  profile onto a 2-D radius map. Mirrors ``panco2.utils.prof2map``
  (``panco2/panco2/utils.py:252``).

``jnp.interp`` clamps at the data range; panco2 extrapolates using the slope of
the nearest segment, so we implement the bracketing + linear formula by hand.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _interp_extrap(x: Array, xp: Array, fp: Array) -> Array:
    """Linear interpolation of ``(xp, fp)`` at ``x`` with edge-slope
    extrapolation.

    ``xp`` must be sorted strictly increasing with at least two points.
    For ``x`` outside ``[xp[0], xp[-1]]`` the nearest segment's slope is
    extended (matching ``interp1d(fill_value="extrapolate")``).
    Differentiable w.r.t. ``fp``.
    """
    n = xp.shape[0]
    # Bracketing segment [i, i+1]; clip so out-of-range points reuse the edge
    # segment, whose linear formula then extrapolates.
    i = jnp.clip(jnp.searchsorted(xp, x, side="right") - 1, 0, n - 2)
    x0 = xp[i]
    x1 = xp[i + 1]
    f0 = fp[i]
    f1 = fp[i + 1]
    return f0 + (f1 - f0) * (x - x0) / (x1 - x0)


def interp_powerlaw(x: Array, y: Array, x_new: Array) -> Array:
    """Power-law (log-log linear) interpolation/extrapolation.

    Equivalent to ``panco2.utils.interp_powerlaw``: linear interpolation
    of ``(log x, log y)``, i.e. a piecewise power law, extended beyond the
    grid with the edge segments' slopes. ``x``, ``y``, ``x_new`` must be
    strictly positive.
    """
    log_y_new = _interp_extrap(jnp.log(x_new), jnp.log(x), jnp.log(y))
    return jnp.exp(log_y_new)


def prof2map(prof_x: Array, r_x: Array, r_xy: Array) -> Array:
    """Project a 1-D profile onto a 2-D radius map.

    Equivalent to ``panco2.utils.prof2map``: linear-in-``r`` interpolation of
    ``log10(prof_x)`` evaluated at the 2-D radii ``r_xy`` (note: ``r`` is *not*
    logged here, unlike ``interp_powerlaw``).
    """
    log_prof = _interp_extrap(r_xy, r_x, jnp.log10(prof_x))
    return 10.0**log_prof

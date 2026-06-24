"""NumPy reference oracle for the analytic line-of-sight (Abel) integral.

Verbatim copy of ``panco2/panco2/abell.py`` (the cleaned-up vectorised
version of panco2's ``shell_pl``). It is used **only** to validate panco3's
differentiable numerical quadrature (``panco3.integrate``); it is never
imported by the forward model. See ``tests/test_integrate.py``.
"""

import numpy as np
import scipy.special as sps

__all__ = [
    "plsphere",
    "plshell",
    "plsphole",
    "plinfty",
    "projected_power_law",
]

# ------------------------------------------------------------------- #
# Helper functions                                                    #
# ------------------------------------------------------------------- #


def _beta_prefactor(p: float) -> float:
    """
    Return :math:`\\sqrt{\\pi}\\,\\Gamma(p-1/2)/\\Gamma(p)`.
    """
    return np.sqrt(np.pi) * sps.gamma(p - 0.5) / sps.gamma(p)


def _rincbeta(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Regularised incomplete Beta, extended to negative ``a`` via recursion.

    NOTE: panco2's ``abell.py`` (the unused refactor) had a SIGN ERROR here,
    using ``- x^a (1-x)^b / (a B(a,b))``. The correct recurrence (DLMF 8.17.20,
    and panco2's *production* ``shell_pl.myrincbeta``) is ``+``:
        I_x(a, b) = I_x(a+1, b) + x^a (1-x)^b / (a B(a, b)).
    Without this fix the oracle is wrong for slopes alpha < 1 (negative a),
    which occur for the shallow inner regions of real cluster profiles. Fixed
    here so abell_ref is a valid reference oracle across the full slope range.
    """
    x = np.clip(x, 0.0, 1.0)
    if a >= 0:
        return sps.betainc(a, b, x)

    beta_ab = sps.beta(a, b)
    return sps.betainc(a + 1.0, b, x) + x**a * (1.0 - x) ** b / (a * beta_ab)


# ------------------------------------------------------------------- #
# Core Abell transforms (all vectorised)                              #
# ------------------------------------------------------------------- #


def plsphere(p: float, r_max: float, R: np.ndarray) -> np.ndarray:
    """
    Line-of-sight integral for a *full* sphere (0 ≤ r ≤ r_max).
    """
    R = np.asanyarray(R, dtype=float)
    sigma = np.zeros_like(R)

    inside = R <= r_max
    if not np.any(inside):
        return sigma

    Ri = R[inside]
    x = (Ri / r_max) ** 2
    pref = _beta_prefactor(p) * (r_max ** (2.0 * p))
    sigma[inside] = (
        pref * (Ri ** (1.0 - 2.0 * p)) * (1.0 - _rincbeta(x, p - 0.5, 0.5))
    )
    return sigma


def plshell(p: float, r_min: float, r_max: float, R: np.ndarray) -> np.ndarray:
    """
    Line-of-sight integral for a *shell* r_min ≤ r ≤ r_max.
    """
    R = np.asanyarray(R, dtype=float)
    sigma = np.zeros_like(R)

    inside = R <= r_max
    if not np.any(inside):
        return sigma

    Ri = R[inside]
    sir = Ri / r_min
    pl_pow = sir ** (1.0 - 2.0 * p)
    cbf = _beta_prefactor(p)
    pref = r_min * cbf

    # Region r_min ≤ R ≤ r_max
    sor2 = (Ri / r_max) ** 2
    sigma[inside] = pref * pl_pow * (1.0 - _rincbeta(sor2, p - 0.5, 0.5))

    # Region R < r_min  (needs extra subtraction)
    inner_abs_mask = inside & (R < r_min)
    if np.any(inner_abs_mask):
        R_inner = R[inner_abs_mask]
        ibir = _rincbeta((R_inner / r_min) ** 2, p - 0.5, 0.5)
        ibor = _rincbeta((R_inner / r_max) ** 2, p - 0.5, 0.5)
        pl_pow_in = (R_inner / r_min) ** (1.0 - 2.0 * p)
        sigma[inner_abs_mask] = pref * pl_pow_in * (ibir - ibor)

    return sigma


def plsphole(p: float, r_min: float, R: np.ndarray) -> np.ndarray:
    """
    Line-of-sight integral for a profile extending to ∞ but with an
    *inner* spherical cavity (r < r_min is empty).

    Convergence requires p > 0.5.
    """
    if p <= 0.5:
        raise ValueError("plsphole requires p > 0.5 for convergence.")

    R = np.asanyarray(R, dtype=float)
    sigma = np.zeros_like(R)
    cbf = _beta_prefactor(p)
    pref = r_min * cbf

    # R ≥ r_min
    outer = R >= r_min
    if np.any(outer):
        sigma[outer] = pref * (R[outer] / r_min) ** (1.0 - 2.0 * p)

    # R < r_min
    inner = R < r_min
    if np.any(inner):
        ibor = _rincbeta((R[inner] / r_min) ** 2, p - 0.5, 0.5)
        sigma[inner] = pref * (R[inner] / r_min) ** (1.0 - 2.0 * p) * ibor

    return sigma


def plinfty(p: float, R: np.ndarray) -> np.ndarray:
    """
    Scale-free result for an *infinite* power-law distribution.

    Caller must supply their own normalisation.
    """
    return np.asanyarray(R, dtype=float) ** (1.0 - 2.0 * p)


# ------------------------------------------------------------------- #
# Public wrapper                                                      #
# ------------------------------------------------------------------- #


def projected_power_law(
    eps0: float,
    sindex: float,
    r_min: float,
    r_max: float | None,
    R: np.ndarray,
    *,
    c: float = 1.0,
    normalised_at_rmin: bool = False,
    zero_fudge: float = 1e-3,
) -> np.ndarray:
    """Line-of-sight integral of a piecewise power-law spherical
    distribution.

    Parameters
    ----------
    eps0
        Emissivity normalisation constant.
    sindex
        Spectral index (the *positive* power-law exponent on *r*).
    r_min, r_max
        Inner and outer 3-D radii of the bin.  Use ``r_max=None`` (or
        < 0) to indicate integration to ∞.
    R
        Projected radius array (scalar or ndarray).
    c
        Axis-ratio along the line of sight for an ellipsoid.
    normalised_at_rmin
        If *True* the caller already normalised ``eps0`` at *r_min*.
    zero_fudge
        Substitute this value for *exactly* R = 0 to avoid singularities.

    Returns
    -------
    sigma(R)
        Same shape as ``R``.
    """
    R = np.where(R == 0, zero_fudge, np.asanyarray(R, dtype=float))
    p = 0.5 * sindex

    # Choose the geometry
    if r_max is None or r_max < 0:
        sigma = plinfty(p, R) if r_min == 0 else plsphole(p, r_min, R)
    else:
        sigma = (
            plsphere(p, r_max, R)
            if r_min == 0
            else plshell(p, r_min, r_max, R)
        )

    # Normalisation
    if not normalised_at_rmin and r_min > 0 and r_max and r_max > 0:
        eps0 *= (r_max / r_min) ** sindex

    return eps0 * sigma * c

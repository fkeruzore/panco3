"""Cluster definition and static cosmological quantities.

These quantities (R_500, P_500, d_A, rho_crit, sz_fact, the Arnaud et al. 2010
universal-profile parameters) are computed **once** at setup from the fixed
``(z, M_500, cosmo)`` triple. They are *not* differentiated during sampling, so
we evaluate them eagerly and store plain Python floats: downstream they act as
static constants captured by the forward model.

Cosmology is provided as a ``jax_cosmo.Cosmology`` object. Distances come from
``jax_cosmo.background`` and the critical density from ``halox.cosmology``;
both work in ``h``-units, which we convert to panco2's physical
kpc / Msun / keV conventions here. See ``panco2/panco2/cluster.py`` for the
reference implementation this mirrors.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax_cosmo as jc
from jax_cosmo import background as jcb
from halox import cosmology as hcosmo
import astropy.units as u
from astropy.constants import sigma_T, c, m_e

# (sigma_T / (m_e c^2)) in [cm^3 keV^-1 kpc^-1]. Identical to panco2's sz_fact:
# multiply a pressure [keV cm^-3] integrated along a l.o.s. [kpc] to get a
# dimensionless Compton-y.
SZ_FACT = (sigma_T / (m_e * c**2)).to(u.cm**3 / u.keV / u.kpc).value

# Unit conversions between jax-cosmo / halox h-units and panco2 physical units.
_MPC_TO_KPC = 1.0e3
_RAD_TO_ARCMIN = float(u.rad.to("arcmin"))
_ARCSEC_TO_RAD = float(u.arcsec.to("rad"))


def cosmo_from_flat_lcdm(
    H0: float = 70.0,
    Om0: float = 0.3,
    Ob0: float = 0.05,
    n_s: float = 0.96,
    sigma8: float = 0.8,
) -> jc.Cosmology:
    """Build a flat-LCDM ``jax_cosmo.Cosmology``.

    The default reproduces panco2's ``FlatLambdaCDM(70.0, 0.3)`` for all
    *background* quantities (``n_s``/``sigma8`` only matter for the power
    spectrum, which panco3 does not use).
    """
    h = H0 / 100.0
    return jc.Cosmology(
        Omega_c=Om0 - Ob0,
        Omega_b=Ob0,
        h=h,
        n_s=n_s,
        sigma8=sigma8,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
    )


def default_cosmo() -> jc.Cosmology:
    """panco2's default cosmology: flat LCDM with ``H0=70``, ``Om0=0.3``."""
    return cosmo_from_flat_lcdm()


def _scalar(x) -> float:
    """Extract a Python float from a (possibly 1-element array)
    jax/np value."""
    return float(np.asarray(x).reshape(-1)[0])


class Cluster:
    """A galaxy cluster and its basic (static) properties.

    Parameters
    ----------
    z : float
        Cluster redshift.
    M_500 : float
        Cluster mass within R_500 [Msun].
    cosmo : jax_cosmo.Cosmology, optional
        Cosmology to assume. Defaults to flat LCDM with ``h=0.7``, ``Om0=0.3``.

    Attributes mirror ``panco2.cluster.Cluster``: ``E``, ``dens_crit``
    [Msun kpc-3], ``d_a`` [kpc], ``sz_fact`` [cm3 keV-1 kpc-1],
    ``R_500`` [kpc], ``theta_500`` [arcmin], ``P_500`` [keV cm-3], and
    ``A10_params`` ``[P_0*P_500, r_p, a, b, c]``.
    """

    def __init__(
        self, z: float, M_500: float, cosmo: jc.Cosmology | None = None
    ):
        if cosmo is None:
            cosmo = default_cosmo()
        self.cosmo = cosmo
        self.z = float(z)
        self.M_500 = float(M_500)

        h = float(cosmo.h)
        a = jnp.atleast_1d(1.0 / (1.0 + self.z))

        # Cosmology-related quantities, converted to physical units.
        self.E = _scalar(jnp.sqrt(jcb.Esqr(cosmo, a)))
        # halox critical_density is in [h^2 Msun Mpc^-3]; -> [Msun kpc^-3].
        self.dens_crit = (
            _scalar(hcosmo.critical_density(self.z, cosmo))
            * h**2
            / _MPC_TO_KPC**3
        )
        # jax-cosmo angular diameter distance is in [Mpc/h]; -> [kpc].
        self.d_a = (
            _scalar(jcb.angular_diameter_distance(cosmo, a)) / h * _MPC_TO_KPC
        )
        self.sz_fact = SZ_FACT

        # Characteristic radius/mass (eq. defining R_500).
        self.R_500 = (3 * self.M_500 / (4 * np.pi * 500 * self.dens_crit)) ** (
            1.0 / 3.0
        )
        self.theta_500 = np.arctan(self.R_500 / self.d_a) * _RAD_TO_ARCMIN

        # Arnaud et al. (2010) universal pressure profile, h_70-scaled.
        h_70 = h / 0.7
        self.P_500 = (
            1.65e-3
            * self.E ** (8.0 / 3.0)
            * (self.M_500 * h_70 / 3e14) ** (2.0 / 3.0 + 0.12)
            * h_70**2
        )  # eq. (13) of A10
        self.A10_params = np.array(  # eq. (12) of A10
            [
                8.403 * self.P_500,
                self.R_500 / 1.177,
                1.0510,
                5.4905,
                0.3081,
            ]
        )

    def arcsec2kpc(self, angle):
        """Convert an on-sky angle [arcsec] to a physical distance [kpc]."""
        return np.tan(np.asarray(angle) * _ARCSEC_TO_RAD) * self.d_a

    def kpc2arcsec(self, dist):
        """Convert a physical distance [kpc] to an on-sky angle [arcsec]."""
        return np.arctan(np.asarray(dist) / self.d_a) / _ARCSEC_TO_RAD

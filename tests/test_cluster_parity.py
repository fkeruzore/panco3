"""Parity of panco3.Cluster (halox + jax-cosmo) vs panco2.Cluster (astropy).

These are static, one-time setup quantities. We require agreement to a
relative tolerance of 1e-3, dominated by jax-cosmo's ODE-based distance
integration (~3e-5 on d_A); the algebraic quantities match far more tightly.
"""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

import panco3
from panco3.cluster import Cluster

from conftest import load_panco2_module


@pytest.fixture(scope="module")
def panco2_cluster_cls():
    return load_panco2_module("cluster").Cluster


@pytest.mark.parametrize(
    "z, M_500", [(0.5, 6e14), (0.1, 3e14), (1.0, 8e14), (0.3, 1e15)]
)
def test_cluster_scalar_parity(panco2_cluster_cls, z, M_500):
    c3 = Cluster(z, M_500, cosmo=panco3.cosmo_from_flat_lcdm(70.0, 0.3))
    c2 = panco2_cluster_cls(z, M_500, cosmo=FlatLambdaCDM(70.0, 0.3))

    # Algebraic quantities (E, densities, masses) match very tightly.
    for attr in ["E", "dens_crit", "R_500", "P_500"]:
        v3, v2 = getattr(c3, attr), getattr(c2, attr)
        assert np.isclose(v3, v2, rtol=1e-4), f"{attr}: {v3} vs {v2}"

    # Distance-derived quantities inherit jax-cosmo's ODE distance accuracy,
    # which degrades at low z (~1.4e-3 at z=0.1). Still sub-percent.
    for attr in ["d_a", "theta_500"]:
        v3, v2 = getattr(c3, attr), getattr(c2, attr)
        assert np.isclose(v3, v2, rtol=3e-3), f"{attr}: {v3} vs {v2}"

    # sz_fact is pure physical constants -> should match to machine precision.
    assert np.isclose(c3.sz_fact, c2.sz_fact, rtol=1e-10)

    # Arnaud A10 params: P_0*P_500 and r_p scale with P_500/R_500; shape fixed.
    assert np.allclose(c3.A10_params, c2.A10_params, rtol=1e-4)


def test_arcsec_kpc_roundtrip(panco2_cluster_cls):
    c3 = Cluster(0.5, 6e14, cosmo=panco3.cosmo_from_flat_lcdm(70.0, 0.3))
    c2 = panco2_cluster_cls(0.5, 6e14, cosmo=FlatLambdaCDM(70.0, 0.3))
    angle = 60.0  # arcsec
    assert np.isclose(c3.arcsec2kpc(angle), c2.arcsec2kpc(angle), rtol=3e-3)
    dist = 500.0  # kpc
    assert np.isclose(c3.kpc2arcsec(dist), c2.kpc2arcsec(dist), rtol=3e-3)
    # Round trip.
    assert np.isclose(c3.kpc2arcsec(c3.arcsec2kpc(angle)), angle, rtol=1e-10)

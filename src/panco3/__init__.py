"""panco3: a differentiable (JAX) pressure-profile fitter from tSZ
observations.

A JAX port of panco2 with a differentiable, GPU-capable forward model, enabling
Hamiltonian Monte Carlo / NUTS inference (via BlackJAX) of cluster electron
pressure profiles from thermal Sunyaev-Zeldovich maps.
"""

# Import config FIRST so that float64 is enabled before any array is created.
from . import config  # noqa: F401
from . import (
    utils,
    interp,
    integrate,
    geometry,
    filtering,
    model,
    io,
    fitter,
    priors,
    posterior,
    inference,
    results,
)
from .cluster import Cluster, cosmo_from_flat_lcdm, default_cosmo, SZ_FACT
from .model import ModelBinned
from .filtering import Filter, Filter1d, Filter2d
from .geometry import make_radii
from .fitter import PressureProfileFitter

__all__ = [
    "config",
    "utils",
    "interp",
    "integrate",
    "geometry",
    "filtering",
    "model",
    "io",
    "fitter",
    "priors",
    "posterior",
    "inference",
    "results",
    "Cluster",
    "cosmo_from_flat_lcdm",
    "default_cosmo",
    "SZ_FACT",
    "ModelBinned",
    "Filter",
    "Filter1d",
    "Filter2d",
    "make_radii",
    "PressureProfileFitter",
]

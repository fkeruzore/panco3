"""Small shared utilities (JAX-friendly)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def gNFW(r, P0, rp, a, b, c) -> Array:
    """Generalized NFW pressure profile (matches ``panco2.utils.gNFW``)."""
    x = jnp.asarray(r) / rp
    return P0 / ((x**c) * (1.0 + x**a) ** ((b - c) / a))


def gNFW_from_params(r, params) -> Array:
    """``gNFW(r, *params)`` for a 5-vector ``[P0, rp, a, b, c]`` (e.g. A10)."""
    P0, rp, a, b, c = params
    return gNFW(r, P0, rp, a, b, c)

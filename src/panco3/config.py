"""Global JAX configuration for panco3.

This module **must** be imported before any JAX array is created so that
double precision (float64) is enabled. panco2 is a pure-float64 code and its
validation tolerances are tight; running panco3 in JAX's default float32 would
break parity with panco2 and degrade the conditioning of the HMC gradients.
"""

import jax

# Enable 64-bit precision globally. This is a hard requirement for parity with
# panco2 (see plan: "x64 everywhere").
jax.config.update("jax_enable_x64", True)


def using_x64() -> bool:
    """Return True if JAX is configured for float64 (it always should be)."""
    return jax.config.jax_enable_x64


def default_device() -> str:
    """Return the platform of the default JAX device
    (``cpu``/``gpu``/``tpu``)."""
    return jax.devices()[0].platform

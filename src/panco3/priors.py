"""Differentiable priors with unconstrained reparametrization (for HMC).

panco2 specifies per-parameter priors as ``scipy.stats`` distributions
(``define_priors``), but ``scipy.stats`` log-densities are not JAX
differentiable, and HMC needs an *unconstrained* sampling space. Each
prior here exposes:

* ``constrain(z)``  -- map an unconstrained scalar ``z in R`` to the natural
  parameter ``theta`` (e.g. a positive pressure).
* ``log_prob(z)``   -- the prior log-density evaluated in the *sampling*
  coordinate ``z`` (including the change-of-variables Jacobian), so its
  pushforward through ``constrain`` is the intended prior on ``theta``.

This lets the sampler run on an unconstrained vector while the model sees
natural parameters, with smooth, finite gradients everywhere (no hard
``-inf`` walls).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax.nn as jnn
from jax.scipy.stats import norm


def _gauss_pdf(x, loc, scale):
    return np.exp(-0.5 * ((x - loc) / scale) ** 2) / (
        scale * np.sqrt(2 * np.pi)
    )


class Normal:
    """Gaussian prior on a real parameter (identity coordinate).

    Suitable for ``conv``, ``zero``, and point-source fluxes.
    """

    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def constrain(self, z):
        return z

    def log_prob(self, z):
        return norm.logpdf(z, self.loc, self.scale)

    def init(self):
        return float(self.loc)

    def pdf(self, x):
        """Prior density in *natural*-parameter space (for plotting)."""
        return _gauss_pdf(np.asarray(x, dtype=float), self.loc, self.scale)


class LogNormal:
    """Log-normal prior on a positive parameter; sample ``z = log(theta)``.

    The prior is ``Normal(loc, scale)`` on ``log(theta)`` and we sample in
    that coordinate, so no extra Jacobian is needed. Ideal for pressure bins
    under HMC (unbounded, smooth). ``loc``/``scale`` are in natural log
    units.
    """

    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def constrain(self, z):
        return jnp.exp(z)

    def log_prob(self, z):
        return norm.logpdf(z, self.loc, self.scale)

    def init(self):
        return float(self.loc)

    def pdf(self, x):
        """Log-normal density in natural-parameter space (for plotting)."""
        x = np.asarray(x, dtype=float)
        safe = np.where(x > 0, x, 1.0)
        g = _gauss_pdf(np.log(safe), self.loc, self.scale)
        return np.where(x > 0, g / safe, 0.0)


class LogUniform:
    """Log-uniform prior on ``theta in [low, high]`` (panco2's pressure prior).

    Reparametrized to an unconstrained ``z`` via a sigmoid so HMC sees no
    hard bounds: ``u = log(low) + (log(high)-log(low)) * sigmoid(z)``,
    ``theta = exp(u)``. The Jacobian makes the pushforward flat in
    ``log theta`` (i.e. ``p(theta) ∝ 1/theta`` on ``[low, high]``).
    """

    def __init__(self, low: float, high: float):
        self.log_low = float(jnp.log(low))
        self.log_high = float(jnp.log(high))
        self._span = self.log_high - self.log_low

    def _u(self, z):
        return self.log_low + self._span * jnn.sigmoid(z)

    def constrain(self, z):
        return jnp.exp(self._u(z))

    def log_prob(self, z):
        # log| d u / d z | = log(span) + log sigmoid(z) + log sigmoid(-z);
        # the uniform density on u is constant (dropped).
        return jnn.log_sigmoid(z) + jnn.log_sigmoid(-z) + jnp.log(self._span)

    def init(self):
        return 0.0  # sigmoid(0) -> midpoint in log-space

    def pdf(self, x):
        """``p(theta) = 1/(theta * span)`` on ``[low, high]`` (for plotting)."""
        x = np.asarray(x, dtype=float)
        low, high = np.exp(self.log_low), np.exp(self.log_high)
        safe = np.where(x > 0, x, 1.0)
        val = 1.0 / (safe * self._span)
        return np.where((x >= low) & (x <= high), val, 0.0)

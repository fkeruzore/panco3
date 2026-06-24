"""Build a pure, differentiable log-posterior over unconstrained parameters.

Ports panco2's likelihood/posterior (``panco2/panco2/panco2.py:578-609,
895``) to a single JAX function ``log_posterior(z)`` where ``z`` is the
*unconstrained* parameter vector (see :mod:`panco3.priors`). The function
closes over the static data, model, and priors, so ``jax.grad``/``jax.jit``
give the gradients BlackJAX needs.
"""

from __future__ import annotations

import jax.numpy as jnp


def ordered_priors(model, priors_spec: dict) -> list:
    """Build the per-parameter prior list, in ``par_vec`` index order.

    ``priors_spec`` mirrors :meth:`PressureProfileFitter.define_priors`:
    ``P_bins`` (one prior for all bins, or a list of ``n_bins``), ``conv``,
    ``zero`` (if the model has a zero level), and ``ps_fluxes`` (one per
    source).
    """
    out = []

    P_bins = priors_spec["P_bins"]
    if isinstance(P_bins, (list, tuple)):
        assert len(P_bins) == model.n_bins, "need one P prior per bin"
        out.extend(P_bins)
    else:
        out.extend([P_bins] * model.n_bins)

    out.append(priors_spec["conv"])
    if model.i_zero is not None:
        out.append(priors_spec["zero"])

    ps = priors_spec.get("ps_fluxes") or []
    assert len(ps) == model.n_ps, "need one flux prior per point source"
    out.extend(ps)

    assert len(out) == model.n_params, (
        f"prior count {len(out)} != n_params {model.n_params}"
    )
    return out


def make_log_posterior(fitter):
    """Return ``(log_posterior, prior_list, init_z, constrain)`` for a fitter.

    * ``log_posterior(z)`` : scalar log-posterior at unconstrained ``z``.
    * ``prior_list``       : per-parameter :mod:`panco3.priors` objects.
    * ``init_z``           : a reasonable unconstrained starting point.
    * ``constrain(z)``     : map ``z`` -> natural ``par_vec``.
    """
    model = fitter.model
    prior_list = ordered_priors(model, fitter.priors)

    sz_map = fitter.sz_map
    sz_rms = fitter.sz_rms
    has_covmat = fitter.has_covmat
    inv_covmat = fitter.inv_covmat
    has_integ_Y = fitter.has_integ_Y
    if has_integ_Y:
        Y_obs, dY = fitter.integ_Y

    def constrain(z):
        return jnp.stack([p.constrain(z[i]) for i, p in enumerate(prior_list)])

    def log_posterior(z):
        theta = constrain(z)
        log_prior = 0.0
        for i, p in enumerate(prior_list):
            log_prior = log_prior + p.log_prob(z[i])

        model_map = model.model_map(theta)
        resid = sz_map - model_map
        if has_covmat:
            d = resid.ravel()
            ll = -0.5 * (d @ (inv_covmat @ d))
        else:
            ll = -0.5 * jnp.sum((resid / sz_rms) ** 2)

        if has_integ_Y:
            ll = ll - 0.5 * ((model.integ_Y(theta) - Y_obs) / dY) ** 2

        return ll + log_prior

    init_z = jnp.asarray([p.init() for p in prior_list])
    return log_posterior, prior_list, init_z, constrain

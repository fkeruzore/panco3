"""HMC/NUTS inference with BlackJAX.

Replaces panco2's ``emcee`` ensemble sampler. Given the differentiable
unconstrained log-posterior from :func:`panco3.posterior.make_log_posterior`,
we run BlackJAX window adaptation (step size + mass matrix) followed by NUTS,
optionally over several chains, and return the samples (with an ArviZ
``InferenceData`` for diagnostics).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import blackjax

from .posterior import make_log_posterior


def _inference_loop(rng_key, kernel, initial_state, num_samples):
    """Run ``num_samples`` NUTS steps via ``lax.scan``; return states +
    infos."""

    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel.step(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos


def run_nuts(
    log_posterior,
    init_z,
    *,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    rng_key=None,
    target_acceptance_rate: float = 0.8,
    init_jitter: float = 0.1,
    dense_mass: bool = True,
):
    """Sample ``log_posterior`` with windowed-warmup NUTS.

    Returns a dict with ``samples`` (shape ``(num_chains, num_samples,
    n_dim)``, in unconstrained space), plus per-chain acceptance and
    divergence info.

    ``dense_mass=True`` adapts a full mass matrix during warmup, which is
    important when parameters are correlated (e.g. the conversion factor vs
    the overall pressure amplitude in tSZ fits) -- a diagonal mass matrix
    mixes such ridges very slowly.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    init_z = jnp.asarray(init_z)
    n_dim = init_z.shape[0]

    warmup_key, sample_key, init_key = jax.random.split(rng_key, 3)

    # Disperse chain starting points around init_z.
    z0 = init_z[None, :] + init_jitter * jax.random.normal(
        init_key, (num_chains, n_dim)
    )

    def run_one_chain(key, z_start):
        wkey, skey = jax.random.split(key)
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            log_posterior,
            is_mass_matrix_diagonal=not dense_mass,
            target_acceptance_rate=target_acceptance_rate,
        )
        (last_state, parameters), _ = warmup.run(
            wkey, z_start, num_steps=num_warmup
        )
        kernel = blackjax.nuts(log_posterior, **parameters)
        states, infos = _inference_loop(skey, kernel, last_state, num_samples)
        return states.position, infos.acceptance_rate, infos.is_divergent

    chain_keys = jax.random.split(sample_key, num_chains)
    samples, accept, divergent = jax.vmap(run_one_chain)(chain_keys, z0)

    return {
        "samples": samples,  # (num_chains, num_samples, n_dim), unconstrained
        "acceptance_rate": np.asarray(accept),
        "divergences": np.asarray(divergent),
    }


def constrained_samples(result, constrain) -> np.ndarray:
    """Map unconstrained samples to natural ``par_vec`` space.

    Returns array of shape ``(num_chains, num_samples, n_params)``.
    """
    z = result["samples"]
    constrained = jax.vmap(jax.vmap(constrain))(z)
    return np.asarray(constrained)


def to_arviz(result, param_names=None):
    """Build an ``arviz.InferenceData`` from unconstrained samples
    (diagnostics)."""
    import arviz as az

    samples = np.asarray(result["samples"])  # (chains, draws, dim)
    n_dim = samples.shape[-1]
    if param_names is None:
        param_names = [f"z_{i}" for i in range(n_dim)]
    posterior = {name: samples[:, :, i] for i, name in enumerate(param_names)}
    return az.from_dict(posterior=posterior)


def fit(fitter, **kwargs):
    """Convenience: build the log-posterior from ``fitter`` and run NUTS.

    Returns ``(result, constrain, log_posterior)``.
    """
    log_posterior, prior_list, init_z, constrain = make_log_posterior(fitter)
    result = run_nuts(log_posterior, init_z, **kwargs)
    return result, constrain, log_posterior

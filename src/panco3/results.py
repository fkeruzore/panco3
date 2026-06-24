"""Post-processing and plotting of posterior samples.

Lightweight, NumPy/matplotlib helpers (post-processing is not in the
gradient path). Covers what panco2's much larger ``results.py`` did for the
common case: turn samples into a named dict, reconstruct pressure-profile
credible intervals, and plot the data / model / residual maps. Diagnostics
(R-hat, ESS, trace, corner) go through ArviZ.
"""

from __future__ import annotations

import numpy as np


def constrained_to_dict(constrained_samples, model) -> dict[str, np.ndarray]:
    """Map a ``(..., n_params)`` sample array to ``{param_name: samples}``."""
    cs = np.asarray(constrained_samples)
    names = model.params
    return {names[i]: cs[..., i] for i in range(len(names))}


def median_par_vec(constrained_samples) -> np.ndarray:
    """Posterior-median parameter vector (flattening chains/draws)."""
    cs = np.asarray(constrained_samples)
    return np.median(cs.reshape(-1, cs.shape[-1]), axis=0)


def pressure_profile_ci(model, constrained_samples, r, quantiles=(16, 50, 84)):
    """Pressure profile ``P(r)`` percentiles from the posterior samples.

    Returns an array of shape ``(len(quantiles), len(r))`` [keV cm-3].
    """
    from .interp import interp_powerlaw
    import jax

    cs = np.asarray(constrained_samples).reshape(
        -1, np.asarray(constrained_samples).shape[-1]
    )
    P_samps = cs[:, : model.n_bins]
    r = np.asarray(r)

    def one(P_i):
        return interp_powerlaw(model.r_bins, P_i, r)

    profs = np.asarray(jax.vmap(one)(P_samps))  # (n_samples, len(r))
    return np.percentile(profs, quantiles, axis=0)


def plot_pressure_profile(model, constrained_samples, r, truth=None, ax=None):
    """Plot the recovered pressure profile with a 68% credible band."""
    import matplotlib.pyplot as plt

    lo, mid, hi = pressure_profile_ci(model, constrained_samples, r)
    if ax is None:
        _, ax = plt.subplots()
    ax.fill_between(r, lo, hi, alpha=0.3, label="68% CI")
    ax.plot(r, mid, label="median")
    if truth is not None:
        ax.plot(
            np.asarray(model.r_bins), np.asarray(truth), "ko", label="truth"
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="r [kpc]",
        ylabel="P [keV cm$^{-3}$]",
    )
    ax.legend()
    return ax


def plot_data_model_residual(fitter, par_vec, axes=None):
    """3-panel data / model / (data-model)/rms maps."""
    import matplotlib.pyplot as plt

    data = np.asarray(fitter.sz_map)
    rms = np.asarray(fitter.sz_rms)
    model_map = np.asarray(fitter.model_map(par_vec))
    resid = (data - model_map) / rms

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(12, 4))
    vmax = np.nanmax(np.abs(data))
    for ax, img, title, kw in [
        (axes[0], data, "data", dict(vmin=-vmax, vmax=vmax, cmap="RdBu_r")),
        (
            axes[1],
            model_map,
            "model",
            dict(vmin=-vmax, vmax=vmax, cmap="RdBu_r"),
        ),
        (
            axes[2],
            resid,
            "residual / rms",
            dict(vmin=-5, vmax=5, cmap="RdBu_r"),
        ),
    ]:
        im = ax.imshow(img, origin="lower", **kw)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    return axes

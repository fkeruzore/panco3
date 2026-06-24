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


def _truth_dict(truth, model) -> dict[str, float]:
    """Normalize a truth spec (dict or par_vec array) to ``{name: value}``."""
    if truth is None:
        return {}
    if isinstance(truth, dict):
        return {k: float(v) for k, v in truth.items()}
    truth = np.asarray(truth).ravel()
    return {model.params[i]: float(truth[i]) for i in range(len(truth))}


def to_inference_data(constrained_samples, model, var_names=None):
    """Build an ArviZ ``InferenceData`` in *constrained* (physical) space.

    Unlike :func:`panco3.inference.to_arviz` (which keeps the unconstrained
    sampling space), this labels each parameter by its natural name
    (``P_0``, ..., ``conv``, ``zero``, ...) so trace/corner plots line up
    with physical truth values.
    """
    import arviz as az

    cs = np.asarray(constrained_samples)  # (chains, draws, n_params)
    names = model.params if var_names is None else var_names
    idx = {n: i for i, n in enumerate(model.params)}
    posterior = {n: cs[..., idx[n]] for n in names}
    return az.from_dict(posterior=posterior)


def plot_trace(constrained_samples, model, truth=None, var_names=None):
    """Trace + marginal-density plot (ArviZ), with optional truth lines.

    ``truth`` may be a ``{name: value}`` dict or a full par_vec array. Returns
    the array of matplotlib axes from :func:`arviz.plot_trace`.
    """
    import arviz as az

    idata = to_inference_data(constrained_samples, model, var_names)
    names = (
        var_names if var_names is not None else list(idata.posterior.data_vars)
    )
    td = _truth_dict(truth, model)
    lines = [(n, {}, td[n]) for n in names if n in td] or None
    axes = az.plot_trace(idata, var_names=names, lines=lines, compact=False)
    axes.ravel()[0].figure.tight_layout()
    return axes


def _prior_dict(priors, model) -> dict:
    """Normalize a priors spec to ``{name: prior_obj}``.

    Accepts ``None``, a ``{name: prior}`` dict, or an ordered list aligned
    with ``model.params`` (e.g. the ``prior_list`` from
    :func:`panco3.posterior.make_log_posterior`).
    """
    if priors is None:
        return {}
    if isinstance(priors, dict):
        return priors
    return {
        model.params[i]: priors[i]
        for i in range(min(len(priors), len(model.params)))
    }


def _corner_range(x, truth_val=None, pad=0.05):
    lo, hi = float(np.min(x)), float(np.max(x))
    if truth_val is not None and np.isfinite(truth_val):
        lo, hi = min(lo, float(truth_val)), max(hi, float(truth_val))
    span = hi - lo
    d = (span * pad) if span > 0 else (abs(hi) * pad or 1e-6)
    return lo - d, hi + d


_DEFAULT_FRACS = (1 - np.exp(-0.5), 1 - np.exp(-2.0))


def _credible_levels(H, fracs=_DEFAULT_FRACS):
    """Density levels enclosing ``fracs`` of the probability mass.

    Defaults are the 2-D 1- and 2-sigma regions (``1 - exp(-n^2/2)``
    = 39.3% / 86.5%), the standard ``corner.py`` convention.
    """
    flat = np.sort(H.ravel())[::-1]
    csum = np.cumsum(flat)
    if csum[-1] == 0:
        return np.array([0.0])
    csum /= csum[-1]
    levels = [
        flat[min(np.searchsorted(csum, f), len(flat) - 1)] for f in fracs
    ]
    return np.unique(levels)  # ascending: [2-sigma, 1-sigma]


def plot_corner(
    constrained_samples,
    model,
    truth=None,
    priors=None,
    var_names=None,
    bins=24,
    smooth=1.0,
    color="C0",
    truth_color="C3",
    prior_color="C1",
    figsize=None,
):
    """Corner plot: contours below the diagonal, point cloud above.

    * **Lower triangle** -- filled 1- & 2-sigma credible contours (no points).
    * **Upper triangle** -- the posterior point cloud (no contours).
    * **Diagonal** -- 1-D marginal histogram, with the ``truth`` value as a
      vertical line and the ``priors`` density overlaid (both optional).

    ``truth`` may be a ``{name: value}`` dict or a full par_vec array.
    ``priors`` may be a ``{name: prior}`` dict or the ordered ``prior_list``
    from :func:`panco3.posterior.make_log_posterior`. Returns the
    ``(k, k)`` array of matplotlib axes.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.ndimage import gaussian_filter

    cs = np.asarray(constrained_samples)
    cs = cs.reshape(-1, cs.shape[-1])
    idx = {n: i for i, n in enumerate(model.params)}
    names = list(model.params) if var_names is None else list(var_names)
    data = {n: cs[:, idx[n]] for n in names}
    k = len(names)

    td = _truth_dict(truth, model)
    pri = _prior_dict(priors, model)
    rng = {n: _corner_range(data[n], td.get(n)) for n in names}

    base = np.array(mcolors.to_rgb(color))
    fill_colors = [
        tuple(0.55 + 0.45 * base),  # light  -> outer (2-sigma) band
        tuple(0.25 + 0.75 * base),  # darker -> inner (1-sigma) band
    ]

    if figsize is None:
        figsize = (2.3 * k, 2.3 * k)
    fig, axes = plt.subplots(k, k, figsize=figsize, squeeze=False)
    fig.subplots_adjust(wspace=0.07, hspace=0.07)

    for i in range(k):
        ni = names[i]
        for j in range(k):
            nj = names[j]
            ax = axes[i, j]

            if i == j:
                # --- diagonal: 1-D marginal + truth line + prior ---------- #
                h, edges = np.histogram(
                    data[ni], bins=bins, range=rng[ni], density=True
                )
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.fill_between(
                    centers, h, step="mid", color=color, alpha=0.25
                )
                ax.step(centers, h, where="mid", color=color, lw=1.2)
                ymax = (h.max() or 1.0) * 1.35
                if pri.get(ni) is not None:
                    xs = np.linspace(*rng[ni], 256)
                    ax.plot(
                        xs,
                        np.asarray(pri[ni].pdf(xs)),
                        color=prior_color,
                        ls="--",
                        lw=1.3,
                    )
                if ni in td:
                    ax.axvline(td[ni], color=truth_color, lw=1.4)
                ax.set_xlim(*rng[ni])
                ax.set_ylim(0, ymax)
                ax.set_yticks([])

            elif j < i:
                # --- lower triangle: filled credible contours ------------- #
                H, xe, ye = np.histogram2d(
                    data[nj], data[ni], bins=bins, range=[rng[nj], rng[ni]]
                )
                if smooth:
                    H = gaussian_filter(H, smooth)
                xc = 0.5 * (xe[:-1] + xe[1:])
                yc = 0.5 * (ye[:-1] + ye[1:])
                lv = _credible_levels(H)
                cf = np.concatenate([lv, [H.max() * (1 + 1e-6) + 1e-12]])
                nb = len(cf) - 1
                ax.contourf(xc, yc, H.T, levels=cf, colors=fill_colors[-nb:])
                ax.contour(
                    xc, yc, H.T, levels=lv, colors=color, linewidths=0.9
                )
                if ni in td and nj in td:
                    ax.plot(
                        td[nj],
                        td[ni],
                        "s",
                        color=truth_color,
                        ms=5,
                        mec="k",
                        mew=0.5,
                    )
                ax.set_xlim(*rng[nj])
                ax.set_ylim(*rng[ni])

            else:
                # --- upper triangle: posterior point cloud ---------------- #
                ax.scatter(
                    data[nj],
                    data[ni],
                    s=3,
                    alpha=0.15,
                    color=color,
                    edgecolors="none",
                    rasterized=True,
                )
                if ni in td and nj in td:
                    ax.plot(
                        td[nj],
                        td[ni],
                        "s",
                        color=truth_color,
                        ms=5,
                        mec="k",
                        mew=0.5,
                    )
                ax.set_xlim(*rng[nj])
                ax.set_ylim(*rng[ni])

            # --- ticks / labels: outer edges only ------------------------ #
            if i == k - 1:
                ax.set_xlabel(nj)
                for lab in ax.get_xticklabels():
                    lab.set_rotation(45)
                    lab.set_ha("right")
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(ni)
            elif i != j:
                ax.set_yticklabels([])

    return axes


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

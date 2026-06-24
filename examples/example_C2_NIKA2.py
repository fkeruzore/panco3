"""End-to-end panco3 example: fit a pressure profile to a NIKA2 tSZ map.

Mirrors panco2's ``examples/C2_NIKA2`` workflow, but with the differentiable
JAX forward model and BlackJAX NUTS sampling. Run with::

    uv run python examples/example_C2_NIKA2.py

It loads the mock NIKA2 map shipped with panco2, fits a 5-bin pressure profile,
and saves a profile-recovery figure, a data/model/residual figure, and NUTS
diagnostic figures (trace + corner, truth overlaid) to ``examples/output/``.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
from astropy.coordinates import SkyCoord

import panco3
from panco3 import priors, posterior, inference, results

HERE = os.path.dirname(__file__)
C2 = os.path.join(HERE, "..", "panco2", "examples", "C2_NIKA2")
FITS = os.path.join(C2, "C2_nk2.fits")
TF_FILE = os.path.join(C2, "nk2_tf.npz")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)


def main():
    # --- 1. Load data (mock NIKA2 map; HDU 1 = signal, HDU 5 = RMS) ---- #
    ppf = panco3.PressureProfileFitter(
        FITS,
        hdu_data=1,
        hdu_rms=5,
        z=0.5,
        M_500=6e14,
        coords_center=SkyCoord("12h00m00s +00d00m00s"),
        map_size=3.0,
    )

    # --- 2. Radial binning (beam scale -> ~1.1 x half-map) ----------------- #
    # Place bins at/above the resolution scale: bins far below the 18" beam are
    # unconstrained by the data, become prior-dominated, and stall HMC mixing.
    beam_kpc = ppf.cluster.arcsec2kpc(18.0)
    half_kpc = ppf.cluster.arcsec2kpc(ppf.map_size * 60 / 2)
    r_bins = np.logspace(np.log10(beam_kpc), np.log10(1.1 * half_kpc), 4)
    ppf.define_model(r_bins, n_nodes=16)

    # --- 3. Beam + transfer-function filtering ----------------------------- #
    # The mock map was generated WITH the NIKA2 angular transfer function (a
    # high-pass filter: tf -> 0 at low ell, killing extended signal). Fitting
    # with the beam ONLY would mis-model the data -- the model would predict
    # large-scale emission that has been filtered out, and the fit would
    # suppress the outer pressure bins to compensate, biasing the recovered
    # profile low (increasingly so outward). So we must apply the same TF.
    tf = np.load(TF_FILE)
    ppf.add_filtering(
        beam_fwhm=18.0, ell=tf["ell"], tf=tf["tf_150GHz"], pad=20
    )

    # --- 4. Priors: log-normal on pressures (A10 guess), normal nuisances -- #
    # NOTE on `conv`: the model is ``conv * filter(y(P))``, so `conv` (the
    # Compton-y -> map-unit conversion) trades multiplicatively against the
    # overall pressure amplitude. `conv` is a calibration factor known to ~%,
    # so a tight prior is physical and also avoids that (mild funnel)
    # degeneracy. (The priors sample in whitened/standardized coordinates --
    # see panco3.priors -- which is what makes NUTS mix efficiently here
    # despite the very different parameter scales and the transfer function.)
    P_a10 = np.asarray(
        panco3.utils.gNFW_from_params(r_bins, ppf.cluster.A10_params)
    )
    ppf.define_priors(
        P_bins=[priors.LogNormal(np.log(P), 2.0) for P in P_a10],
        conv=priors.Normal(-12.0, 0.05),
        zero=priors.Normal(0.0, 1e-4),
    )

    # --- 5. NUTS sampling -------------------------------------------------- #
    log_post, plist, init_z, constrain = posterior.make_log_posterior(ppf)
    print(f"log-posterior at init: {float(log_post(init_z)):.1f}", flush=True)
    print("running NUTS (dense mass matrix) ...", flush=True)
    result = inference.run_nuts(
        log_post,
        init_z,
        num_warmup=500,
        num_samples=500,
        num_chains=2,
        dense_mass=True,
        rng_key=jax.random.PRNGKey(0),
    )
    print(
        f"mean acceptance: {result['acceptance_rate'].mean():.2f}, "
        f"divergences: {int(result['divergences'].sum())}",
        flush=True,
    )

    cs = inference.constrained_samples(result, constrain)
    idata = inference.to_arviz(
        result, param_names=[f"z_{n}" for n in ppf.model.params]
    )
    import arviz as az

    print(az.summary(idata)[["mean", "sd", "r_hat", "ess_bulk"]])

    # --- 6. Figures -------------------------------------------------------- #
    # Truth: the Arnaud+2010 (A10) profile the mock map was generated from,
    # with the calibration `conv` and `zero` level at their nominal values.
    truth = {f"P_{i}": P for i, P in enumerate(P_a10)}
    truth["conv"] = -12.0
    truth["zero"] = 0.0

    r = np.logspace(np.log10(r_bins[0]), np.log10(r_bins[-1]), 50)
    ax = results.plot_pressure_profile(ppf.model, cs, r, truth=P_a10)
    ax.set_title("Recovered pressure profile (truth = A10)")
    ax.figure.savefig(
        os.path.join(OUT, "profile_recovery.png"), dpi=120, bbox_inches="tight"
    )

    med = results.median_par_vec(cs)
    axes = results.plot_data_model_residual(ppf, med)
    axes[0].figure.savefig(
        os.path.join(OUT, "data_model_residual.png"),
        dpi=120,
        bbox_inches="tight",
    )

    # Trace + marginals (truth lines overlaid).
    tr_axes = results.plot_trace(cs, ppf.model, truth=truth)
    tr_axes.ravel()[0].figure.savefig(
        os.path.join(OUT, "trace.png"), dpi=120, bbox_inches="tight"
    )

    # Corner plot: contours (lower) + point cloud (upper), with truth lines
    # and prior densities overlaid on the 1-D marginals.
    co_axes = results.plot_corner(cs, ppf.model, truth=truth, priors=plist)
    co_axes.ravel()[0].figure.savefig(
        os.path.join(OUT, "corner.png"), dpi=120, bbox_inches="tight"
    )

    plt.close("all")
    print(f"Saved figures to {OUT}")


if __name__ == "__main__":
    main()

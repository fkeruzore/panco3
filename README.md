# panco3

A differentiable, GPU-capable **JAX port of [panco2](./panco2)** for recovering
galaxy-cluster electron-pressure profiles from thermal Sunyaev–Zeldovich (tSZ)
maps. The forward model is fully autodiff-able, so inference uses **Hamiltonian
Monte Carlo / NUTS** (via [BlackJAX](https://github.com/blackjax-devs/blackjax))
instead of panco2's ensemble MCMC.

## What's different from panco2

| | panco2 | panco3 |
|---|---|---|
| Arrays / autodiff | NumPy | **JAX** (jit + grad, CPU/GPU) |
| Line-of-sight integral | analytic Abel transform (incomplete Beta) | **numerical quadrature** (differentiable; the analytic form's `betainc` has no parameter gradients in JAX) |
| Sampler | `emcee` (ensemble) | **BlackJAX NUTS** (gradient-based) |
| Cosmology / halo props | `astropy` | **`halox` + `jax-cosmo`** |

The forward model is validated against panco2 to ~1e-6 (see `tests/`).

## Install

```bash
uv sync --extra dev      # JAX (x64), blackjax, jax-cosmo, halox, astropy, arviz, ...
```

## Workflow

```python
import numpy as np
import panco3
from panco3 import priors, posterior, inference, results

ppf = panco3.PressureProfileFitter(
    "map.fits", hdu_data=1, hdu_rms=5, z=0.5, M_500=6e14, map_size=4.5,
)
r_bins = np.logspace(np.log10(20), np.log10(2000), 5)        # kpc
ppf.define_model(r_bins)
ppf.add_filtering(beam_fwhm=18.0, ell=ell, tf=tf)            # beam + transfer function

P0 = panco3.utils.gNFW_from_params(r_bins, ppf.cluster.A10_params)
ppf.define_priors(
    P_bins=[priors.LogNormal(np.log(P), 2.0) for P in P0],   # or priors.LogUniform(lo, hi)
    conv=priors.Normal(-12.0, 1.2),
    zero=priors.Normal(0.0, 1e-4),
)

result, constrain, log_post = inference.fit(
    ppf, num_warmup=500, num_samples=1000, num_chains=4,
)
samples = inference.constrained_samples(result, constrain)   # natural-parameter space
results.plot_pressure_profile(ppf.model, samples, r_bins)
```

See `examples/example_C2_NIKA2.py` for a full runnable example on the mock NIKA2
map shipped with panco2.

## Key modules (`src/panco3/`)

- `cluster.py` — static cluster/cosmology quantities (`halox` + `jax-cosmo`).
- `integrate.py` — differentiable LOS Compton-y quadrature (segmented Gauss–Legendre).
- `interp.py` — log-log / lin-log interpolation with extrapolation.
- `filtering.py` — FFT transfer function + `scipy`-matching Gaussian beam (JAX).
- `model.py` — `ModelBinned`: the full differentiable forward model.
- `priors.py` / `posterior.py` — differentiable priors and the unconstrained log-posterior.
- `inference.py` — BlackJAX NUTS driver (+ ArviZ diagnostics).
- `fitter.py` — `PressureProfileFitter` orchestrator (mirrors panco2's API).

## Notes / quirks inherited from panco2

- The pressure integral is **truncated at `r_bins[-1]`** (pressure treated as
  zero beyond the outermost bin), matching panco2. This strongly affects the
  predicted signal near the outer radius. Pass `r_max_integ` to `define_model`
  to include more of the tail.
- The analytic Abel transform is singular at integer slope `α=1`; the numerical
  quadrature handles it cleanly (one reason for the quadrature choice).

## Tests

```bash
uv run pytest -m "not slow"   # fast: parity, gradients, I/O, results
uv run pytest                 # also the NUTS recovery test (~100 s)
```

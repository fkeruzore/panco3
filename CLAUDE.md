# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (dev dependencies included)
uv sync

# Run tests (fast only — parity, gradients, I/O, results)
uv run pytest -m "not slow"

# Run all tests including end-to-end NUTS recovery (~100 s)
uv run pytest

# Run a single test file or test
uv run pytest tests/test_integrate.py
uv run pytest tests/test_model_parity.py::test_compton_map

# Lint and format (--respect-gitignore covers examples/ too)
uv run ruff check --respect-gitignore
uv run ruff format src/ tests/ examples/

# Run the example (writes figures to examples/output/)
uv run python examples/example_C2_NIKA2.py
```

## Architecture

panco3 is a JAX port of panco2 (vendored in `panco2/`). The user-facing API mirrors panco2 but uses HMC/NUTS instead of ensemble MCMC.

### Forward model pipeline

`par_vec` → pressure bins `P_i` → Compton-y profile `y(R)` (LOS quadrature) → 2-D y-map (interpolation) → filtered SZ map (beam + TF) → `* conv + zero`

`par_vec` layout: `[P_0 .. P_{n-1}, conv, zero, F_1 .. F_k]` where `zero` is optional and `F_i` are point-source fluxes. The `model.indices` dict maps parameter names to positions.

### Module roles

- `config.py` — **must be imported first** (enables float64 globally before any JAX array is created; `__init__.py` does this)
- `cluster.py` — static cluster/cosmology quantities (`Cluster`, `SZ_FACT`); these are computed once and stored as Python floats, not traced by JAX
- `integrate.py` — differentiable LOS Compton-y quadrature using segmented Gauss-Legendre; replaces the analytic incomplete-Beta Abel transform from panco2 (which has no JAX gradient w.r.t. shape parameters)
- `interp.py` — log-log power-law interpolation (`interp_powerlaw`) and radial-profile-to-2D-map projection (`prof2map`)
- `filtering.py` — FFT-based beam + transfer-function filtering; three classes: `Filter` (beam only), `Filter1d` (1-D azimuthally symmetric TF), `Filter2d` (2-D TF)
- `model.py` — `ModelBinned`: the full differentiable forward model; all steps are pure JAX functions of `par_vec`
- `priors.py` — differentiable priors with unconstrained reparametrization: `Normal` (identity), `LogNormal` (log coordinate), `LogUniform` (logit-sigmoid remap into `[low, high]`); each exposes `constrain(z)`, `log_prob(z)`, `init()`, `pdf(x)`
- `posterior.py` — `make_log_posterior(fitter)` assembles `log_posterior(z)` (unconstrained space) closing over data + model + priors; returns `(log_posterior, prior_list, init_z, constrain)`
- `inference.py` — BlackJAX NUTS driver: `run_nuts` runs windowed warmup + sampling (vmapped over chains); `fit(fitter)` is the convenience entry point
- `fitter.py` — `PressureProfileFitter` orchestrator; mirrors panco2's user-facing API; serializable with `dill`
- `results.py` — `plot_pressure_profile`, `plot_trace`, `plot_corner` (posterior visualization + optional truth overlay and prior density)

### Testing strategy

Tests in `tests/` use panco2 as a reference oracle. `conftest.py` loads individual panco2 submodules by file path (via `importlib`) without importing the full panco2 package (which pulls in `emcee`/`chainconsumer`). Parity tests verify the forward model matches panco2 to ~1e-6.

## Linting rules (ruff selects E, F, B; line-length = 79)

- **79-char line limit**: stricter than the ruff default (88). Docstrings and
  comments must wrap too — ruff enforces E501 on all lines.
- **No function calls in default args (B008)**: use a module-level constant
  instead. Example: `_DEFAULTS = (1 - np.exp(-0.5),)` then `def f(x=_DEFAULTS)`.
- **No ambiguous single-letter names (E741)**: avoid `l`, `O`, `I` as variable
  names; use descriptive alternatives (`ll`, `los`, etc.).
- **Lambda must bind loop variables (B023)**: capture loop variables via default
  arguments — `lambda x, v=val: ...` — not by closure over a mutable loop var.

## Key design constraints

- **float64 required**: JAX defaults to float32; `config.py` sets `jax_enable_x64=True`. Import `panco3` before creating any JAX arrays, or call `import panco3.config` first.
- **Pressure truncation at `r_bins[-1]`**: the LOS integral drops the outer tail by default (matching panco2). This strongly affects the signal near the outermost bin. Use `r_max_integ` in `define_model` to extend integration further.
- **Unconstrained sampling**: HMC samples in unconstrained space `z`; the prior's `constrain(z)` maps back to natural parameters. Always use `inference.constrained_samples(result, constrain)` before analyzing posteriors.
- **Dense mass matrix**: `run_nuts` defaults to `dense_mass=True` for full mass-matrix adaptation; important when pressure amplitudes correlate with the conversion factor `conv`.

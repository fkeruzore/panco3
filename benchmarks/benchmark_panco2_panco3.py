#!/usr/bin/env python3
"""Benchmark matched panco2 and panco3 NIKA2 fits.

The default problem is panco2's C2/NIKA2 validation map.  Each implementation
uses the same map crop, five pressure bins, 18-arcsec beam, NIKA2 transfer
function, log-uniform pressure priors, normal calibration/zero priors, and
the same number of concurrent samplers.

The samplers are different (emcee StretchMove versus BlackJAX NUTS), so a
"step" means one proposed draw *per sampler*: one emcee walker iteration or
one NUTS draw.  Every column runs the same fixed number of warmup steps
(discarded emcee burn-in, or BlackJAX window adaptation) followed by the same
number of sampling steps.  The report gives the total and sampling wall times
and, computed on the sampling draws only, the rank-normalized split R-hat,
the integrated autocorrelation time, and the bulk/tail ESS.

Run from the repository root after installing the benchmark dependency::

    uv sync --group dev
    uv run python benchmarks/benchmark_panco2_panco3.py

The command writes a Markdown table and a JSON sidecar.  It launches each
column in a fresh Python process so that forcing JAX to CPU for the panco3 CPU
case cannot affect the GPU case.  If no GPU JAX backend is installed, the GPU
column is retained and marked unavailable.  On systems where JAX does not
select the accelerator by default, pass ``--gpu-platform`` (for example,
``metal`` or ``cuda``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "panco2/validation/results/C2/NIKA2/input_map.fits"
DEFAULT_TF = ROOT / "panco2/validation/example_data/NIKA2/nk2_tf.npz"


@dataclass(frozen=True)
class Settings:
    """All options that define the matched inference problem."""

    map_file: str
    tf_file: str
    map_size: float
    n_bins: int
    n_nodes: int
    n_samplers: int
    workers: int
    gpu_chain_batch_size: int
    warmup: int
    steps: int
    chunk_size: int
    posterior_repeats: int
    throughput_batch: int
    seed: int


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".benchmarks" / "panco2_panco3.md",
        help="Markdown report path (a .json sidecar is also written).",
    )
    p.add_argument("--map-file", type=Path, default=DEFAULT_MAP)
    p.add_argument("--tf-file", type=Path, default=DEFAULT_TF)
    p.add_argument("--map-size", type=float, default=6.5)
    p.add_argument("--n-bins", type=int, default=5)
    p.add_argument(
        "--n-nodes",
        type=int,
        default=32,
        help="panco3 line-of-sight quadrature nodes (panco2 is analytic).",
    )
    p.add_argument(
        "--n-samplers",
        type=int,
        default=24,
        help="Both the emcee walker count and the NUTS chain count.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help=(
            "panco2 CPU worker processes, and panco3 CPU devices (chains "
            "are split evenly across them); default: min(8, CPU count)."
        ),
    )
    p.add_argument(
        "--gpu-chain-batch-size",
        type=int,
        default=None,
        help="panco3 GPU chains evaluated concurrently; defaults to all.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=500,
        help="Warmup steps per sampler (emcee burn-in / NUTS adaptation).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Sampling steps per sampler, after warmup.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "panco3 draws per compiled call, i.e. the progress-bar "
            "granularity; must divide --steps (default: min(100, --steps))."
        ),
    )
    p.add_argument("--posterior-repeats", type=int, default=20)
    p.add_argument(
        "--throughput-batch",
        type=int,
        default=None,
        help=(
            "Points per batched posterior/gradient call (panco3 vmap, "
            "panco2 worker pool); defaults to --n-samplers."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpu-platform",
        default=None,
        help="Optional JAX platform to force for the GPU worker (e.g. metal).",
    )
    p.add_argument(
        "--skip-gpu", action="store_true", help="Do not attempt the GPU case."
    )
    p.add_argument("--worker", choices=("panco2", "panco3-cpu", "panco3-gpu"))
    p.add_argument("--settings-json", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--result-json", type=Path, help=argparse.SUPPRESS)
    return p


def settings_from_args(args: argparse.Namespace) -> Settings:
    if args.n_samplers < 2 * (args.n_bins + 2):
        raise ValueError(
            "--n-samplers must be at least twice the parameter count "
            f"({2 * (args.n_bins + 2)}) for emcee."
        )
    if args.warmup < 1 or args.steps < 4:
        raise ValueError("--warmup must be >= 1 and --steps >= 4.")
    chunk_size = args.chunk_size or min(100, args.steps)
    if chunk_size < 1 or args.steps % chunk_size:
        raise ValueError("--chunk-size must be positive and divide --steps.")
    gpu_chain_batch_size = args.gpu_chain_batch_size or args.n_samplers
    throughput_batch = args.throughput_batch or args.n_samplers
    if min(args.workers, gpu_chain_batch_size, throughput_batch) < 1:
        raise ValueError(
            "--workers, --gpu-chain-batch-size and --throughput-batch must "
            "be positive."
        )
    # panco3 CPU splits chains and throughput points evenly over one XLA
    # device per worker.
    if args.n_samplers % args.workers or throughput_batch % args.workers:
        raise ValueError(
            "--n-samplers and --throughput-batch must be multiples of "
            "--workers."
        )
    return Settings(
        map_file=str(args.map_file.resolve()),
        tf_file=str(args.tf_file.resolve()),
        map_size=args.map_size,
        n_bins=args.n_bins,
        n_nodes=args.n_nodes,
        n_samplers=args.n_samplers,
        workers=args.workers,
        gpu_chain_batch_size=gpu_chain_batch_size,
        warmup=args.warmup,
        steps=args.steps,
        chunk_size=chunk_size,
        posterior_repeats=args.posterior_repeats,
        throughput_batch=throughput_batch,
        seed=args.seed,
    )


def radial_bins(fitter: Any, n_bins: int) -> np.ndarray:
    """The C2/NIKA2 binning used by panco2 validation."""
    pix_kpc = fitter.cluster.arcsec2kpc(fitter.pix_size)
    half_map_kpc = fitter.cluster.arcsec2kpc(fitter.map_size * 30.0)
    beam_kpc = fitter.cluster.arcsec2kpc(18.0)
    return np.concatenate(
        (
            [pix_kpc],
            np.logspace(
                np.log10(beam_kpc), np.log10(1.1 * half_map_kpc), n_bins - 1
            ),
        )
    )


def diagnostics(draws: np.ndarray) -> dict[str, Any]:
    """Convergence metrics of post-warmup draws, shaped (chain, draw, param).

    R-hat and ESS are ArviZ's rank-normalized estimators.  The integrated
    autocorrelation time is emcee's estimator (autocorrelation averaged over
    chains), in steps; it is only trusted when the chain is at least 50 times
    longer than the estimate.
    """
    import arviz as az
    from emcee.autocorr import integrated_time

    n_params = draws.shape[-1]
    rhat = [az.rhat(draws[..., i], method="rank") for i in range(n_params)]
    ess_bulk = [az.ess(draws[..., i], method="bulk") for i in range(n_params)]
    ess_tail = [az.ess(draws[..., i], method="tail") for i in range(n_params)]
    # emcee expects (step, walker, param); quiet=True warns instead of
    # raising when the chain is too short for a reliable estimate.
    tau = integrated_time(np.swapaxes(draws, 0, 1), quiet=True)
    max_tau = float(np.nanmax(tau))
    return {
        "max_rhat": float(np.nanmax(rhat)),
        "max_autocorr_time": max_tau,
        "autocorr_reliable": bool(draws.shape[1] >= 50 * max_tau),
        "min_ess": float(np.nanmin(np.minimum(ess_bulk, ess_tail))),
    }


def natural_starts(
    pressure: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """``n`` natural-space points well inside the common prior support."""
    pressure = np.asarray(pressure)
    center = np.concatenate((pressure, [-12.0, 0.0]))
    scale = np.concatenate((0.10 * pressure, [0.09, 1e-6]))
    starts = center + rng.normal(size=(n, center.size)) * scale
    starts[:, : pressure.size] = np.clip(
        starts[:, : pressure.size], 0.011 * pressure, 99.0 * pressure
    )
    return starts


def natural_to_z(theta: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Map natural-space points exactly to panco3's unconstrained z space."""
    n_bins = pressure.size
    span = np.log(100.0 / 0.01)
    frac = (np.log(theta[:, :n_bins]) - np.log(0.01 * pressure)) / span
    z_press = np.log(frac / (1.0 - frac))
    z_other = np.column_stack(
        ((theta[:, n_bins] + 12.0) / 0.9, theta[:, n_bins + 1] / 1e-5)
    )
    return np.column_stack((z_press, z_other))


def throughput_points(pressure: np.ndarray, settings: Settings) -> np.ndarray:
    """Natural-space points for batched timing, independent of the starts."""
    rng = np.random.default_rng(settings.seed + 1)
    return natural_starts(pressure, settings.throughput_batch, rng)


def measure_panco2_posterior(
    log_probability: Any, positions: np.ndarray, repeats: int
) -> float:
    """Steady-state serial time for one panco2 posterior evaluation."""
    for position in positions:
        log_probability(position)
    start = time.perf_counter()
    for _ in range(repeats):
        for position in positions:
            log_probability(position)
    elapsed = time.perf_counter() - start
    return elapsed / (repeats * len(positions))


def measure_panco2_throughput(
    log_probability: Any, positions: np.ndarray, settings: Settings
) -> float:
    """Posterior evaluations per second through a worker pool.

    This is how emcee evaluates a batch of walkers, so it is panco2's
    counterpart to a vmapped panco3 call.
    """
    from multiprocessing import Pool

    with Pool(processes=settings.workers) as pool:
        # The first map starts the workers and ships the model to them.
        pool.map(log_probability, positions)
        start = time.perf_counter()
        for _ in range(settings.posterior_repeats):
            pool.map(log_probability, positions)
        elapsed = time.perf_counter() - start
    return settings.posterior_repeats * len(positions) / elapsed


def run_panco2(settings: Settings) -> dict[str, Any]:
    """Run a fixed emcee burn-in, then a fixed number of steps."""
    import emcee
    import panco2 as p2
    import scipy.stats as ss
    from astropy.coordinates import SkyCoord

    ppf = p2.PressureProfileFitter(
        settings.map_file,
        1,
        5,
        z=0.5,
        M_500=6e14,
        coords_center=SkyCoord("12h00m00s +00d00m00s"),
        map_size=settings.map_size,
    )
    r_bins = radial_bins(ppf, settings.n_bins)
    ppf.define_model(r_bins)
    tf = np.load(settings.tf_file)
    ppf.add_filtering(
        beam_fwhm=18.0, ell=tf["ell"], tf=tf["tf_150GHz"], pad=20
    )
    pressure = p2.utils.gNFW(r_bins, *ppf.cluster.A10_params)
    ppf.define_priors(
        P_bins=[
            ss.loguniform(0.01 * value, 100.0 * value) for value in pressure
        ],
        conv=ss.norm(-12.0, 0.9),
        zero=ss.norm(0.0, 1e-5),
    )

    from functools import partial

    from panco2.panco2 import log_post

    # A partial (not a closure) so that the worker pool can pickle it.
    log_probability = partial(
        log_post, log_lhood=ppf._log_lhood, log_prior=ppf.model.log_prior
    )

    rng = np.random.default_rng(settings.seed)
    starts = natural_starts(pressure, settings.n_samplers, rng)
    posterior_seconds = measure_panco2_posterior(
        log_probability, starts, settings.posterior_repeats
    )
    posterior_per_second = measure_panco2_throughput(
        log_probability, throughput_points(pressure, settings), settings
    )

    # Import here to avoid creating worker processes while setup is timed.
    from multiprocessing import Pool

    # Total time includes worker start-up, the parallel counterpart of JAX
    # compilation for panco3.
    started = time.perf_counter()
    with Pool(processes=settings.workers) as pool:
        sampler = emcee.EnsembleSampler(
            settings.n_samplers,
            starts.shape[1],
            log_post,
            pool=pool,
            args=[ppf._log_lhood, ppf.model.log_prior],
        )
        state = starts
        bar = tqdm(total=settings.warmup, desc="panco2 warmup", unit="step")
        for step in sampler.sample(state, iterations=settings.warmup):
            state = step
            bar.update(1)
        bar.close()
        sampling_started = time.perf_counter()
        bar = tqdm(total=settings.steps, desc="panco2 sampling", unit="step")
        for step in sampler.sample(state, iterations=settings.steps):
            state = step
            bar.update(1)
        bar.close()
        finished = time.perf_counter()
    # emcee stores draws as (draw, walker, parameter); burn-in is discarded.
    draws = np.swapaxes(sampler.get_chain(discard=settings.warmup), 0, 1)
    return {
        "implementation": "panco2 (CPU-only)",
        "device": "cpu",
        "warmup_steps": settings.warmup,
        "sampling_steps": int(draws.shape[1]),
        "posterior_seconds": posterior_seconds,
        "posterior_per_second": posterior_per_second,
        "total_seconds": finished - started,
        "sampling_seconds": finished - sampling_started,
        "n_samplers": settings.n_samplers,
        **diagnostics(draws),
    }


def run_panco3(settings: Settings, expected_device: str) -> dict[str, Any]:
    """Run NUTS window adaptation, then a fixed number of draws."""
    import blackjax
    import jax
    import jax.numpy as jnp
    from astropy.coordinates import SkyCoord
    import panco3
    from panco3 import inference, posterior, priors, utils

    device = jax.devices()[0].platform
    if expected_device == "cpu" and device != "cpu":
        raise RuntimeError(f"expected CPU JAX backend, got {device!r}")
    if expected_device == "gpu" and device == "cpu":
        raise RuntimeError("JAX selected CPU; no GPU backend is available")

    ppf = panco3.PressureProfileFitter(
        settings.map_file,
        1,
        5,
        z=0.5,
        M_500=6e14,
        coords_center=SkyCoord("12h00m00s +00d00m00s"),
        map_size=settings.map_size,
    )
    r_bins = radial_bins(ppf, settings.n_bins)
    ppf.define_model(r_bins, n_nodes=settings.n_nodes)
    tf = np.load(settings.tf_file)
    ppf.add_filtering(
        beam_fwhm=18.0, ell=tf["ell"], tf=tf["tf_150GHz"], pad=20
    )
    pressure = np.asarray(
        utils.gNFW_from_params(r_bins, ppf.cluster.A10_params)
    )
    ppf.define_priors(
        P_bins=[
            priors.LogUniform(0.01 * value, 100.0 * value)
            for value in pressure
        ],
        conv=priors.Normal(-12.0, 0.9),
        zero=priors.Normal(0.0, 1e-5),
    )
    log_posterior, _, init_z, constrain = posterior.make_log_posterior(ppf)

    rng = np.random.default_rng(settings.seed)
    # Use shared natural-space starts, transformed exactly to panco3's z space.
    z0 = jnp.asarray(
        natural_to_z(
            natural_starts(pressure, settings.n_samplers, rng), pressure
        )
    )

    single_log_posterior = jax.jit(log_posterior)
    single_log_posterior(z0[0]).block_until_ready()
    # This deliberately times scalar calls rather than a vmap throughput
    # measurement: it is the same unit of work as the panco2 timing.
    start = time.perf_counter()
    for _ in range(settings.posterior_repeats):
        for position in z0:
            single_log_posterior(position).block_until_ready()
    posterior_seconds = (time.perf_counter() - start) / (
        settings.posterior_repeats * settings.n_samplers
    )

    # Chains are laid out as (device, chain on device): pmap runs each
    # device's share independently, vmap batches within a device.  The CPU
    # worker has one XLA device per panco2 worker; the GPU column uses one
    # GPU.
    devices = jax.local_devices()
    if expected_device == "gpu":
        devices = devices[:1]
    n_devices = len(devices)
    batch_size = (
        settings.gpu_chain_batch_size
        if expected_device == "gpu"
        else settings.n_samplers
    )
    if batch_size % n_devices or settings.throughput_batch % n_devices:
        raise RuntimeError(
            f"chain batches and throughput points must split evenly over "
            f"{n_devices} devices"
        )

    def per_device(x: Any) -> Any:
        return x.reshape(n_devices, x.shape[0] // n_devices, *x.shape[1:])

    def parallel(fn: Any) -> Any:
        return jax.pmap(jax.vmap(fn), devices=devices)

    # Batched throughput is the GPU-relevant figure: one vmapped call over
    # many points amortizes kernel launches that dominate a scalar call.
    z_batch = per_device(
        jnp.asarray(
            natural_to_z(throughput_points(pressure, settings), pressure)
        )
    )

    def per_second(batched: Any) -> float:
        jax.block_until_ready(batched(z_batch))
        start = time.perf_counter()
        for _ in range(settings.posterior_repeats):
            jax.block_until_ready(batched(z_batch))
        elapsed = time.perf_counter() - start
        return settings.posterior_repeats * settings.throughput_batch / elapsed

    posterior_per_second = per_second(parallel(log_posterior))
    gradient_per_second = per_second(
        parallel(jax.value_and_grad(log_posterior))
    )

    warmup_key, sample_key = jax.random.split(
        jax.random.PRNGKey(settings.seed)
    )
    chain_keys = jax.random.split(warmup_key, settings.n_samplers)

    def warm_one(key, position):
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            log_posterior,
            is_mass_matrix_diagonal=False,
            target_acceptance_rate=0.8,
        )
        return warmup.run(key, position, num_steps=settings.warmup)[0]

    def draw_chunk(key, state, parameters):
        kernel = blackjax.nuts(log_posterior, **parameters)
        trajectory, infos = inference._inference_loop(
            key, kernel, state, settings.chunk_size
        )
        last_state = jax.tree.map(lambda leaf: leaf[-1], trajectory)
        return (
            last_state,
            trajectory.position,
            infos.num_integration_steps,
            infos.is_divergent,
        )

    # Built once so equal-sized chain batches reuse one compiled program.
    run_warmup = parallel(warm_one)
    run_chunk = parallel(draw_chunk)
    started = time.perf_counter()
    state_batches = []
    parameter_batches = []
    z_batches = []
    firsts = range(0, settings.n_samplers, batch_size)
    for first in tqdm(firsts, desc="panco3 warmup", unit="batch"):
        last = min(first + batch_size, settings.n_samplers)
        states, parameters = run_warmup(
            per_device(chain_keys[first:last]), per_device(z0[first:last])
        )
        jax.block_until_ready(states)
        state_batches.append(states)
        parameter_batches.append(parameters)
        z_batches.append((first, last))

    # Compile the sampling kernel ahead of time (once per distinct batch
    # size), so that compilation counts in the total but not the sampling
    # time.
    compiled: dict[int, Any] = {}
    for batch, (first, last) in enumerate(z_batches):
        if last - first not in compiled:
            compiled[last - first] = run_chunk.lower(
                per_device(chain_keys[first:last]),
                state_batches[batch],
                parameter_batches[batch],
            ).compile()

    positions: list[list[np.ndarray]] = [[] for _ in z_batches]
    leapfrogs: list[np.ndarray] = []
    divergent: list[np.ndarray] = []
    sampling_started = time.perf_counter()
    # One unit is one draw for one chain batch, so the bar moves after every
    # batch rather than only once all batches have finished a chunk.
    bar = tqdm(
        total=settings.steps * len(z_batches),
        desc="panco3 sampling",
        unit="draw",
    )
    for _chunk in range(settings.steps // settings.chunk_size):
        sample_key, chunk_key = jax.random.split(sample_key)
        keys = jax.random.split(chunk_key, settings.n_samplers)
        for batch, (first, last) in enumerate(z_batches):
            states, chunk_positions, n_steps, is_divergent = compiled[
                last - first
            ](
                per_device(keys[first:last]),
                state_batches[batch],
                parameter_batches[batch],
            )
            state_batches[batch] = states
            # Copying to the host also waits for the chunk to finish.
            chunk_positions = np.asarray(chunk_positions)
            positions[batch].append(
                chunk_positions.reshape(-1, *chunk_positions.shape[2:])
            )
            leapfrogs.append(np.asarray(n_steps).ravel())
            divergent.append(np.asarray(is_divergent).ravel())
            bar.update(settings.chunk_size)
    bar.close()
    finished = time.perf_counter()

    # (chain, draw, param) in natural space, mapped after timing stops.
    z_draws = np.concatenate(
        [np.concatenate(chunks, axis=1) for chunks in positions], axis=0
    )
    draws = np.asarray(jax.jit(jax.vmap(jax.vmap(constrain)))(z_draws))
    leapfrogs_all = np.concatenate(leapfrogs)
    divergent_all = np.concatenate(divergent)
    return {
        "implementation": f"panco3 ({device})",
        "device": device,
        "n_devices": n_devices,
        "warmup_steps": settings.warmup,
        "sampling_steps": int(draws.shape[1]),
        "posterior_seconds": posterior_seconds,
        "posterior_per_second": posterior_per_second,
        "gradient_per_second": gradient_per_second,
        "total_seconds": finished - started,
        "sampling_seconds": finished - sampling_started,
        "n_samplers": settings.n_samplers,
        "chain_batch_size": batch_size,
        "mean_leapfrog_steps": float(leapfrogs_all.mean()),
        "max_leapfrog_steps": int(leapfrogs_all.max()),
        "divergent_fraction": float(divergent_all.mean()),
        **diagnostics(draws),
    }


def worker(args: argparse.Namespace) -> None:
    settings = Settings(**json.loads(args.settings_json.read_text()))
    if args.worker == "panco2":
        result = run_panco2(settings)
    elif args.worker == "panco3-cpu":
        result = run_panco3(settings, "cpu")
    else:
        result = run_panco3(settings, "gpu")
    result["ess_per_second"] = result["min_ess"] / result["sampling_seconds"]
    args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True))


def run_worker(
    case: str,
    settings: Settings,
    settings_path: Path,
    result_path: Path,
    gpu_platform: str | None,
) -> dict[str, Any]:
    env = os.environ.copy()
    if case == "panco3-cpu":
        env["JAX_PLATFORMS"] = "cpu"
        # One XLA CPU device per worker: a single device runs its program
        # mostly on one thread, so this is panco3's counterpart to panco2's
        # process pool.
        devices = f"--xla_force_host_platform_device_count={settings.workers}"
        env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} {devices}".strip()
    elif case == "panco3-gpu":
        # Leaving this unset lets JAX select the installed accelerator backend.
        env.pop("JAX_PLATFORMS", None)
        if gpu_platform is not None:
            env["JAX_PLATFORMS"] = gpu_platform
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        case,
        "--settings-json",
        str(settings_path),
        "--result-json",
        str(result_path),
    ]
    # stderr is inherited so worker progress bars and tracebacks stay visible.
    completed = subprocess.run(
        command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE
    )
    if completed.returncode:
        return {
            "implementation": case,
            "unavailable": True,
            "error": (
                completed.stdout.strip()
                or f"worker exited with code {completed.returncode}; "
                "see stderr above"
            ),
        }
    return json.loads(result_path.read_text())


def value(result: dict[str, Any], key: str, fmt: str) -> str:
    """Format one table cell with ``fmt`` (a str.format spec + unit)."""
    if result.get("unavailable"):
        return "unavailable"
    if key not in result:
        return "n/a"
    cell = fmt.format(result[key])
    if key == "max_autocorr_time" and not result["autocorr_reliable"]:
        cell += " (unreliable)"
    return cell


# (label, result key, cell format)
ROWS = (
    ("Warmup steps per sampler", "warmup_steps", "{}"),
    ("Sampling steps per sampler", "sampling_steps", "{}"),
    (
        "Time per posterior evaluation (serial)",
        "posterior_seconds",
        "{:.3g} s",
    ),
    ("Batched posterior throughput", "posterior_per_second", "{:.3g} evals/s"),
    ("Batched gradient throughput", "gradient_per_second", "{:.3g} evals/s"),
    ("Total time", "total_seconds", "{:.4g} s"),
    ("Sampling time", "sampling_seconds", "{:.4g} s"),
    ("Mean leapfrog steps per draw", "mean_leapfrog_steps", "{:.3g}"),
    ("Max leapfrog steps per draw", "max_leapfrog_steps", "{}"),
    ("Divergent draws", "divergent_fraction", "{:.2%}"),
    ("Max split R-hat", "max_rhat", "{:.4f}"),
    (
        "Max integrated autocorrelation time",
        "max_autocorr_time",
        "{:.3g} steps",
    ),
    ("Min ESS (bulk/tail)", "min_ess", "{:.0f}"),
    ("Min ESS per second of sampling", "ess_per_second", "{:.3g}"),
)


def format_table(results: list[dict[str, Any]], settings: Settings) -> str:
    labels = ["panco2 (CPU-only)", "panco3 (CPU-only)", "panco3 (GPU)"]
    lines = [
        "| Metric | " + " | ".join(labels) + " |",
        "|---|---|---|---|",
    ]
    for label, key, fmt in ROWS:
        cells = [value(result, key, fmt) for result in results]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Each column runs {settings.n_samplers} samplers for "
        f"{settings.warmup} warmup steps (emcee burn-in, discarded; NUTS "
        f"window adaptation) and then {settings.steps} sampling steps. "
        f"panco2 uses {settings.workers} CPU worker processes. panco3 CPU "
        f"splits its chains evenly over {settings.workers} XLA CPU devices "
        "(pmap over devices, vmap over each device's chains); panco3 GPU "
        f"runs up to {settings.gpu_chain_batch_size} chains per call on one "
        "GPU. A step costs one posterior evaluation per emcee walker but a "
        "whole NUTS trajectory per chain: the leapfrog rows count the "
        "gradient evaluations per NUTS draw.",
        "",
        "Total time runs from the start of warmup to the end of sampling and "
        "includes worker start-up and all JAX compilation. Sampling time "
        "covers the post-warmup steps only; the panco3 sampling kernel is "
        "compiled before it starts.",
        "",
        "Convergence metrics use the sampling draws only, in natural "
        "parameter space, and report the worst parameter: ArviZ "
        "rank-normalized split R-hat and bulk/tail ESS, and emcee's "
        "integrated autocorrelation time (marked unreliable when the chain "
        "is shorter than 50 times the estimate).",
        "",
        "Serial timing evaluates one point per call. Batched throughput "
        f"evaluates {settings.throughput_batch} points per call: a vmap for "
        "panco3, and a map over the worker pool for panco2 (the way emcee "
        "evaluates walkers). The gradient row is the vmapped value and "
        "gradient that NUTS uses; panco2 has no gradients. Both exclude "
        "JAX compilation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parser().parse_args()
    if args.worker:
        if args.settings_json is None or args.result_json is None:
            raise ValueError(
                "worker mode requires hidden settings/result paths"
            )
        worker(args)
        return
    settings = settings_from_args(args)
    if (
        not Path(settings.map_file).is_file()
        or not Path(settings.tf_file).is_file()
    ):
        raise FileNotFoundError(
            "The default panco2 validation map/TF was not found."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    settings_path = args.output.with_suffix(".settings.json")
    settings_path.write_text(
        json.dumps(asdict(settings), indent=2, sort_keys=True)
    )
    cases = ["panco2", "panco3-cpu"]
    if not args.skip_gpu:
        cases.append("panco3-gpu")
    results = []
    for case in cases:
        print(f"Running {case}...", flush=True)
        result_path = args.output.with_name(f".{case}.json")
        result = run_worker(
            case, settings, settings_path, result_path, args.gpu_platform
        )
        results.append(result)
    if args.skip_gpu:
        results.append({"implementation": "panco3-gpu", "unavailable": True})
    report = format_table(results, settings)
    args.output.write_text(report)
    args.output.with_suffix(".json").write_text(
        json.dumps(results, indent=2, sort_keys=True)
    )
    print(report, end="")
    print(f"Wrote {args.output} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()

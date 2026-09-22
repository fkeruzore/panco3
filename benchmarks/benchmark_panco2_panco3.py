#!/usr/bin/env python3
"""Benchmark matched panco2 and panco3 NIKA2 fits.

The default problem is panco2's C2/NIKA2 validation map.  Each implementation
uses the same map crop, five pressure bins, 18-arcsec beam, NIKA2 transfer
function, log-uniform pressure priors, normal calibration/zero priors, and
the same number of concurrent samplers.

The samplers are different (emcee StretchMove versus BlackJAX NUTS), so a
"step" means one proposed draw *per sampler*: one emcee walker iteration or
one post-warmup NUTS draw.  Both runs stop at the same ArviZ diagnostics:
maximum rank-normalized split R-hat and minimum bulk/tail ESS.  NUTS warmup
is counted in the panco3 step total and in time to convergence.

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
    chain_batch_size: int
    warmup: int
    check_every: int
    max_steps: int
    min_ess: int
    max_rhat: float
    posterior_repeats: int
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
        help="panco2 CPU worker processes (default: min(8, CPU count)).",
    )
    p.add_argument(
        "--chain-batch-size",
        type=int,
        default=None,
        help=(
            "panco3 chains evaluated concurrently; defaults to --workers. "
            "All chains still contribute to convergence diagnostics."
        ),
    )
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--check-every", type=int, default=250)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--min-ess", type=int, default=400)
    p.add_argument("--max-rhat", type=float, default=1.01)
    p.add_argument("--posterior-repeats", type=int, default=20)
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
    if args.check_every < 4:
        raise ValueError("--check-every must be at least 4.")
    chain_batch_size = args.chain_batch_size or args.workers
    if chain_batch_size < 1:
        raise ValueError("--chain-batch-size must be positive.")
    return Settings(
        map_file=str(args.map_file.resolve()),
        tf_file=str(args.tf_file.resolve()),
        map_size=args.map_size,
        n_bins=args.n_bins,
        n_nodes=args.n_nodes,
        n_samplers=args.n_samplers,
        workers=args.workers,
        chain_batch_size=chain_batch_size,
        warmup=args.warmup,
        check_every=args.check_every,
        max_steps=args.max_steps,
        min_ess=args.min_ess,
        max_rhat=args.max_rhat,
        posterior_repeats=args.posterior_repeats,
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


def diagnostics(draws: np.ndarray, settings: Settings) -> dict[str, Any]:
    """Return the common convergence diagnostic for (chain, draw, param)."""
    import arviz as az

    if draws.shape[1] < 4:
        return {"converged": False, "max_rhat": None, "min_ess": None}
    rhat = np.asarray(
        [
            az.rhat(draws[..., i], method="rank")
            for i in range(draws.shape[-1])
        ],
        dtype=float,
    )
    ess_bulk = np.asarray(
        [az.ess(draws[..., i], method="bulk") for i in range(draws.shape[-1])],
        dtype=float,
    )
    ess_tail = np.asarray(
        [az.ess(draws[..., i], method="tail") for i in range(draws.shape[-1])],
        dtype=float,
    )
    max_rhat = float(np.nanmax(rhat))
    min_ess = float(np.nanmin(np.minimum(ess_bulk, ess_tail)))
    return {
        "converged": bool(
            max_rhat <= settings.max_rhat and min_ess >= settings.min_ess
        ),
        "max_rhat": max_rhat,
        "min_ess": min_ess,
    }


def show_diagnostics(bar: tqdm, checked: dict[str, Any]) -> None:
    """Display the latest convergence diagnostics on a progress bar."""
    if checked["max_rhat"] is not None:
        bar.set_postfix(
            rhat=f"{checked['max_rhat']:.3f}", ess=f"{checked['min_ess']:.0f}"
        )


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


def run_panco2(settings: Settings) -> dict[str, Any]:
    """Run emcee in batches, using the shared ArviZ stopping criterion."""
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

    from panco2.panco2 import log_post

    def log_probability(theta: np.ndarray) -> float:
        return log_post(theta, ppf._log_lhood, ppf.model.log_prior)[0]

    rng = np.random.default_rng(settings.seed)
    # Starts lie well inside the common prior support.  The same relative
    # spread is converted to panco3's unconstrained coordinates below.
    center = np.concatenate((np.asarray(pressure), [-12.0, 0.0]))
    scale = np.concatenate((0.10 * np.asarray(pressure), [0.09, 1e-6]))
    starts = (
        center + rng.normal(size=(settings.n_samplers, center.size)) * scale
    )
    starts[:, : settings.n_bins] = np.clip(
        starts[:, : settings.n_bins], 0.011 * pressure, 99.0 * pressure
    )
    posterior_seconds = measure_panco2_posterior(
        log_probability, starts, settings.posterior_repeats
    )

    # Import here to avoid creating worker processes while setup is timed.
    from multiprocessing import Pool

    started = time.perf_counter()
    checked: dict[str, Any] = {
        "converged": False,
        "max_rhat": None,
        "min_ess": None,
    }
    bar = tqdm(total=settings.max_steps, desc="panco2 sampling", unit="step")
    with Pool(processes=settings.workers) as pool:
        sampler = emcee.EnsembleSampler(
            settings.n_samplers,
            center.size,
            log_post,
            pool=pool,
            args=[ppf._log_lhood, ppf.model.log_prior],
        )
        state = starts
        for _completed in range(
            settings.check_every, settings.max_steps + 1, settings.check_every
        ):
            for step in sampler.sample(state, iterations=settings.check_every):
                state = step
                bar.update(1)
            # emcee stores draws as (draw, walker, parameter).
            draws = np.swapaxes(sampler.get_chain(), 0, 1)
            checked = diagnostics(draws, settings)
            show_diagnostics(bar, checked)
            if checked["converged"]:
                break
    bar.close()
    elapsed = time.perf_counter() - started
    draws = np.swapaxes(sampler.get_chain(), 0, 1)
    return {
        "implementation": "panco2 (CPU-only)",
        "device": "cpu",
        "converged": checked["converged"],
        "steps_per_sampler": int(draws.shape[1]),
        "posterior_seconds": posterior_seconds,
        "time_to_convergence_seconds": elapsed,
        "max_rhat": checked["max_rhat"],
        "min_ess": checked["min_ess"],
        "n_samplers": settings.n_samplers,
    }


def run_panco3(settings: Settings, expected_device: str) -> dict[str, Any]:
    """Run adapted NUTS in chunks, checking common convergence each chunk."""
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
    natural = np.concatenate((pressure, [-12.0, 0.0]))
    scale = np.concatenate((0.10 * pressure, [0.09, 1e-6]))
    theta0 = (
        natural + rng.normal(size=(settings.n_samplers, natural.size)) * scale
    )
    theta0[:, : settings.n_bins] = np.clip(
        theta0[:, : settings.n_bins], 0.011 * pressure, 99.0 * pressure
    )
    span = np.log(100.0 / 0.01)
    frac = (
        np.log(theta0[:, : settings.n_bins]) - np.log(0.01 * pressure)
    ) / span
    z_press = np.log(frac / (1.0 - frac))
    z_other = np.column_stack(
        (
            (theta0[:, settings.n_bins] + 12.0) / 0.9,
            theta0[:, settings.n_bins + 1] / 1e-5,
        )
    )
    z0 = jnp.asarray(np.column_stack((z_press, z_other)))

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
            key, kernel, state, settings.check_every
        )
        last_state = jax.tree.map(lambda leaf: leaf[-1], trajectory)
        return last_state, trajectory.position, infos

    # jit once so equal-sized chain batches reuse one compiled program.
    run_warmup = jax.jit(jax.vmap(warm_one))
    run_chunk = jax.jit(jax.vmap(draw_chunk))
    constrain_draws = jax.jit(jax.vmap(jax.vmap(constrain)))
    started = time.perf_counter()
    state_batches = []
    parameter_batches = []
    z_batches = []
    firsts = range(0, settings.n_samplers, settings.chain_batch_size)
    for first in tqdm(firsts, desc="panco3 warmup", unit="batch"):
        last = min(first + settings.chain_batch_size, settings.n_samplers)
        states, parameters = run_warmup(chain_keys[first:last], z0[first:last])
        jax.block_until_ready(states)
        state_batches.append(states)
        parameter_batches.append(parameters)
        z_batches.append((first, last))
    # The first block both compiles and executes NUTS; compilation is included
    # in time-to-convergence because it is paid by a real end-to-end run.
    draws: list[np.ndarray] = []
    checked: dict[str, Any] = {
        "converged": False,
        "max_rhat": None,
        "min_ess": None,
    }
    # One unit is one draw for one chain batch, so the bar moves after every
    # batch rather than only once all batches have finished a chunk.
    bar = tqdm(
        total=settings.max_steps * len(z_batches),
        desc="panco3 sampling",
        unit="draw",
    )
    for _completed in range(
        settings.check_every, settings.max_steps + 1, settings.check_every
    ):
        sample_key, chunk_key = jax.random.split(sample_key)
        keys = jax.random.split(chunk_key, settings.n_samplers)
        draw_batches = []
        for batch, (first, last) in enumerate(z_batches):
            states, positions, infos = run_chunk(
                keys[first:last],
                state_batches[batch],
                parameter_batches[batch],
            )
            state_batches[batch] = states
            draw_batches.append(np.asarray(constrain_draws(positions)))
            bar.update(settings.check_every)
        draws.append(np.concatenate(draw_batches, axis=0))
        samples = np.concatenate(draws, axis=1)
        checked = diagnostics(samples, settings)
        show_diagnostics(bar, checked)
        if checked["converged"]:
            break
    bar.close()
    jax.block_until_ready(infos.acceptance_rate)
    elapsed = time.perf_counter() - started
    samples = np.concatenate(draws, axis=1)
    return {
        "implementation": f"panco3 ({device})",
        "device": device,
        "converged": checked["converged"],
        "steps_per_sampler": settings.warmup + int(samples.shape[1]),
        "posterior_seconds": posterior_seconds,
        "time_to_convergence_seconds": elapsed,
        "max_rhat": checked["max_rhat"],
        "min_ess": checked["min_ess"],
        "n_samplers": settings.n_samplers,
        "chain_batch_size": settings.chain_batch_size,
    }


def worker(args: argparse.Namespace) -> None:
    settings = Settings(**json.loads(args.settings_json.read_text()))
    if args.worker == "panco2":
        result = run_panco2(settings)
    elif args.worker == "panco3-cpu":
        result = run_panco3(settings, "cpu")
    else:
        result = run_panco3(settings, "gpu")
    args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True))


def run_worker(
    case: str, settings_path: Path, result_path: Path, gpu_platform: str | None
) -> dict[str, Any]:
    env = os.environ.copy()
    if case == "panco3-cpu":
        env["JAX_PLATFORMS"] = "cpu"
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


def value(result: dict[str, Any], key: str, unit: str = "") -> str:
    if result.get("unavailable"):
        return "unavailable"
    item = result[key]
    if isinstance(item, float):
        item = f"{item:.3g}"
    if not result["converged"]:
        return f"not converged ({item}{unit})"
    return f"{item}{unit}"


def format_table(results: list[dict[str, Any]], settings: Settings) -> str:
    labels = ["panco2 (CPU-only)", "panco3 (CPU-only)", "panco3 (GPU)"]
    cells = []
    for result in results:
        cells.append(
            (
                value(result, "steps_per_sampler"),
                value(result, "posterior_seconds", " s"),
                value(result, "time_to_convergence_seconds", " s"),
            )
        )
    lines = [
        "| Metric | " + " | ".join(labels) + " |",
        "|---|---|---|---|",
        "| Number of proposed steps to convergence | "
        + " | ".join(x[0] for x in cells)
        + " |",
        "| Time per posterior evaluation | "
        + " | ".join(x[1] for x in cells)
        + " |",
        "| Time to convergence | " + " | ".join(x[2] for x in cells) + " |",
        "",
        "Convergence requires max rank-normalized split R-hat "
        f"<= {settings.max_rhat} and min(bulk ESS, tail ESS) "
        f">= {settings.min_ess}. "
        f"Each column uses {settings.n_samplers} samplers; panco2 uses "
        f"{settings.workers} CPU worker processes. panco3 NUTS warmup "
        f"({settings.warmup} steps) is included in its step and "
        "convergence-time totals.",
        f"panco3 evaluates up to {settings.chain_batch_size} chains at once.",
        "",
        "Posterior timing is steady-state and excludes JAX compilation; "
        "time to convergence includes sampler initialization, adaptation, "
        "and JAX compilation.",
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
            case, settings_path, result_path, args.gpu_platform
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

#!/usr/bin/env python3
"""
NeuralGCM timing for one init — **recommended: delta method** (encode cancels out).

Why not time a full 240 h unroll on one GPU?
  Long JAX scans can OOM or dominate compile/memory; for **planning** you only need the marginal
  unroll cost per native step (or per 6 h wall-clock segment).

**Delta method** (default): measure ``T(lead_a)`` and ``T(lead_b)`` where each run is
``encode + unroll`` to that horizon. Then ``T(b) - T(a)`` **does not include encode** (it appears in
both). Dividing by the extra native steps gives mean time per step; multiply by step count for
``--extrap-lead-hours`` (e.g. 240 h). Optionally add **one** separate encode timing for a full
forecast estimate: ``encode_once + unroll_extrap``.

This checkpoint uses **1 h** native dt unless your pickle differs: 6 h lead ⇒ 6 unroll steps,
12 h lead ⇒ 12 steps, so ``T(12h)-T(6h)`` ≈ cost of **6** native steps (= one 6 h wall-clock block
when dt=1 h). Same 10-day length as other models' ``40×6 h`` is **240 h** wall-clock → **240**
native steps here, or **40** blocks of 6 h.

The repository efficiency CSV row for NeuralGCM timed ``encode+unroll`` **inside every repeat** and
used a short lead — misleading vs production (encode once, then long unroll).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _apply_gpu_benchmark_runtime_defaults() -> None:
    extra_xla = "--xla_gpu_enable_triton_gemm=false"
    prev = os.environ.get("XLA_FLAGS", "").strip()
    if extra_xla not in prev:
        os.environ["XLA_FLAGS"] = f"{prev} {extra_xla}".strip()
    os.environ.setdefault("CUDNN_FRONTEND_HEURISTIC_ENABLED", "0")


_apply_gpu_benchmark_runtime_defaults()


def _steps_for_lead_hours(lead_hours: int, dt_hours: float) -> int:
    n_steps_f = lead_hours / dt_hours
    n = int(round(n_steps_f))
    if n <= 0 or not np.isclose(n_steps_f, n):
        raise ValueError(
            f"lead_hours={lead_hours} incompatible with model dt={dt_hours}h "
            f"(need integer number of native steps)"
        )
    return n


def main() -> None:
    import jax

    from src.models.neuralgcm_runner import _build_regridded_init_dataset, _load_model_cached

    ap = argparse.ArgumentParser(
        description="NeuralGCM timing: delta (12h−6h) extrapolation, or full single-lead run"
    )
    ap.add_argument("--weights-root", type=Path, default=None)
    ap.add_argument("--era5-root", type=Path, default=None)
    ap.add_argument("--init-time", type=str, default="2025060100", help="YYYYMMDDHH")
    ap.add_argument(
        "--method",
        choices=["delta", "full"],
        default="delta",
        help="delta: T(lead_high)-T(lead_low) for marginal unroll (default). full: one lead-hours run.",
    )
    ap.add_argument(
        "--delta-low-hours",
        type=int,
        default=6,
        help="Shorter horizon (wall-clock hours) for delta pair.",
    )
    ap.add_argument(
        "--delta-high-hours",
        type=int,
        default=12,
        help="Longer horizon (wall-clock hours); must be > delta-low.",
    )
    ap.add_argument(
        "--extrap-lead-hours",
        type=int,
        default=240,
        help="Extrapolate unroll cost to this horizon (e.g. 240 = 10 days).",
    )
    ap.add_argument(
        "--regression-lead-hours",
        type=str,
        default=None,
        metavar="H,H,...",
        help=(
            "Optional comma-separated leads (e.g. 6,12,18,24). Measures T(lead)=encode+unroll for each, "
            "then fits T ≈ intercept + slope·n_native_steps (slope = ms per step, includes in-step decode). "
            "Also reports equal-width block deltas; algebra: (T18−T6)−(T12−T6) = T18−T12."
        ),
    )
    ap.add_argument(
        "--lead-hours",
        type=int,
        default=240,
        help="[method=full] Total forecast hours for a single encode+unroll timing.",
    )
    ap.add_argument(
        "--chunk-steps",
        type=int,
        default=None,
        metavar="N",
        help=(
            "[method=full] Split unroll into chunks (same physics); use if long scan OOMs."
        ),
    )
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    if args.weights_root is None:
        args.weights_root = Path(
            os.environ.get("NWP_WEIGHTS_ROOT", "/ecmwf-era5-datasets/nwp_bench/assets/weights")
        )
    if args.era5_root is None:
        from src.common.data_reader import DEFAULT_ERA5_NPY_ROOT

        args.era5_root = DEFAULT_ERA5_NPY_ROOT

    if args.delta_high_hours <= args.delta_low_hours:
        raise SystemExit("--delta-high-hours must be > --delta-low-hours")

    init_time = datetime.strptime(args.init_time, "%Y%m%d%H")
    model = _load_model_cached(args.weights_root)
    dt_hours = float(model.timestep / np.timedelta64(1, "h"))

    ds_input = _build_regridded_init_dataset(init_time, args.era5_root, model)
    t0_np = np.datetime64(pd.Timestamp(init_time))
    ds_init = ds_input.sel(time=t0_np, method="nearest")
    inputs = model.inputs_from_xarray(ds_init)
    input_forcings = model.forcings_from_xarray(ds_init)
    all_forcings = model.forcings_from_xarray(
        ds_input.sel(time=[t0_np], method="nearest")
    )
    rng_key = jax.random.key(42)

    def encode_once():
        return model.encode(inputs, input_forcings, rng_key)

    def time_encode_ms() -> float:
        t0 = time.perf_counter()
        enc = encode_once()
        jax.block_until_ready(enc)
        return (time.perf_counter() - t0) * 1000.0

    def unroll_steps(encoded_state, n_steps: int):
        chunk = args.chunk_steps
        if chunk is None or chunk <= 0 or chunk >= n_steps:
            _, predictions = model.unroll(
                encoded_state,
                all_forcings,
                steps=n_steps,
                timedelta=model.timestep,
                start_with_input=False,
            )
            jax.block_until_ready(predictions)
            return
        state = encoded_state
        predictions = None
        remaining = n_steps
        while remaining > 0:
            n = min(chunk, remaining)
            state, predictions = model.unroll(
                state,
                all_forcings,
                steps=n,
                timedelta=model.timestep,
                start_with_input=False,
            )
            remaining -= n
        assert predictions is not None
        jax.block_until_ready(predictions)

    def time_encode_plus_unroll_ms(n_steps: int) -> float:
        t0 = time.perf_counter()
        enc = encode_once()
        jax.block_until_ready(enc)
        unroll_steps(enc, n_steps)
        return (time.perf_counter() - t0) * 1000.0

    report: dict = {
        "init_time": args.init_time,
        "model_dt_hours": dt_hours,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "method": args.method,
    }

    if args.method == "delta":
        n_lo = _steps_for_lead_hours(args.delta_low_hours, dt_hours)
        n_hi = _steps_for_lead_hours(args.delta_high_hours, dt_hours)
        extra_steps = n_hi - n_lo
        if extra_steps <= 0:
            raise SystemExit("delta pair yields no extra native steps (check leads vs dt)")

        logger.info(
            "Delta: %dh -> %d steps, %dh -> %d steps (extra %d native steps)",
            args.delta_low_hours,
            n_lo,
            args.delta_high_hours,
            n_hi,
            extra_steps,
        )

        for _ in range(args.warmup):
            time_encode_plus_unroll_ms(n_lo)
            time_encode_plus_unroll_ms(n_hi)

        deltas_ms: list[float] = []
        short_ms: list[float] = []
        long_ms: list[float] = []
        for _ in range(args.repeats):
            t_s = time_encode_plus_unroll_ms(n_lo)
            t_l = time_encode_plus_unroll_ms(n_hi)
            short_ms.append(t_s)
            long_ms.append(t_l)
            deltas_ms.append(t_l - t_s)

        encode_only_ms = [time_encode_ms() for _ in range(args.repeats)]

        def mean_std(samples: list[float]) -> tuple[float, float]:
            if len(samples) == 1:
                return samples[0], 0.0
            return float(statistics.mean(samples)), float(statistics.stdev(samples))

        d_m, d_s = mean_std(deltas_ms)
        enc_m, enc_s = mean_std(encode_only_ms)
        ms_per_native_step = d_m / extra_steps
        n_extrap = _steps_for_lead_hours(args.extrap_lead_hours, dt_hours)
        unroll_extrap_ms = ms_per_native_step * n_extrap
        total_with_single_encode_ms = enc_m + unroll_extrap_ms

        upd = {
            "delta_low_hours": args.delta_low_hours,
            "delta_high_hours": args.delta_high_hours,
            "delta_native_steps": extra_steps,
            "delta_encode_plus_unroll_ms_mean_short": mean_std(short_ms)[0],
            "delta_encode_plus_unroll_ms_mean_long": mean_std(long_ms)[0],
            "delta_T_long_minus_T_short_ms_mean": d_m,
            "delta_T_long_minus_T_short_ms_std": d_s,
            "encode_only_ms_mean": enc_m,
            "encode_only_ms_std": enc_s,
            "ms_per_native_timestep_from_delta": ms_per_native_step,
            "extrap_lead_hours": args.extrap_lead_hours,
            "extrap_native_steps": n_extrap,
            "extrap_unroll_ms_from_linear_model": unroll_extrap_ms,
            "extrap_total_ms_encode_once_plus_unroll": total_with_single_encode_ms,
            "wall_clock_6h_blocks_in_extrap": args.extrap_lead_hours // 6
            if args.extrap_lead_hours % 6 == 0
            else None,
            "note": (
                "T(long)-T(short) removes one copy of encode from the marginal unroll cost. "
                "extrap_total uses one encode timing + linear unroll extrapolation. "
                "If decode is folded into each advance(), marginal deltas already include it; "
                "use --regression-lead-hours for slope/block diagnostics."
            ),
        }

        if args.regression_lead_hours:
            raw_leads = [
                int(x.strip())
                for x in args.regression_lead_hours.split(",")
                if x.strip()
            ]
            leads = sorted(set(raw_leads))
            if len(leads) < 2:
                raise SystemExit("--regression-lead-hours needs at least two distinct leads")
            lead_to_steps = {lh: _steps_for_lead_hours(lh, dt_hours) for lh in leads}
            for _ in range(args.warmup):
                for lh in leads:
                    time_encode_plus_unroll_ms(lead_to_steps[lh])

            slopes: list[float] = []
            intercepts: list[float] = []
            # Equal-spacing block deltas: consecutive leads (same wall-clock span between steps)
            block_labels: list[str] = []
            block_delta_batches: list[list[float]] = []

            ns_sorted = [lead_to_steps[lh] for lh in leads]
            if len(set(ns_sorted[i + 1] - ns_sorted[i] for i in range(len(leads) - 1))) == 1:
                step_span = ns_sorted[1] - ns_sorted[0]
                for i in range(len(leads) - 1):
                    block_labels.append(f"n{ns_sorted[i]}_to_n{ns_sorted[i + 1]}")
                    block_delta_batches.append([])

            nested_formula_ms: list[float] = []

            for _ in range(args.repeats):
                t_by_n: dict[int, float] = {}
                for lh in leads:
                    n_st = lead_to_steps[lh]
                    t_by_n[n_st] = time_encode_plus_unroll_ms(n_st)
                n_arr = np.array([lead_to_steps[lh] for lh in leads], dtype=np.float64)
                t_arr = np.array([t_by_n[int(x)] for x in n_arr], dtype=np.float64)
                slope_i, intercept_i = np.polyfit(n_arr, t_arr, 1)
                slopes.append(float(slope_i))
                intercepts.append(float(intercept_i))

                if block_delta_batches:
                    for bi in range(len(leads) - 1):
                        na, nb = ns_sorted[bi], ns_sorted[bi + 1]
                        block_delta_batches[bi].append(t_by_n[nb] - t_by_n[na])

                # (T18−T6) − (T12−T6) = T18−T12 when those lead hours exist
                if 18 in leads and 12 in leads and 6 in leads:
                    n6, n12, n18 = lead_to_steps[6], lead_to_steps[12], lead_to_steps[18]
                    nested_formula_ms.append(
                        (t_by_n[n18] - t_by_n[n6]) - (t_by_n[n12] - t_by_n[n6])
                    )

            slope_m, slope_s = mean_std(slopes)
            icept_m, icept_s = mean_std(intercepts)
            unroll_from_slope = slope_m * n_extrap
            total_from_slope = enc_m + unroll_from_slope

            upd["regression_lead_hours"] = leads
            upd["regression_ms_per_native_step_slope_mean"] = slope_m
            upd["regression_ms_per_native_step_slope_std"] = slope_s
            upd["regression_intercept_ms_mean"] = icept_m
            upd["regression_intercept_ms_std"] = icept_s
            upd["extrap_unroll_ms_from_regression_slope"] = unroll_from_slope
            upd["extrap_total_ms_encode_once_plus_unroll_regression"] = total_from_slope

            if block_labels:
                upd["regression_equal_native_span_blocks"] = {
                    block_labels[i]: {
                        "delta_ms_mean": mean_std(block_delta_batches[i])[0],
                        "delta_ms_std": mean_std(block_delta_batches[i])[1],
                        "native_step_span": ns_sorted[i + 1] - ns_sorted[i],
                    }
                    for i in range(len(block_labels))
                }

            if nested_formula_ms:
                nm, ns = mean_std(nested_formula_ms)
                upd["nested_T18_minus_T12_via_T18_T6_and_T12_T6_ms_mean"] = nm
                upd["nested_T18_minus_T12_via_T18_T6_and_T12_T6_ms_std"] = ns
                upd["nested_equals_T18_minus_T12_algebraically"] = True

        report.update(upd)

    else:
        max_steps = _steps_for_lead_hours(args.lead_hours, dt_hours)
        logger.info(
            "full: lead %d h -> %d native steps (chunk=%s)",
            args.lead_hours,
            max_steps,
            args.chunk_steps,
        )

        def full_once() -> None:
            enc = encode_once()
            unroll_steps(enc, max_steps)

        for _ in range(args.warmup):
            full_once()

        full_times: list[float] = []
        encode_times: list[float] = []
        unroll_times: list[float] = []

        for _ in range(args.repeats):
            t0 = time.perf_counter()
            enc = encode_once()
            jax.block_until_ready(enc)
            t1 = time.perf_counter()
            unroll_steps(enc, max_steps)
            t2 = time.perf_counter()
            encode_times.append((t1 - t0) * 1000.0)
            unroll_times.append((t2 - t1) * 1000.0)
            full_times.append((t2 - t0) * 1000.0)

        def ms_stats(samples: list[float]) -> tuple[float, float]:
            if len(samples) == 1:
                return samples[0], 0.0
            return float(statistics.mean(samples)), float(statistics.stdev(samples))

        f_m, f_s = ms_stats(full_times)
        e_m, e_s = ms_stats(encode_times)
        u_m, u_s = ms_stats(unroll_times)

        report.update(
            {
                "lead_hours": args.lead_hours,
                "unroll_steps": max_steps,
                "chunk_steps": args.chunk_steps,
                "encode_ms_mean": e_m,
                "encode_ms_std": e_s,
                "unroll_ms_mean": u_m,
                "unroll_ms_std": u_s,
                "full_encode_plus_unroll_ms_mean": f_m,
                "full_encode_plus_unroll_ms_std": f_s,
                "ms_per_native_step_amortized_unroll_only": u_m / max_steps,
            }
        )

    print(json.dumps(report, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()

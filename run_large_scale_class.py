#!/usr/bin/env python3
"""
Class-based large-scale driver for Pangu only.

This file is intentionally parallel to run_large_scale.py while keeping the
existing script untouched.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_large_scale import (  # noqa: E402
    _compute_weighted_metrics,
    _lat_721,
    _load_climatology_721,
    _lon_1440,
    iter_init_times,
    load_gt_subset_by_model,
    resolve_channel_subset,
    subset_channels,
    sufficient_era5_snapshot,
)
from src.common.data_reader import (  # noqa: E402
    DEFAULT_ERA5_NPY_ROOT,
    Era5NpyLayout,
    pangu_channel_names,
)
from src.common.saver import Saver  # noqa: E402
from src.models.pangu_runner_class import PanguForecastRunner  # noqa: E402

logger = logging.getLogger("run_large_scale_class")


@dataclass
class ModelAdapter:
    name: str
    run: Callable[..., Dict[int, np.ndarray]]
    channel_names: List[str]
    lat: np.ndarray
    load_gt: Callable[[datetime], np.ndarray]


def build_pangu_adapter(
    era5_root: Path,
    weights_root: Path | None,
) -> ModelAdapter:
    runner = PanguForecastRunner(era5_root=era5_root, weights_root=weights_root)

    # Keep `load_gt` contract compatibility with run_large_scale helper methods.
    def _dummy_gt(_t: datetime) -> np.ndarray:
        return np.zeros((len(pangu_channel_names()), 721, 1440), dtype=np.float32)

    return ModelAdapter(
        name="pangu",
        run=runner.run,
        channel_names=pangu_channel_names(),
        lat=_lat_721(),
        load_gt=_dummy_gt,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Class-based Pangu large-scale driver")
    parser.add_argument("--model", type=str, default="pangu", choices=["pangu"])
    parser.add_argument("--init_time", type=str, default=None, help="Single init time YYYYMMDDHH")
    parser.add_argument("--start", type=str, required=False, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=False, help="YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--init_hours",
        type=int,
        nargs="+",
        default=[0, 12],
        help="UTC hours for both daily initializations",
    )
    parser.add_argument(
        "--lead_times",
        type=int,
        nargs="+",
        required=True,
        help="Lead times in hours (6h-multiple for this class-based pangu path)",
    )
    parser.add_argument("--mode", choices=("online", "offline", "both"), default="online")
    parser.add_argument(
        "--era5_root",
        type=Path,
        default=DEFAULT_ERA5_NPY_ROOT,
        help="ERA5 np.25 root directory",
    )
    parser.add_argument(
        "--weights_root",
        type=Path,
        default=None,
        help="Optional root containing weights/pangu/*.onnx",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Online mode: append metrics here",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=5,
        help="Write CSV every N completed initialization times",
    )
    parser.add_argument(
        "--save_lead_range",
        type=int,
        nargs=2,
        default=None,
        metavar=("MIN_H", "MAX_H"),
        help="Offline: only save leads in this inclusive range (hours)",
    )
    parser.add_argument(
        "--save_vars",
        type=str,
        nargs="*",
        default=None,
        help="Offline: subset of channel short names (e.g. z_500 t_850)",
    )
    parser.add_argument("--nc_dir", type=Path, default=None, help="Offline NetCDF root")
    parser.add_argument(
        "--eval_vars",
        type=str,
        nargs="*",
        default=None,
        help="Online metrics: evaluate only this channel subset (default: all model channels)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.init_time:
        inits = [datetime.strptime(args.init_time, "%Y%m%d%H")]
    else:
        if not args.start or not args.end:
            raise ValueError("--start/--end are required when --init_time is not provided.")
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23, minute=59)
        inits = iter_init_times(start, end, args.init_hours)

    layout = Era5NpyLayout(args.era5_root)
    adapter = build_pangu_adapter(args.era5_root, args.weights_root)
    eval_idx, eval_names = resolve_channel_subset(
        adapter.channel_names, args.eval_vars, arg_name="eval_vars"
    )
    lead_times = sorted(set(args.lead_times))
    logger.info(
        "ClassDriver model=%s inits=%d leads=%s eval_vars=%d",
        args.model,
        len(inits),
        lead_times,
        len(eval_names),
    )

    rows: List[dict] = []
    done = 0
    out_csv = args.output_csv or (REPO_ROOT / "outputs" / "pangu_class" / "metrics.csv")

    saver: Optional[Saver] = None
    nc_root = args.nc_dir or (REPO_ROOT / "outputs" / "pangu_class" / "nc")
    if args.mode in ("offline", "both"):
        saver = Saver(str(nc_root))
        if args.save_lead_range is None:
            raise ValueError("offline/both mode requires --save_lead_range MIN MAX")

    t_job0 = time.perf_counter()
    for init_time in inits:
        t_init0 = time.perf_counter()
        if not sufficient_era5_snapshot(args.model, init_time, layout):
            logger.warning("Skip init %s (missing NPY).", init_time)
            continue

        try:
            t_infer0 = time.perf_counter()
            preds = adapter.run(init_time, lead_times)
            infer_s = time.perf_counter() - t_infer0
            logger.info("Init %s infer total %.3fs", init_time.strftime("%Y%m%d%H"), infer_s)
        except FileNotFoundError as e:
            logger.warning("Skip init %s: %s", init_time, e)
            continue

        init_str = init_time.strftime("%Y%m%d%H")
        save_nc_s = 0.0
        eval_s = 0.0

        for lead in lead_times:
            if lead not in preds:
                continue
            pred = preds[lead]
            valid_time = init_time + timedelta(hours=int(lead))

            if args.mode in ("offline", "both"):
                assert saver is not None and args.save_lead_range is not None
                lo, hi = args.save_lead_range
                if lo <= lead <= hi:
                    t_save0 = time.perf_counter()
                    sub, sub_names = subset_channels(pred, adapter.channel_names, args.save_vars)
                    saver.save(
                        data=sub,
                        channel_mapping=sub_names,
                        init_time_str=init_str,
                        lead_time_hours=int(lead),
                        lat_values=adapter.lat.astype(np.float64),
                        lon_values=_lon_1440(),
                    )
                    dt_save = time.perf_counter() - t_save0
                    save_nc_s += dt_save
                    logger.info("Init %s lead=%sh save took %.3fs", init_str, lead, dt_save)
                if args.mode == "offline":
                    continue

            if not adapter.channel_names:
                continue

            t_eval0 = time.perf_counter()
            pred_eval = pred[eval_idx, ...]
            try:
                gt_stack = load_gt_subset_by_model(
                    args.model,
                    valid_time,
                    eval_names,
                    era5_root=args.era5_root,
                    adapter=adapter,
                )
            except FileNotFoundError:
                logger.warning(
                    "No ERA5 analysis for valid %s (lead %sh) — skip metrics.",
                    valid_time,
                    lead,
                )
                continue

            if gt_stack.shape != pred_eval.shape:
                logger.error(
                    "Shape mismatch pred %s vs gt %s at lead %sh",
                    pred_eval.shape,
                    gt_stack.shape,
                    lead,
                )
                continue

            climatology_stack: np.ndarray | None = None
            try:
                climatology_stack = _load_climatology_721(
                    valid_time, eval_names, era5_root=args.era5_root
                )
            except Exception as e:
                logger.warning("Climatology load failed at %s: %s", valid_time, e)
                climatology_stack = None
            if climatology_stack is not None and climatology_stack.shape != pred_eval.shape:
                logger.warning(
                    "Climatology shape mismatch %s vs pred %s at lead %sh; ACC set NaN.",
                    climatology_stack.shape,
                    pred_eval.shape,
                    lead,
                )
                climatology_stack = None

            m = _compute_weighted_metrics(pred_eval, gt_stack, climatology_stack)
            for vi, var in enumerate(eval_names):
                rows.append(
                    {
                        "init_time": init_str,
                        "valid_time": valid_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "lead_hours": lead,
                        "variable": var,
                        "wrmse": float(m["wrmse"][vi]),
                        "bias": float(m["bias"][vi]),
                        "mae": float(m["mae"][vi]),
                        "activity": float(m["activity"][vi]),
                        "acc": float(m["acc"][vi]),
                    }
                )
            dt_eval = time.perf_counter() - t_eval0
            eval_s += dt_eval
            logger.info("Init %s lead=%sh eval took %.3fs", init_str, lead, dt_eval)

        done += 1
        logger.info(
            "Init %s done: total=%.3fs infer=%.3fs save=%.3fs eval=%.3fs",
            init_str,
            time.perf_counter() - t_init0,
            infer_s,
            save_nc_s,
            eval_s,
        )
        if args.mode in ("online", "both") and rows and done % args.flush_every == 0:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            logger.info("Flushed %d rows → %s", len(rows), out_csv)

    if args.mode in ("online", "both") and rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        logger.info("Wrote %d rows → %s", len(rows), out_csv)
    elif args.mode in ("online", "both"):
        logger.warning("No metrics rows produced (check data availability).")

    logger.info("ClassDriver total runtime %.3fs", time.perf_counter() - t_job0)


if __name__ == "__main__":
    main()


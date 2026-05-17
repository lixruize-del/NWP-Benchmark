#!/usr/bin/env python3
"""Step 3+4 for object-based heatwave verification (v2).

Input: p90 exceedance masks from extract_heatwave_events_v2.py
Output:
  - counts_tp_fp_fn.nc
  - metrics_grid.nc
  - metrics_global.json (cos-lat weighted)
  - metrics_latband.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.heatwave_object_common_v2 import match_temporal_iou_greedy_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate object-based heatwave metrics (v2).")
    p.add_argument("--masks-file", type=Path, required=True, help="hot_masks_p90.nc")
    p.add_argument("--event-type", type=str, choices=("heatwave", "coldwave"), default="heatwave")
    p.add_argument("--pred-mask-var", type=str, default="")
    p.add_argument("--gt-mask-var", type=str, default="")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--min-duration-days", type=int, default=3)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def _compute_metric_arrays(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    gt_total: np.ndarray,
    pred_total: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorized metric computation with strict NaN/0 edge semantics."""
    tp_f = tp.astype(np.float64)
    fp_f = fp.astype(np.float64)
    fn_f = fn.astype(np.float64)

    n = tp.shape[0]
    precision = np.full(n, np.nan, dtype=np.float64)
    recall = np.full(n, np.nan, dtype=np.float64)
    f1 = np.full(n, np.nan, dtype=np.float64)
    far = np.full(n, np.nan, dtype=np.float64)
    pod = np.full(n, np.nan, dtype=np.float64)
    csi = np.full(n, np.nan, dtype=np.float64)

    both0 = (gt_total == 0) & (pred_total == 0)
    no_gt = (gt_total == 0) & (pred_total > 0)
    no_pred = (gt_total > 0) & (pred_total == 0)
    general = (gt_total > 0) & (pred_total > 0)

    # Case2: no GT, has Pred
    precision[no_gt] = 0.0
    far[no_gt] = 1.0
    csi[no_gt] = 0.0

    # Case3: has GT, no Pred
    recall[no_pred] = 0.0
    pod[no_pred] = 0.0
    csi[no_pred] = 0.0

    # General case
    if np.any(general):
        g = np.where(general)[0]
        denom_pr = tp_f[g] + fp_f[g]
        denom_re = tp_f[g] + fn_f[g]
        denom_f1 = 2.0 * tp_f[g] + fp_f[g] + fn_f[g]
        denom_csi = tp_f[g] + fp_f[g] + fn_f[g]

        precision[g] = np.where(denom_pr > 0, tp_f[g] / denom_pr, np.nan)
        recall[g] = np.where(denom_re > 0, tp_f[g] / denom_re, np.nan)
        pod[g] = recall[g]
        far[g] = np.where(denom_pr > 0, fp_f[g] / denom_pr, np.nan)
        f1[g] = np.where(denom_f1 > 0, 2.0 * tp_f[g] / denom_f1, np.nan)
        csi[g] = np.where(denom_csi > 0, tp_f[g] / denom_csi, np.nan)

    # both0 remains all-NaN by initialization.
    _ = both0  # explicit readability

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": far,
        "pod": pod,
        "csi": csi,
    }


def _weighted_mean_2d(field: np.ndarray, lat: np.ndarray, lat_mask: np.ndarray | None = None) -> float:
    """Cos-lat weighted mean over valid finite cells."""
    w_lat = np.cos(np.deg2rad(lat.astype(np.float64)))
    w2d = w_lat[:, None] * np.ones(field.shape[1], dtype=np.float64)[None, :]
    valid = np.isfinite(field)
    if lat_mask is not None:
        valid = valid & lat_mask[:, None]
    if not np.any(valid):
        return float("nan")
    num = np.nansum(field[valid] * w2d[valid])
    den = np.nansum(w2d[valid])
    return float(num / den) if den > 0 else float("nan")


def main() -> None:
    args = parse_args()
    if args.min_duration_days <= 0:
        raise ValueError("min_duration_days must be positive")
    if not (0.0 <= args.iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be in [0, 1]")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts_nc = args.out_dir / "counts_tp_fp_fn.nc"
    metrics_nc = args.out_dir / "metrics_grid.nc"
    global_json = args.out_dir / "metrics_global.json"
    latband_csv = args.out_dir / "metrics_latband.csv"

    ds = xr.open_dataset(args.masks_file)
    try:
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        lat = ds[lat_name].values.astype(np.float64)
        lon = ds[lon_name].values.astype(np.float64)

        default_pred = "pred_mask_p90" if args.event_type == "heatwave" else "pred_mask_p10"
        default_gt = "gt_mask_p90" if args.event_type == "heatwave" else "gt_mask_p10"
        pred_var = args.pred_mask_var or default_pred
        gt_var = args.gt_mask_var or default_gt
        if gt_var not in ds.data_vars or pred_var not in ds.data_vars:
            raise ValueError(
                f"masks-file missing vars: need pred={pred_var}, gt={gt_var}, "
                f"available={list(ds.data_vars)}"
            )

        gt = np.asarray(ds[gt_var].values == 1, dtype=np.bool_)
        pred = np.asarray(ds[pred_var].values == 1, dtype=np.bool_)
    finally:
        ds.close()

    ntime, nlat, nlon = gt.shape
    gt2 = gt.reshape(ntime, nlat * nlon)
    pred2 = pred.reshape(ntime, nlat * nlon)

    tp, fp, fn, gt_total, pred_total = match_temporal_iou_greedy_batch(
        gt2,
        pred2,
        min_duration_days=int(args.min_duration_days),
        iou_threshold=float(args.iou_threshold),
    )
    metrics_1d = _compute_metric_arrays(tp, fp, fn, gt_total, pred_total)

    tp2 = tp.reshape(nlat, nlon).astype(np.int16)
    fp2 = fp.reshape(nlat, nlon).astype(np.int16)
    fn2 = fn.reshape(nlat, nlon).astype(np.int16)
    gt_tot2 = gt_total.reshape(nlat, nlon).astype(np.int16)
    pred_tot2 = pred_total.reshape(nlat, nlon).astype(np.int16)

    ds_counts = xr.Dataset(
        data_vars={
            "tp": (("latitude", "longitude"), tp2),
            "fp": (("latitude", "longitude"), fp2),
            "fn": (("latitude", "longitude"), fn2),
            "gt_total": (("latitude", "longitude"), gt_tot2),
            "pred_total": (("latitude", "longitude"), pred_tot2),
        },
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "iou_threshold": float(args.iou_threshold),
            "min_duration_days": int(args.min_duration_days),
            "description": "Per-gridpoint event counts from greedy Temporal-IoU matching",
        },
    )
    ds_counts.to_netcdf(counts_nc)

    ds_metrics = xr.Dataset(
        data_vars={
            k: (("latitude", "longitude"), v.reshape(nlat, nlon).astype(np.float32))
            for k, v in metrics_1d.items()
        },
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "description": "Per-gridpoint object-based heatwave metrics",
            "iou_threshold": float(args.iou_threshold),
            "min_duration_days": int(args.min_duration_days),
        },
    )
    ds_metrics.to_netcdf(metrics_nc)

    global_metrics = {}
    for k, arr in metrics_1d.items():
        global_metrics[k] = _weighted_mean_2d(arr.reshape(nlat, nlon), lat)

    global_payload = {
        "event_type": args.event_type,
        "iou_threshold": float(args.iou_threshold),
        "min_duration_days": int(args.min_duration_days),
        "metric_global_weighted_mean": global_metrics,
        "count_sums": {
            "tp_sum": int(tp.sum()),
            "fp_sum": int(fp.sum()),
            "fn_sum": int(fn.sum()),
            "gt_total_sum": int(gt_total.sum()),
            "pred_total_sum": int(pred_total.sum()),
        },
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with open(global_json, "w", encoding="utf-8") as f:
        json.dump(global_payload, f, ensure_ascii=True, indent=2)

    # Latitude-band summaries.
    bands = [
        ("tropics_30S_30N", (lat >= -30.0) & (lat <= 30.0)),
        ("north_midhigh_30N_90N", (lat > 30.0) & (lat <= 90.0)),
        ("south_midhigh_90S_30S", (lat >= -90.0) & (lat < -30.0)),
    ]
    rows = []
    for band_name, lat_mask in bands:
        row = {"lat_band": band_name}
        for k, arr in metrics_1d.items():
            row[k] = _weighted_mean_2d(arr.reshape(nlat, nlon), lat, lat_mask=lat_mask)
        rows.append(row)
    pd.DataFrame(rows).to_csv(latband_csv, index=False)

    print("Step3+4 generated:")
    print(f"  - counts: {counts_nc}")
    print(f"  - metrics_grid: {metrics_nc}")
    print(f"  - global: {global_json}")
    print(f"  - latband: {latband_csv}")


if __name__ == "__main__":
    main()


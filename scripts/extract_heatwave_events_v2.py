#!/usr/bin/env python3
"""Step 2 for object-based event verification (v2).

Supports:
  - heatwave: Tmax > P90
  - coldwave: Tmin < P10
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import xarray as xr
from netCDF4 import Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.heatwave_object_common_v2 import build_event_id_series_1d  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract event masks/events from pred/gt daily series and percentile baseline."
        )
    )
    p.add_argument("--event-type", type=str, choices=("heatwave", "coldwave"), default="heatwave")
    p.add_argument("--pred-file", type=Path, required=True, help="pred_tmax_daily.nc or pred_tmin_daily.nc")
    p.add_argument("--gt-file", type=Path, required=True, help="gt cache or gt_tmax/gt_tmin daily file")
    p.add_argument("--baseline-file", type=Path, required=True, help="p90/p10 baseline nc")
    p.add_argument("--baseline-var", type=str, default="", help="Optional baseline var name")
    p.add_argument("--pred-var", type=str, default="", help="Optional pred variable name")
    p.add_argument("--gt-var", type=str, default="", help="Optional gt variable name")
    p.add_argument("--min-duration-days", type=int, default=3)
    p.add_argument("--write-event-ids", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def _pick_var(ds: xr.Dataset, preferred: str, fallback_hint: str) -> str:
    if preferred:
        if preferred not in ds.data_vars:
            raise ValueError(f"Variable '{preferred}' not found in {list(ds.data_vars)}")
        return preferred
    for k in ds.data_vars:
        if fallback_hint in k.lower():
            return k
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise ValueError(f"Cannot pick variable from {list(ds.data_vars)}")


def _pick_baseline_var(ds: xr.Dataset, preferred: str, event_type: str) -> str:
    if preferred:
        if preferred not in ds.data_vars:
            raise ValueError(f"Baseline var '{preferred}' not found: {list(ds.data_vars)}")
        return preferred
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    hint = "p90" if event_type == "heatwave" else "p10"
    for k in ds.data_vars:
        if hint in k.lower():
            return k
    raise ValueError(f"Cannot auto-pick baseline var from {list(ds.data_vars)}")


def _ensure_mask_file(
    out_file: Path,
    time_vals: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    pred_mask_name: str,
    gt_mask_name: str,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(str(out_file), "w", format="NETCDF4") as ds:
        ds.createDimension("time", int(time_vals.shape[0]))
        ds.createDimension("latitude", int(lat_vals.shape[0]))
        ds.createDimension("longitude", int(lon_vals.shape[0]))

        tv = ds.createVariable("time", "f8", ("time",))
        tv[:] = time_vals.astype(np.float64)
        tv.units = "seconds since 1970-01-01 00:00:00"
        tv.calendar = "standard"

        la = ds.createVariable("latitude", "f8", ("latitude",))
        la[:] = lat_vals.astype(np.float64)
        lo = ds.createVariable("longitude", "f8", ("longitude",))
        lo[:] = lon_vals.astype(np.float64)

        v1 = ds.createVariable(
            pred_mask_name,
            "i1",
            ("time", "latitude", "longitude"),
            zlib=True,
            complevel=2,
            fill_value=-1,
        )
        v1.units = "0_or_1"
        v2 = ds.createVariable(
            gt_mask_name,
            "i1",
            ("time", "latitude", "longitude"),
            zlib=True,
            complevel=2,
            fill_value=-1,
        )
        v2.units = "0_or_1"


def _write_event_ids(
    masks_file: Path,
    out_file: Path,
    min_duration_days: int,
    pred_mask_name: str,
    gt_mask_name: str,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(str(masks_file), "r") as ds_in, Dataset(str(out_file), "w", format="NETCDF4") as ds_out:
        ntime = int(ds_in.dimensions["time"].size)
        nlat = int(ds_in.dimensions["latitude"].size)
        nlon = int(ds_in.dimensions["longitude"].size)
        ds_out.createDimension("time", ntime)
        ds_out.createDimension("latitude", nlat)
        ds_out.createDimension("longitude", nlon)

        tv = ds_out.createVariable("time", "f8", ("time",))
        tv[:] = ds_in.variables["time"][:]
        tv.units = getattr(ds_in.variables["time"], "units", "seconds since 1970-01-01 00:00:00")
        tv.calendar = getattr(ds_in.variables["time"], "calendar", "standard")
        la = ds_out.createVariable("latitude", "f8", ("latitude",))
        la[:] = ds_in.variables["latitude"][:]
        lo = ds_out.createVariable("longitude", "f8", ("longitude",))
        lo[:] = ds_in.variables["longitude"][:]

        pred_out = ds_out.createVariable(
            "pred_event_id",
            "u2",
            ("time", "latitude", "longitude"),
            zlib=True,
            complevel=2,
            fill_value=0,
        )
        gt_out = ds_out.createVariable(
            "gt_event_id",
            "u2",
            ("time", "latitude", "longitude"),
            zlib=True,
            complevel=2,
            fill_value=0,
        )

        pred_mask = ds_in.variables[pred_mask_name]
        gt_mask = ds_in.variables[gt_mask_name]

        for j in range(nlat):
            pred_row = np.asarray(pred_mask[:, j, :], dtype=np.int8)
            gt_row = np.asarray(gt_mask[:, j, :], dtype=np.int8)
            pred_ids_row = np.zeros((ntime, nlon), dtype=np.uint16)
            gt_ids_row = np.zeros((ntime, nlon), dtype=np.uint16)
            for i in range(nlon):
                p_ids = build_event_id_series_1d(pred_row[:, i] == 1, min_duration_days=min_duration_days)
                g_ids = build_event_id_series_1d(gt_row[:, i] == 1, min_duration_days=min_duration_days)
                if p_ids.size and int(np.nanmax(p_ids)) > 65535:
                    raise ValueError("pred event id exceeds uint16 capacity")
                if g_ids.size and int(np.nanmax(g_ids)) > 65535:
                    raise ValueError("gt event id exceeds uint16 capacity")
                pred_ids_row[:, i] = p_ids.astype(np.uint16)
                gt_ids_row[:, i] = g_ids.astype(np.uint16)
            pred_out[:, j, :] = pred_ids_row
            gt_out[:, j, :] = gt_ids_row
            if (j + 1) % 60 == 0 or (j + 1) == nlat:
                print(f"event id progress latitude_row={j+1}/{nlat}", flush=True)


def main() -> None:
    args = parse_args()
    if args.min_duration_days <= 0:
        raise ValueError("min_duration_days must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.event_type == "heatwave":
        masks_nc = args.out_dir / "hot_masks_p90.nc"
        events_nc = args.out_dir / "event_ids_p90.nc"
        pred_mask_name = "pred_mask_p90"
        gt_mask_name = "gt_mask_p90"
        pred_hint = "tmax"
        gt_hint = "tmax"
    else:
        masks_nc = args.out_dir / "cold_masks_p10.nc"
        events_nc = args.out_dir / "event_ids_p10.nc"
        pred_mask_name = "pred_mask_p10"
        gt_mask_name = "gt_mask_p10"
        pred_hint = "tmin"
        gt_hint = "tmin"
    meta_json = args.out_dir / "meta_step2.json"

    ds_pred = xr.open_dataset(args.pred_file)
    ds_gt = xr.open_dataset(args.gt_file)
    ds_b = xr.open_dataset(args.baseline_file)
    try:
        pred_var = _pick_var(ds_pred, args.pred_var, pred_hint)
        gt_var = _pick_var(ds_gt, args.gt_var, gt_hint)
        bvar = _pick_baseline_var(ds_b, args.baseline_var, args.event_type)

        pred = ds_pred[pred_var]
        gt = ds_gt[gt_var]
        b = ds_b[bvar]

        lat_name = "latitude" if "latitude" in pred.dims else "lat"
        lon_name = "longitude" if "longitude" in pred.dims else "lon"
        b_lat_name = "latitude" if "latitude" in b.dims else "lat"
        b_lon_name = "longitude" if "longitude" in b.dims else "lon"

        time_vals = pred["time"].values
        lat_vals = pred[lat_name].values.astype(np.float64)
        lon_vals = pred[lon_name].values.astype(np.float64)
        if gt.shape != pred.shape:
            raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
        if int(b[b_lat_name].shape[0]) != int(pred.shape[1]) or int(b[b_lon_name].shape[0]) != int(pred.shape[2]):
            raise ValueError(
                f"baseline grid mismatch baseline={(b[b_lat_name].shape[0], b[b_lon_name].shape[0])} "
                f"pred={(pred.shape[1], pred.shape[2])}"
            )

        doy_arr = ds_pred["time"].dt.dayofyear.values.astype(np.int32)
        b_doys = b["doy"].values.astype(np.int32)
        doy_to_idx = {int(d): i for i, d in enumerate(b_doys.tolist())}

        time_seconds = (
            (time_vals.astype("datetime64[s]") - np.datetime64("1970-01-01T00:00:00"))
            .astype("timedelta64[s]")
            .astype(np.int64)
        )
        _ensure_mask_file(masks_nc, time_seconds, lat_vals, lon_vals, pred_mask_name, gt_mask_name)

        missing_doy_count = 0
        with Dataset(str(masks_nc), "a", format="NETCDF4") as ds_out:
            pred_out = ds_out.variables[pred_mask_name]
            gt_out = ds_out.variables[gt_mask_name]
            ntime = int(pred.shape[0])
            for t in range(ntime):
                doy = int(doy_arr[t])
                if doy not in doy_to_idx:
                    missing_doy_count += 1
                    pred_out[t, :, :] = -1
                    gt_out[t, :, :] = -1
                    continue
                bi = doy_to_idx[doy]
                b2d = np.asarray(b.isel(doy=bi).values, dtype=np.float32)
                p2d = np.asarray(pred.isel(time=t).values, dtype=np.float32)
                g2d = np.asarray(gt.isel(time=t).values, dtype=np.float32)

                if args.event_type == "heatwave":
                    pm = (p2d > b2d).astype(np.int8)
                    gm = (g2d > b2d).astype(np.int8)
                else:
                    pm = (p2d < b2d).astype(np.int8)
                    gm = (g2d < b2d).astype(np.int8)
                pred_out[t, :, :] = pm
                gt_out[t, :, :] = gm
                if (t + 1) % 50 == 0 or (t + 1) == ntime:
                    print(f"mask progress time={t+1}/{ntime}", flush=True)

        if args.write_event_ids:
            _write_event_ids(
                masks_nc,
                events_nc,
                min_duration_days=int(args.min_duration_days),
                pred_mask_name=pred_mask_name,
                gt_mask_name=gt_mask_name,
            )

        meta = {
            "pred_file": str(args.pred_file),
            "gt_file": str(args.gt_file),
            "event_type": args.event_type,
            "baseline_file": str(args.baseline_file),
            "baseline_var": bvar,
            "pred_var": pred_var,
            "gt_var": gt_var,
            "min_duration_days": int(args.min_duration_days),
            "write_event_ids": bool(args.write_event_ids),
            "masks_file": str(masks_nc),
            "event_ids_file": str(events_nc) if args.write_event_ids else "",
            "missing_doy_count": int(missing_doy_count),
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)
    finally:
        ds_pred.close()
        ds_gt.close()
        ds_b.close()

    print("Step2 generated:")
    print(f"  - masks: {masks_nc}")
    if args.write_event_ids:
        print(f"  - event_ids: {events_nc}")
    print(f"  - meta: {meta_json}")


if __name__ == "__main__":
    main()


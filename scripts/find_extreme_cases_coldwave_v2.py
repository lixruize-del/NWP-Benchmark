#!/usr/bin/env python3
"""
Find top cold-surge spatiotemporal blobs in 2025 H2 (ERA5), analogous to find_extreme_cases_v2.py.

Criteria (GT side):
  - gt_mask_p10 == cold-wave object (Tmin below local P10 threshold from step2 pipeline)
  - ERA5 Tmin strictly below an absolute ceiling (default -5 °C), analogous to heatwave Tmax > 35 °C
  - Time extent of the labeled 3D component >= min_duration_days (default 3)

Uses the same 26-connectivity 3D labeling as the heatwave finder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import find_objects, label, sum as ndi_sum

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.common.repo_paths import nwp_outputs_dir  # noqa: E402

_HW = nwp_outputs_dir() / "era5_monthly_202506_v2/metrics/heatwave_object_v2/ifs"
DEFAULT_MASK = _HW / "aifs/lead_day_1/step2_coldwave/cold_masks_p10.nc"
DEFAULT_GT = _HW / "_shared/gt_tmax_daily_2025_20250701_20251231.nc"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank extreme cold-surge events (P10 mask + cold Tmin).")
    p.add_argument("--mask-file", type=Path, default=DEFAULT_MASK, help="cold_masks_p10.nc (GT mask channel).")
    p.add_argument("--gt-file", type=Path, default=DEFAULT_GT, help="GT daily cache with tmin_gt_c.")
    p.add_argument(
        "--tmin-ceiling",
        type=float,
        default=-5.0,
        help="Require ERA5 Tmin < this value (°C). Heat analogue uses Tmax > 35 °C.",
    )
    p.add_argument(
        "--min-duration-days",
        type=int,
        default=3,
        help="Minimum time span (days) of the 3D connected component along the time axis.",
    )
    p.add_argument("--top-n", type=int, default=20, help="How many events to print.")
    p.add_argument(
        "--lat-min",
        type=float,
        default=-75.0,
        help="Restrict labeling to grid rows with latitude >= this (excludes deep Antarctic artifacts).",
    )
    p.add_argument(
        "--lat-max",
        type=float,
        default=75.0,
        help="Restrict labeling to grid rows with latitude <= this.",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default="",
        help="Optional path to write ranked table as CSV.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading datasets...")
    ds_mask = xr.open_dataset(args.mask_file)
    ds_gt = xr.open_dataset(args.gt_file)

    lats = ds_gt["latitude"].values
    lons = ds_gt["longitude"].values
    times = ds_gt["time"].values

    gt_m = ds_mask["gt_mask_p10"].values
    mask_on = gt_m > 0.5

    tmin = ds_gt["tmin_gt_c"].values
    extreme_mask = mask_on & (tmin < args.tmin_ceiling)
    lat_band = (lats >= args.lat_min) & (lats <= args.lat_max)
    extreme_mask &= np.broadcast_to(lat_band[np.newaxis, :, np.newaxis], extreme_mask.shape)

    print(
        f"3D labeling: gt_mask_p10 & (Tmin < {args.tmin_ceiling} °C) & "
        f"lat in [{args.lat_min}, {args.lat_max}] "
        f"(min duration >= {args.min_duration_days} days)..."
    )
    structure = np.ones((3, 3, 3), dtype=int)
    labeled_mask, num_features = label(extreme_mask, structure=structure)

    lat_weights = np.cos(np.deg2rad(lats))
    weights_3d = np.broadcast_to(lat_weights[np.newaxis, :, np.newaxis], extreme_mask.shape)
    event_sizes = ndi_sum(
        weights_3d,
        labeled_mask,
        index=np.arange(1, num_features + 1),
    )

    slices = find_objects(labeled_mask)

    # Filter by duration; optional zero severity for short blobs
    valid = []
    for idx in range(num_features):
        sl = slices[idx]
        time_slice = sl[0]
        ndays = int(time_slice.stop - time_slice.start)
        if ndays >= args.min_duration_days:
            valid.append(idx)

    if not valid:
        print("No events passed duration filter.")
        ds_mask.close()
        ds_gt.close()
        return

    valid = np.array(valid, dtype=int)
    sizes_filtered = event_sizes[valid]
    order = np.argsort(sizes_filtered)[::-1][: args.top_n]
    top_indices = valid[order]

    print("\n" + "=" * 85)
    print(
        f"Top {min(len(top_indices), args.top_n)} Extreme Cold Events "
        f"(Tmin < {args.tmin_ceiling} °C & P10 mask) in 2025 H2"
    )
    print("=" * 85)

    rows: list[dict[str, object]] = []

    for i, idx in enumerate(top_indices):
        real_label = idx + 1
        sl = slices[idx]
        time_slice, lat_slice, lon_slice = sl

        start_date = pd.to_datetime(times[time_slice.start]).strftime("%Y-%m-%d")
        end_date = pd.to_datetime(times[time_slice.stop - 1]).strftime("%Y-%m-%d")
        ndays = int(time_slice.stop - time_slice.start)

        lat_vals_slice = lats[lat_slice]
        lon_vals_slice = lons[lon_slice]
        lat_range = (float(lat_vals_slice.min()), float(lat_vals_slice.max()))
        lon_range = (float(lon_vals_slice.min()), float(lon_vals_slice.max()))

        event_3d = labeled_mask[sl] == real_label
        daily_stats: list[tuple[int, float, float, float]] = []
        for t_idx in range(ndays):
            day_mask = event_3d[t_idx]
            if not day_mask.any():
                continue
            day_tmin = ds_gt["tmin_gt_c"].values[time_slice.start + t_idx, lat_slice, lon_slice]
            mean_peak = float(day_tmin[day_mask].mean())
            min_peak = float(day_tmin[day_mask].min())
            area_weight = float((day_mask * lat_weights[lat_slice, np.newaxis]).sum())
            daily_stats.append((t_idx, area_weight, mean_peak, min_peak))

        if not daily_stats:
            continue
        daily_stats.sort(key=lambda x: x[1], reverse=True)
        best_rel_t_idx, best_area, best_mean_t, best_min_t = daily_stats[0]
        peak_date = pd.to_datetime(times[time_slice.start + best_rel_t_idx]).strftime("%Y-%m-%d")

        severity = float(event_sizes[idx])

        print(f"Rank {i + 1:2d} | Sev: {severity:8.0f} | {start_date} to {end_date} ({ndays}d) | Peak: {peak_date}")
        print(f"        Box: Lat[{lat_range[0]:6.1f}, {lat_range[1]:6.1f}] Lon[{lon_range[0]:6.1f}, {lon_range[1]:6.1f}]")
        print(f"        Peak Stats: AreaWt={best_area:6.1f}, MeanTmin={best_mean_t:5.1f}°C, MinTmin={best_min_t:5.1f}°C")
        print("-" * 85)

        rows.append(
            {
                "rank": i + 1,
                "severity": severity,
                "start_date": start_date,
                "end_date": end_date,
                "duration_days": ndays,
                "peak_date": peak_date,
                "lat_min": lat_range[0],
                "lat_max": lat_range[1],
                "lon_min": lon_range[0],
                "lon_max": lon_range[1],
                "peak_mean_tmin_c": best_mean_t,
                "peak_min_tmin_c": best_min_t,
            }
        )

    if args.csv_out and rows:
        pd.DataFrame(rows).to_csv(args.csv_out, index=False)
        print(f"Wrote {args.csv_out}")

    ds_mask.close()
    ds_gt.close()


if __name__ == "__main__":
    main()

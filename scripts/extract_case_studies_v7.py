#!/usr/bin/env python3
"""V7 case studies: wide plot window vs heatwave span; bias = mean(pred)-mean(gt) over hw_dates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from src.common.repo_paths import data_dir, nwp_outputs_dir  # noqa: E402

_HW_METRICS = nwp_outputs_dir() / "era5_monthly_202506_v2/metrics/heatwave_object_v2"
GT_CACHE = _HW_METRICS / "ifs/_shared/gt_tmax_daily_2025_20250701_20251231.nc"
P90_FILE = data_dir() / "heatwave_baseline_p90_doy_001_366_fullrun_20260504_loadv1.nc"
MODELS_ROOT = _HW_METRICS / "ifs"
OUT_ROOT = MODELS_ROOT / "case_studies_v7"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MODELS = ["aifs", "aurora", "fuxi", "fengwu", "pangu", "graphcast", "stormer"]
LEAD_DAYS = [1, 3, 7, 10]

# plot_dates: wide context for column 2; hw_dates: heatwave core (grey shading only).
# GT cache covers 2025-07-01..2025-12-31 — East_Asia plot window starts 2025-07-01 (not June).
CASE_CFG = {
    "North_America": {
        "plot_dates": ("2025-08-02", "2025-08-14"),
        "hw_dates": ("2025-08-06", "2025-08-10"),
        "peak": "2025-08-08",
        "map_box": {"lat": (24.0, 44.0), "lon": (233.0, 263.0)},
        "analysis_box": {"lat": (29.0, 37.0), "lon": (238.5, 250.5)},
    },
    "East_Asia": {
        "plot_dates": ("2025-07-01", "2025-07-12"),
        "hw_dates": ("2025-07-02", "2025-07-06"),
        "peak": "2025-07-04",
        "map_box": {"lat": (25.0, 45.0), "lon": (108.0, 138.0)},
        "analysis_box": {"lat": (33.0, 41.0), "lon": (112.2, 124.2)},
    },
    "South_America": {
        "plot_dates": ("2025-11-02", "2025-11-14"),
        "hw_dates": ("2025-11-06", "2025-11-10"),
        "peak": "2025-11-08",
        "map_box": {"lat": (-30.0, -10.0), "lon": (285.0, 315.0)},
        "analysis_box": {"lat": (-27.5, -19.5), "lon": (290.5, 302.5)},
    },
    "Australia": {
        "plot_dates": ("2025-11-29", "2025-12-11"),
        "hw_dates": ("2025-12-03", "2025-12-07"),
        "peak": "2025-12-05",
        "map_box": {"lat": (-35.0, -15.0), "lon": (125.0, 155.0)},
        "analysis_box": {"lat": (-32.2, -24.2), "lon": (129.2, 141.2)},
    },
}


def weighted_area_mean(ds: xr.Dataset, var: str, lat_range: tuple[float, float], lon_range: tuple[float, float]) -> xr.DataArray:
    subset = ds.sel(latitude=slice(max(lat_range), min(lat_range)))
    subset = subset.sel(longitude=slice(lon_range[0], lon_range[1]))
    weights = np.cos(np.deg2rad(subset.latitude))
    weighted_sum = (subset[var] * weights).sum(dim=("latitude", "longitude"))
    total_weight = weights.sum(dim="latitude") * subset.longitude.size
    return weighted_sum / total_weight


def extract_map_data(gt_ds: xr.Dataset, p90_ds: xr.Dataset, case_name: str, cfg: dict) -> None:
    peak_dt = np.datetime64(cfg["peak"])
    m_lat = cfg["map_box"]["lat"]
    m_lon = cfg["map_box"]["lon"]
    gt_peak = gt_ds.sel(
        time=peak_dt,
        latitude=slice(max(m_lat), min(m_lat)),
        longitude=slice(m_lon[0], m_lon[1]),
    )
    doy = pd.to_datetime(cfg["peak"]).dayofyear
    p90_peak = p90_ds.sel(
        doy=doy,
        latitude=slice(max(m_lat), min(m_lat)),
        longitude=slice(m_lon[0], m_lon[1]),
    )
    out_ds = xr.Dataset(
        data_vars={"tmax_gt": gt_peak["tmax_gt_c"], "p90": p90_peak["t2m_p90_c"]},
        attrs={
            "case": case_name,
            "peak_day": cfg["peak"],
            "plot_dates": str(cfg["plot_dates"]),
            "hw_dates": str(cfg["hw_dates"]),
        },
    )
    out_ds.to_netcdf(OUT_ROOT / f"case_{case_name}_map_data.nc")


def extract_timeseries_data(gt_ds: xr.Dataset, case_name: str, cfg: dict) -> None:
    start_dt, end_dt = (np.datetime64(cfg["plot_dates"][0]), np.datetime64(cfg["plot_dates"][1]))
    if start_dt > end_dt:
        raise ValueError(f"{case_name}: plot_dates reversed: {cfg['plot_dates']}")

    a_lat = cfg["analysis_box"]["lat"]
    a_lon = cfg["analysis_box"]["lon"]
    gt_ts = weighted_area_mean(gt_ds.sel(time=slice(start_dt, end_dt)), "tmax_gt_c", a_lat, a_lon)
    results: dict[str, object] = {"time": gt_ts.time.values, "era5": gt_ts.values}

    for model in MODELS:
        pred_path = MODELS_ROOT / model / "lead_day_3" / "step1" / "pred_tmax_daily.nc"
        if not pred_path.exists():
            results[model] = np.full(len(gt_ts), np.nan)
            continue
        with xr.open_dataset(pred_path) as ds_m:
            pred_ts = weighted_area_mean(ds_m.sel(time=slice(start_dt, end_dt)), "tmax_pred_c", a_lat, a_lon)
            results[model] = pred_ts.values

    pd.DataFrame(results).to_csv(OUT_ROOT / f"case_{case_name}_timeseries.csv", index=False)


def extract_bias_data(gt_ds: xr.Dataset, case_name: str, cfg: dict) -> None:
    hw_start, hw_end = (np.datetime64(cfg["hw_dates"][0]), np.datetime64(cfg["hw_dates"][1]))
    if hw_start > hw_end:
        raise ValueError(f"{case_name}: hw_dates reversed: {cfg['hw_dates']}")

    a_lat = cfg["analysis_box"]["lat"]
    a_lon = cfg["analysis_box"]["lon"]

    gt_ts_hw = weighted_area_mean(gt_ds.sel(time=slice(hw_start, hw_end)), "tmax_gt_c", a_lat, a_lon)
    gt_mean = float(gt_ts_hw.mean(dim="time").values.item())

    results: dict[str, object] = {"lead_day": LEAD_DAYS}
    for model in MODELS:
        biases: list[float] = []
        for ld in LEAD_DAYS:
            pred_path = MODELS_ROOT / model / f"lead_day_{ld}" / "step1" / "pred_tmax_daily.nc"
            if not pred_path.exists():
                biases.append(float("nan"))
                continue
            with xr.open_dataset(pred_path) as ds_m:
                pred_ts_hw = weighted_area_mean(
                    ds_m.sel(time=slice(hw_start, hw_end)), "tmax_pred_c", a_lat, a_lon
                )
                pred_mean = float(pred_ts_hw.mean(dim="time").values.item())
                biases.append(pred_mean - gt_mean)
        results[model] = biases

    pd.DataFrame(results).to_csv(OUT_ROOT / f"case_{case_name}_bias.csv", index=False)


def main() -> None:
    meta_path = OUT_ROOT / "case_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(CASE_CFG, f, ensure_ascii=True, indent=2)

    gt_ds = xr.open_dataset(GT_CACHE)
    p90_ds = xr.open_dataset(P90_FILE)
    for name, cfg in CASE_CFG.items():
        print(f"Extracting V7: {name}...")
        extract_map_data(gt_ds, p90_ds, name, cfg)
        extract_timeseries_data(gt_ds, name, cfg)
        extract_bias_data(gt_ds, name, cfg)
    gt_ds.close()
    p90_ds.close()
    print(f"Done. Meta: {meta_path}")


if __name__ == "__main__":
    main()

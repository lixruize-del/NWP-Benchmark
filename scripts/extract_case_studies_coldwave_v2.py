#!/usr/bin/env python3
"""Cold-surge case studies v2: Rank2 / Rank6 / Rank17 — map & analysis boxes auto-tuned to heatwave-style spans."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label as nd_label

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.common.repo_paths import data_dir, nwp_outputs_dir  # noqa: E402

_HW = nwp_outputs_dir() / "era5_monthly_202506_v2/metrics/heatwave_object_v2"
GT_CACHE = _HW / "ifs/_shared/gt_tmax_daily_2025_20250701_20251231.nc"
P10_FILE = data_dir() / "coldwave_baseline_p10_doy_001_366_fullrun_20260504_loadv1.nc"
MODELS_ROOT = _HW / "ifs"
OUT_ROOT = MODELS_ROOT / "case_studies_coldwave_v2"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MODELS = ["aifs", "aurora", "fuxi", "fengwu", "pangu", "graphcast", "stormer"]
LEAD_DAYS = [1, 3, 7, 10]

# Match typical heatwave case-study map/analysis spans (v7): map ~20°×30°, analysis ~8°×12°.
MAP_D_LAT = 20.0
MAP_D_LON = 30.0
ANAL_D_LAT = 8.0
ANAL_D_LON = 12.0

# Only temporal + seed regions; map_box / analysis_box filled by compute_boxes_from_peak().
CASE_BASE = {
    "Rank2_EastAsia": {
        "plot_dates": ("2025-09-26", "2025-11-03"),
        "cw_dates": ("2025-10-07", "2025-10-23"),
        "peak": "2025-10-16",
        "seed_region": {"lat": (36.0, 66.0), "lon": (58.0, 128.0)},
    },
    "Rank6_NA_Lakes": {
        "plot_dates": ("2025-11-28", "2025-12-31"),
        "cw_dates": ("2025-12-08", "2025-12-26"),
        "peak": "2025-12-12",
        "seed_region": {"lat": (43.0, 71.0), "lon": (188.0, 262.0)},
    },
    "Rank17_NA_US": {
        "plot_dates": ("2025-11-22", "2025-12-10"),
        "cw_dates": ("2025-11-29", "2025-12-06"),
        "peak": "2025-12-05",
        "seed_region": {"lat": (30.0, 58.0), "lon": (238.0, 306.0)},
    },
}


def weighted_area_mean(ds: xr.Dataset, var: str, lat_range: tuple[float, float], lon_range: tuple[float, float]) -> xr.DataArray:
    subset = ds.sel(latitude=slice(max(lat_range), min(lat_range)))
    subset = subset.sel(longitude=slice(lon_range[0], lon_range[1]))
    weights = np.cos(np.deg2rad(subset.latitude))
    weighted_sum = (subset[var] * weights).sum(dim=("latitude", "longitude"))
    total_weight = weights.sum(dim="latitude") * subset.longitude.size
    return weighted_sum / total_weight


def compute_boxes_from_peak(
    gt_ds: xr.Dataset,
    p10_ds: xr.Dataset,
    peak: str,
    seed: dict[str, tuple[float, float]],
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """
    Map box: fixed MAP_D_LAT × MAP_D_LON, centred on the area-weighted centroid of the largest
    connected component where P10 - Tmin > 0 inside seed_region (P10 cold-surge blob vs ERA5 Tmin).

    Analysis box: fixed ANAL_D_LAT × ANAL_D_LON, centred on the coldest Tmin gridpoint within the
    final map crop (prefer pixels with Tmin < P10 when present).
    """
    peak_dt = np.datetime64(peak)
    slat = seed["lat"]
    slon = seed["lon"]

    gt_sub = gt_ds.sel(
        time=peak_dt,
        latitude=slice(max(slat), min(slat)),
        longitude=slice(slon[0], slon[1]),
    )
    doy = pd.to_datetime(peak).dayofyear
    p10_sub = p10_ds.sel(
        doy=doy,
        latitude=slice(max(slat), min(slat)),
        longitude=slice(slon[0], slon[1]),
    )
    tmin = gt_sub["tmin_gt_c"].values.astype(np.float64)
    p10v = p10_sub["t2m_p10_c"].values.astype(np.float64)
    diff = p10v - tmin
    exceed = diff > 0

    lats_1d = gt_sub["latitude"].values.astype(np.float64)
    lons_1d = gt_sub["longitude"].values.astype(np.float64)
    lat_grid, lon_grid = np.meshgrid(lats_1d, lons_1d, indexing="ij")

    if np.any(exceed):
        labeled, nfeat = nd_label(exceed)
        best_k = 1
        if nfeat > 1:
            best_size = -1.0
            for k in range(1, nfeat + 1):
                s = float(np.sum(labeled == k))
                if s > best_size:
                    best_size = s
                    best_k = k
        mask = labeled == best_k
        w = np.cos(np.deg2rad(lat_grid)) * mask.astype(np.float64)
        sw = float(w.sum())
        if sw > 0:
            lat_c = float((lat_grid * w).sum() / sw)
            lon_c = float((lon_grid * w).sum() / sw)
        else:
            lat_c = float(lat_grid[mask].mean())
            lon_c = float(lon_grid[mask].mean())
    else:
        lat_c = 0.5 * (slat[0] + slat[1])
        lon_c = 0.5 * (slon[0] + slon[1])

    half_lat = MAP_D_LAT / 2.0
    half_lon = MAP_D_LON / 2.0
    map_lat = (lat_c - half_lat, lat_c + half_lat)
    lon_lo = float(np.clip(lon_c - half_lon, 0.0, 360.0))
    lon_hi = float(np.clip(lon_c + half_lon, 0.0, 360.0))
    if lon_hi <= lon_lo:
        lon_hi = min(360.0, lon_lo + MAP_D_LON)
    map_lon = (lon_lo, lon_hi)

    # Coldest point on map_box crop (prefer cold-surge pixels)
    gt_m = gt_ds.sel(
        time=peak_dt,
        latitude=slice(max(map_lat), min(map_lat)),
        longitude=slice(map_lon[0], map_lon[1]),
    )
    p10_m = p10_ds.sel(
        doy=doy,
        latitude=slice(max(map_lat), min(map_lat)),
        longitude=slice(map_lon[0], map_lon[1]),
    )
    tmin_m = gt_m["tmin_gt_c"].values.astype(np.float64)
    p10_m = p10_m["t2m_p10_c"].values.astype(np.float64)
    cold_mask = (p10_m - tmin_m) > 0
    lat_m = gt_m["latitude"].values.astype(np.float64)
    lon_m = gt_m["longitude"].values.astype(np.float64)

    if np.any(cold_mask):
        t_search = np.where(cold_mask, tmin_m, np.nan)
        iy, ix = np.unravel_index(int(np.nanargmin(t_search)), tmin_m.shape)
    else:
        iy, ix = np.unravel_index(int(np.argmin(tmin_m)), tmin_m.shape)
    lat_star = float(lat_m[iy])
    lon_star = float(lon_m[ix])

    ah_lat = ANAL_D_LAT / 2.0
    ah_lon = ANAL_D_LON / 2.0
    a_lat = (lat_star - ah_lat, lat_star + ah_lat)
    a_lon = (lon_star - ah_lon, lon_star + ah_lon)
    # Keep analysis inside map extent
    a_lat = (max(map_lat[0], a_lat[0]), min(map_lat[1], a_lat[1]))
    a_lon = (max(map_lon[0], a_lon[0]), min(map_lon[1], a_lon[1]))

    map_box = {"lat": (float(map_lat[0]), float(map_lat[1])), "lon": (float(map_lon[0]), float(map_lon[1]))}
    analysis_box = {"lat": (float(a_lat[0]), float(a_lat[1])), "lon": (float(a_lon[0]), float(a_lon[1]))}
    return map_box, analysis_box


def extract_map_data(gt_ds: xr.Dataset, p10_ds: xr.Dataset, case_name: str, cfg: dict) -> None:
    peak_dt = np.datetime64(cfg["peak"])
    m_lat = cfg["map_box"]["lat"]
    m_lon = cfg["map_box"]["lon"]
    gt_peak = gt_ds.sel(
        time=peak_dt,
        latitude=slice(max(m_lat), min(m_lat)),
        longitude=slice(m_lon[0], m_lon[1]),
    )
    doy = pd.to_datetime(cfg["peak"]).dayofyear
    p10_peak = p10_ds.sel(
        doy=doy,
        latitude=slice(max(m_lat), min(m_lat)),
        longitude=slice(m_lon[0], m_lon[1]),
    )
    out_ds = xr.Dataset(
        data_vars={
            "tmin_gt": gt_peak["tmin_gt_c"],
            "p10": p10_peak["t2m_p10_c"],
        },
        attrs={
            "case": case_name,
            "peak_day": cfg["peak"],
            "plot_dates": str(cfg["plot_dates"]),
            "cw_dates": str(cfg["cw_dates"]),
            "map_deg": f"{MAP_D_LAT}lat x {MAP_D_LON}lon",
            "analysis_deg": f"{ANAL_D_LAT}lat x {ANAL_D_LON}lon",
        },
    )
    out_ds.to_netcdf(OUT_ROOT / f"case_{case_name}_map_data.nc")


def extract_timeseries_data(gt_ds: xr.Dataset, case_name: str, cfg: dict) -> None:
    start_dt, end_dt = (np.datetime64(cfg["plot_dates"][0]), np.datetime64(cfg["plot_dates"][1]))
    if start_dt > end_dt:
        raise ValueError(f"{case_name}: plot_dates reversed: {cfg['plot_dates']}")

    a_lat = cfg["analysis_box"]["lat"]
    a_lon = cfg["analysis_box"]["lon"]
    gt_ts = weighted_area_mean(gt_ds.sel(time=slice(start_dt, end_dt)), "tmin_gt_c", a_lat, a_lon)
    results: dict[str, object] = {"time": gt_ts.time.values, "era5": gt_ts.values}

    for model in MODELS:
        pred_path = MODELS_ROOT / model / "lead_day_3" / "step1" / "pred_tmin_daily.nc"
        if not pred_path.exists():
            results[model] = np.full(len(gt_ts), np.nan)
            continue
        with xr.open_dataset(pred_path) as ds_m:
            pred_ts = weighted_area_mean(ds_m.sel(time=slice(start_dt, end_dt)), "tmin_pred_c", a_lat, a_lon)
            results[model] = pred_ts.values

    pd.DataFrame(results).to_csv(OUT_ROOT / f"case_{case_name}_timeseries.csv", index=False)


def extract_bias_data(gt_ds: xr.Dataset, case_name: str, cfg: dict) -> None:
    cw_start, cw_end = (np.datetime64(cfg["cw_dates"][0]), np.datetime64(cfg["cw_dates"][1]))
    if cw_start > cw_end:
        raise ValueError(f"{case_name}: cw_dates reversed: {cfg['cw_dates']}")

    a_lat = cfg["analysis_box"]["lat"]
    a_lon = cfg["analysis_box"]["lon"]

    gt_ts_cw = weighted_area_mean(gt_ds.sel(time=slice(cw_start, cw_end)), "tmin_gt_c", a_lat, a_lon)
    gt_mean = float(gt_ts_cw.mean(dim="time").values.item())

    results: dict[str, object] = {"lead_day": LEAD_DAYS}
    for model in MODELS:
        biases: list[float] = []
        for ld in LEAD_DAYS:
            pred_path = MODELS_ROOT / model / f"lead_day_{ld}" / "step1" / "pred_tmin_daily.nc"
            if not pred_path.exists():
                biases.append(float("nan"))
                continue
            with xr.open_dataset(pred_path) as ds_m:
                pred_ts_cw = weighted_area_mean(
                    ds_m.sel(time=slice(cw_start, cw_end)), "tmin_pred_c", a_lat, a_lon
                )
                pred_mean = float(pred_ts_cw.mean(dim="time").values.item())
                biases.append(pred_mean - gt_mean)
        results[model] = biases

    pd.DataFrame(results).to_csv(OUT_ROOT / f"case_{case_name}_bias.csv", index=False)


def main() -> None:
    gt_ds = xr.open_dataset(GT_CACHE)
    p10_ds = xr.open_dataset(P10_FILE)

    case_cfg_out: dict = {}
    for name, base in CASE_BASE.items():
        seed = base["seed_region"]
        peak = base["peak"]
        map_box, analysis_box = compute_boxes_from_peak(gt_ds, p10_ds, peak, seed)
        case_cfg_out[name] = {
            "plot_dates": base["plot_dates"],
            "cw_dates": base["cw_dates"],
            "peak": peak,
            "map_box": {"lat": list(map_box["lat"]), "lon": list(map_box["lon"])},
            "analysis_box": {"lat": list(analysis_box["lat"]), "lon": list(analysis_box["lon"])},
        }
        print(f"{name}: map_box lat={map_box['lat']} lon={map_box['lon']}")
        print(f"           analysis_box lat={analysis_box['lat']} lon={analysis_box['lon']}")

    meta_path = OUT_ROOT / "case_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(case_cfg_out, f, ensure_ascii=True, indent=2)

    for name, cfg in case_cfg_out.items():
        cfg_inline = {
            "plot_dates": tuple(cfg["plot_dates"]),
            "cw_dates": tuple(cfg["cw_dates"]),
            "peak": cfg["peak"],
            "map_box": {"lat": tuple(cfg["map_box"]["lat"]), "lon": tuple(cfg["map_box"]["lon"])},
            "analysis_box": {"lat": tuple(cfg["analysis_box"]["lat"]), "lon": tuple(cfg["analysis_box"]["lon"])},
        }
        print(f"Extracting coldwave v2: {name}...")
        extract_map_data(gt_ds, p10_ds, name, cfg_inline)
        extract_timeseries_data(gt_ds, name, cfg_inline)
        extract_bias_data(gt_ds, name, cfg_inline)

    gt_ds.close()
    p10_ds.close()
    print(f"Done. Meta: {meta_path}")


if __name__ == "__main__":
    main()

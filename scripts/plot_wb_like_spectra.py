#!/usr/bin/env python3
"""WeatherBench2-style zonal power spectra: compute, cache, and plot.

Purpose
    Build a multi-panel figure comparing several NWP model forecasts to ERA5
    ground truth in zonal wavenumber space (log-log), for chosen variables and
    lead times.

Forecast inputs
    ERA5-initialized monthly benchmark layout: under ``--forecast-root/<model>/<YYYYMMDDHH>/``
    NetCDF files named like ``YYYY-MMdd-<lead>.nc`` (lead in hours). All models
    including ``neuralgcm`` live as sibling directories under the same root.

Ground truth
    ERA5 fields from ``--gt-pressure-root`` and ``--gt-single-root`` (np.25-style
    daily ``*.npy`` trees), aligned to each init + lead valid time.

Spectrum definition
    Matches the WeatherBench2 ``ZonalEnergySpectrum`` normalization: forward
    real FFT, one-sided power scaling for k>0, and latitude-dependent circumference
    weighting before latitude reduction (``--lat-mean``).

Caching
    Writes ``--cache-file`` (pickle) with per-model curves and ERA5 curves.
    ``--resume-cache`` skips recomputation when metadata matches; use
    ``--refresh-models`` / ``--refresh-variables`` for partial invalidation.

Models without a variable (e.g. no ``t2m`` in a file) are skipped for that
variable only; plotting omits missing curves.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import math
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


EARTH_RADIUS_M = 6_371_000.0
LEAD_HOURS_DEFAULT = [6, 72, 120, 240]  # 6h, 3d, 5d, 10d
VARIABLES_DEFAULT = ["z500", "q700", "u850", "t2m"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    from src.common.repo_paths import nwp_outputs_dir

    p.add_argument(
        "--forecast-root",
        default=str(nwp_outputs_dir() / "era5_monthly_202506_v2" / "forecasts"),
        help="Primary forecast root (ERA5-init monthly layout: <model>/<init>/…-lead.nc).",
    )
    p.add_argument(
        "--gt-pressure-root",
        default="/ecmwf-era5-datasets/era5_np.25/2025",
        help="ERA5 pressure-level GT root (daily folders with *.npy).",
    )
    p.add_argument(
        "--gt-single-root",
        default="/ecmwf-era5-datasets/era5_np.25/single/2025",
        help="ERA5 single-level GT root (for t2m).",
    )
    p.add_argument(
        "--models",
        default="aifs,aurora,fuxi,fengwu,pangu,graphcast,stormer,neuralgcm",
        help="Comma-separated model names under --forecast-root.",
    )
    p.add_argument(
        "--variables",
        default=",".join(VARIABLES_DEFAULT),
        help="Comma-separated vars from: z500,q700,u850,t2m",
    )
    p.add_argument(
        "--lead-hours",
        default=",".join(str(x) for x in LEAD_HOURS_DEFAULT),
        help="Comma-separated lead hours (e.g., 6,72,120,240).",
    )
    p.add_argument(
        "--max-inits",
        type=int,
        default=None,
        help="Optional cap on number of init directories per model.",
    )
    p.add_argument(
        "--init-start",
        default=None,
        help="Optional init lower bound, format YYYYMMDD or YYYYMMDDHH (inclusive).",
    )
    p.add_argument(
        "--init-end",
        default=None,
        help="Optional init upper bound, format YYYYMMDD or YYYYMMDDHH (inclusive).",
    )
    p.add_argument(
        "--lat-mean",
        choices=["mean", "cosine"],
        default="mean",
        help="How to reduce spectra over latitude.",
    )
    p.add_argument(
        "--output",
        default="nwp_outputs/era5_monthly_202506_v2/figures/wb_like_spectra_2025.png",
        help="Output figure path.",
    )
    p.add_argument(
        "--cache-file",
        default=None,
        help="Path to cached spectra data (.pkl). Default: <output>.pkl",
    )
    p.add_argument(
        "--resume-cache",
        action="store_true",
        help="Resume from cache-file and skip already computed model curves.",
    )
    p.add_argument(
        "--refresh-models",
        default=None,
        help="Comma-separated model names to recompute even if present in cache.",
    )
    p.add_argument(
        "--refresh-variables",
        default=None,
        help=(
            "Comma-separated vars to recompute for --refresh-models only "
            "(default: all --variables)."
        ),
    )
    return p.parse_args()


def parse_models(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> list[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def parse_variables(s: str) -> list[str]:
    out = [x.strip().lower() for x in s.split(",") if x.strip()]
    valid = {"z500", "q700", "u850", "t2m"}
    bad = [x for x in out if x not in valid]
    if bad:
        raise ValueError(f"Unsupported variables: {bad}. allowed={sorted(valid)}")
    return out


def normalize_init_bound(s: str, is_end: bool) -> str:
    s = s.strip()
    if len(s) == 8 and s.isdigit():
        return s + ("18" if is_end else "00")
    if len(s) == 10 and s.isdigit():
        return s
    raise ValueError(f"Invalid init bound format: {s}. expected YYYYMMDD or YYYYMMDDHH")


def list_init_dirs(
    model_dir: Path,
    max_inits: int | None,
    init_start: str | None = None,
    init_end: str | None = None,
) -> list[Path]:
    dirs = sorted([p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()])
    if init_start is not None:
        dirs = [p for p in dirs if p.name >= init_start]
    if init_end is not None:
        dirs = [p for p in dirs if p.name <= init_end]
    if max_inits is not None:
        dirs = dirs[:max_inits]
    return dirs


LEAD_RE = re.compile(r"-(\d+)\.nc$")


def find_forecast_file_for_lead(init_dir: Path, lead_hour: int) -> Path | None:
    for p in init_dir.glob("*.nc"):
        m = LEAD_RE.search(p.name)
        if m and int(m.group(1)) == int(lead_hour):
            return p
    return None


def parse_init_time(init_name: str) -> dt.datetime:
    # init_name format: YYYYMMDDHH
    return dt.datetime.strptime(init_name, "%Y%m%d%H")


def valid_time_from_init_and_lead(init_name: str, lead_hour: int) -> dt.datetime:
    return parse_init_time(init_name) + dt.timedelta(hours=int(lead_hour))


def field_from_forecast(ds: xr.Dataset, var_key: str) -> xr.DataArray:
    if var_key == "z500":
        return ds["z"].sel(plev_z=500).squeeze(drop=True)
    if var_key == "q700":
        return ds["q"].sel(plev_q=700).squeeze(drop=True)
    if var_key == "u850":
        return ds["u"].sel(plev_u=850).squeeze(drop=True)
    if var_key == "t2m":
        return ds["t2m"].squeeze(drop=True)
    raise ValueError(var_key)


def read_gt_field(
    gt_pressure_root: Path,
    gt_single_root: Path,
    valid_time: dt.datetime,
    var_key: str,
) -> np.ndarray:
    day = valid_time.strftime("%Y-%m-%d")
    hhmmss = valid_time.strftime("%H:%M:%S")
    if var_key == "z500":
        fp = gt_pressure_root / day / f"{hhmmss}-z-500.0.npy"
    elif var_key == "q700":
        fp = gt_pressure_root / day / f"{hhmmss}-q-700.0.npy"
    elif var_key == "u850":
        fp = gt_pressure_root / day / f"{hhmmss}-u-850.0.npy"
    elif var_key == "t2m":
        fp = gt_single_root / day / f"{hhmmss}-t2m.npy"
    else:
        raise ValueError(var_key)
    if not fp.exists():
        raise FileNotFoundError(str(fp))
    return np.load(fp)


def zonal_energy_spectrum_weatherbench(
    field_lat_lon: np.ndarray, latitudes_deg: np.ndarray
) -> np.ndarray:
    """Compute WeatherBench-style zonal energy spectrum per latitude.

    Args:
      field_lat_lon: shape [nlat, nlon]
      latitudes_deg: shape [nlat]
    Returns:
      spectrum_lat_k: shape [nlat, nlon//2 + 1]
    """
    f_k = np.fft.rfft(field_lat_lon, axis=-1, norm="forward")
    power = np.real(f_k * np.conj(f_k))
    mult = np.ones(power.shape[-1], dtype=power.dtype)
    mult[1:] = 2.0
    power = power * mult[None, :]

    circum_equator = 2.0 * np.pi * EARTH_RADIUS_M
    circumference = np.cos(np.deg2rad(latitudes_deg)) * circum_equator
    return power * circumference[:, None]


def reduce_latitude(spectrum_lat_k: np.ndarray, latitudes: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return np.nanmean(spectrum_lat_k, axis=0)
    if mode == "cosine":
        w = np.cos(np.deg2rad(latitudes))
        w = np.where(np.isfinite(w), w, 0.0)
        return np.nansum(spectrum_lat_k * w[:, None], axis=0) / np.nansum(w)
    raise ValueError(mode)


def wavenumbers_for_nlon(nlon: int) -> np.ndarray:
    return np.arange(0, nlon // 2 + 1)


def build_init_lead_index(
    model_dir: Path,
    lead_hours: list[int],
    max_inits: int | None,
    init_start: str | None,
    init_end: str | None,
) -> dict[str, dict[int, Path]]:
    wanted = set(int(x) for x in lead_hours)
    out: dict[str, dict[int, Path]] = {}
    for init_dir in list_init_dirs(model_dir, max_inits, init_start=init_start, init_end=init_end):
        lead_map: dict[int, Path] = {}
        for p in init_dir.glob("*.nc"):
            m = LEAD_RE.search(p.name)
            if not m:
                continue
            lead = int(m.group(1))
            if lead in wanted:
                lead_map[lead] = p
        if lead_map:
            out[init_dir.name] = lead_map
    return out


def compute_model_spectra(
    model_dir: Path,
    variables: list[str],
    lead_hours: list[int],
    max_inits: int | None,
    lat_mean: str,
    init_start: str | None,
    init_end: str | None,
) -> tuple[dict[str, dict[int, tuple[np.ndarray, np.ndarray]]], np.ndarray, int, list[str]]:
    """Compute all model spectra in one pass over files.

    Returns:
      - nested dict curves[var][lead] = (k, mean_spectrum)
      - latitudes array
      - nlon
      - init names that had at least one requested lead
    """
    index = build_init_lead_index(
        model_dir=model_dir,
        lead_hours=lead_hours,
        max_inits=max_inits,
        init_start=init_start,
        init_end=init_end,
    )
    init_names = sorted(index.keys())
    accum: dict[tuple[str, int], list[np.ndarray]] = collections.defaultdict(list)
    latitudes: np.ndarray | None = None
    nlon: int | None = None

    for init_name in init_names:
        lead_map = index[init_name]
        for lead in lead_hours:
            nc_path = lead_map.get(lead)
            if nc_path is None:
                continue
            try:
                ds = xr.open_dataset(nc_path)
                if latitudes is None:
                    latitudes = np.asarray(ds["latitude"].values, dtype=np.float64)
                    nlon = int(ds.sizes["longitude"])
                for var_key in variables:
                    try:
                        field = field_from_forecast(ds, var_key)
                        arr = np.asarray(field.values, dtype=np.float64)
                        if arr.ndim != 2:
                            continue
                        spec_lat_k = zonal_energy_spectrum_weatherbench(arr, latitudes)
                        accum[(var_key, lead)].append(reduce_latitude(spec_lat_k, latitudes, lat_mean))
                    except Exception:
                        continue
                ds.close()
            except Exception:
                continue

    if latitudes is None or nlon is None:
        raise RuntimeError(f"Could not read any forecast files from {model_dir}")

    curves: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {v: {} for v in variables}
    k = wavenumbers_for_nlon(nlon)
    for var_key in variables:
        for lead in lead_hours:
            vals = accum.get((var_key, lead), [])
            if vals:
                curves[var_key][lead] = (k, np.nanmean(np.stack(vals, axis=0), axis=0))
    return curves, latitudes, nlon, init_names


def compute_gt_spectra(
    gt_pressure_root: Path,
    gt_single_root: Path,
    init_names: list[str],
    variables: list[str],
    lead_hours: list[int],
    lat_mean: str,
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    # ERA5 np.25 GT is strictly 721x1440, from 90 to -90
    latitudes = np.linspace(90, -90, 721)
    nlon = 1440
    
    accum: dict[tuple[str, int], list[np.ndarray]] = collections.defaultdict(list)
    for init_name in init_names:
        for lead in lead_hours:
            valid_time = valid_time_from_init_and_lead(init_name, lead)
            for var_key in variables:
                try:
                    arr = read_gt_field(
                        gt_pressure_root=gt_pressure_root,
                        gt_single_root=gt_single_root,
                        valid_time=valid_time,
                        var_key=var_key,
                    )
                    arr = np.asarray(arr, dtype=np.float64)
                    if arr.shape != (latitudes.size, nlon):
                        continue
                    spec_lat_k = zonal_energy_spectrum_weatherbench(arr, latitudes)
                    accum[(var_key, lead)].append(reduce_latitude(spec_lat_k, latitudes, lat_mean))
                except Exception:
                    continue

    curves: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {v: {} for v in variables}
    k = wavenumbers_for_nlon(nlon)
    for var_key in variables:
        for lead in lead_hours:
            vals = accum.get((var_key, lead), [])
            if vals:
                curves[var_key][lead] = (k, np.nanmean(np.stack(vals, axis=0), axis=0))
    return curves


def get_lat_lon_template(model_dir: Path, lead_hour: int, max_inits: int | None) -> tuple[np.ndarray, int]:
    for init_dir in list_init_dirs(model_dir, max_inits):
        p = find_forecast_file_for_lead(init_dir, lead_hour)
        if p is None:
            continue
        ds = xr.open_dataset(p)
        try:
            latitudes = np.asarray(ds["latitude"].values, dtype=np.float64)
            nlon = int(ds.sizes["longitude"])
            return latitudes, nlon
        finally:
            ds.close()
    raise RuntimeError(f"Cannot find sample forecast file in {model_dir} for lead={lead_hour}")


def display_name(var_key: str) -> str:
    return {
        "z500": "Z500",
        "q700": "Q700",
        "u850": "U850",
        "t2m": "T2M",
    }[var_key]


def lead_label(hours: int) -> str:
    if hours % 24 == 0 and hours >= 24:
        return f"{hours // 24}d"
    return f"{hours}h"


# Figure legend: consistent casing and spacing (internal keys are lowercase / ERA5).
def model_legend_name(model_key: str) -> str:
    return {
        "ERA5": "ERA5",
        "aifs": "AIFS",
        "aurora": "Aurora",
        "graphcast": "Graphcast",
        "pangu": "Pangu",
        "fengwu": "FengWu",
        "fuxi": "FuXi",
        "stormer": "Stormer",
        "neuralgcm": "NeuralGCM",
    }.get(model_key, model_key)


def save_cache(cache_file: Path, payload: dict) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_file)


def load_cache(cache_file: Path) -> dict:
    with cache_file.open("rb") as f:
        return pickle.load(f)


def main() -> int:
    args = parse_args()
    forecast_root = Path(args.forecast_root)
    gt_pressure_root = Path(args.gt_pressure_root)
    gt_single_root = Path(args.gt_single_root)
    models = parse_models(args.models)
    variables = parse_variables(args.variables)
    lead_hours = parse_int_list(args.lead_hours)
    refresh_models = set(parse_models(args.refresh_models)) if args.refresh_models else set()
    refresh_variables = (
        parse_variables(args.refresh_variables) if args.refresh_variables else None
    )
    output_path = Path(args.output)
    init_start = normalize_init_bound(args.init_start, is_end=False) if args.init_start else None
    init_end = normalize_init_bound(args.init_end, is_end=True) if args.init_end else None
    cache_file = Path(args.cache_file) if args.cache_file else output_path.with_suffix(".pkl")

    # Get global reference inits from the first valid model.
    ref_model = models[0]
    ref_dir = forecast_root / ref_model

    # spectra[var][lead][model_name] = (k, spec)
    spectra: dict[str, dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]] = {
        var: {lead: {} for lead in lead_hours} for var in variables
    }

    model_curves: dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = {}
    latitudes: np.ndarray | None = None
    nlon: int | None = None
    ref_init_names: list[str] = []

    cache_ok = False
    payload: dict | None = None
    if args.resume_cache and cache_file.exists():
        payload = load_cache(cache_file)
        if (
            payload.get("variables") == variables
            and payload.get("lead_hours") == lead_hours
            and payload.get("models") == models
            and payload.get("lat_mean") == args.lat_mean
            and payload.get("forecast_root") == str(forecast_root)
            and payload.get("init_start") == init_start
            and payload.get("init_end") == init_end
        ):
            model_curves = payload.get("model_curves", {})
            latitudes = payload.get("latitudes")
            nlon = payload.get("nlon")
            ref_init_names = payload.get("ref_init_names", [])
            cache_ok = True
            print(f"[INFO] Resumed cache: {cache_file}")
        else:
            print(f"[WARN] Cache config mismatch, ignored: {cache_file}")

    gt_curves: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] | None = None
    if cache_ok and payload is not None:
        gt_curves = payload.get("gt_curves")

    for model in models:
        model_dir = forecast_root / model
        if not model_dir.exists():
            continue
        if model in model_curves and model not in refresh_models:
            print(f"[INFO] Skip model from cache: {model}")
            continue
        vars_eff = variables
        if model in refresh_models and refresh_variables is not None:
            vars_eff = refresh_variables
        print(f"[INFO] Computing model spectra: {model} ({model_dir}) vars={vars_eff}")
        curves, lat_i, nlon_i, init_names = compute_model_spectra(
            model_dir=model_dir,
            variables=vars_eff,
            lead_hours=lead_hours,
            max_inits=args.max_inits,
            lat_mean=args.lat_mean,
            init_start=init_start,
            init_end=init_end,
        )
        model_curves.setdefault(model, {})
        for vk in vars_eff:
            if vk in curves:
                model_curves[model][vk] = curves[vk]
        if latitudes is None:
            latitudes = lat_i
            nlon = nlon_i
        if model == ref_model:
            ref_init_names = init_names
        chk: dict = {
            "variables": variables,
            "lead_hours": lead_hours,
            "models": models,
            "lat_mean": args.lat_mean,
            "forecast_root": str(forecast_root),
            "init_start": init_start,
            "init_end": init_end,
            "model_curves": model_curves,
            "latitudes": latitudes,
            "nlon": nlon,
            "ref_init_names": ref_init_names,
        }
        if gt_curves is not None:
            chk["gt_curves"] = gt_curves
        save_cache(cache_file, chk)

    if latitudes is None or nlon is None:
        pass # All cached or missing, which is fine, matplotlib handles missing natively if cached
    if not ref_init_names:
        ref_init_names = sorted(
            build_init_lead_index(ref_dir, lead_hours, args.max_inits, init_start, init_end).keys()
        )

    if gt_curves is None:
        print("[INFO] Computing ERA5 GT spectra")
        gt_curves = compute_gt_spectra(
            gt_pressure_root=gt_pressure_root,
            gt_single_root=gt_single_root,
            init_names=ref_init_names,
            variables=variables,
            lead_hours=lead_hours,
            lat_mean=args.lat_mean,
        )
        save_cache(
            cache_file,
            {
                "variables": variables,
                "lead_hours": lead_hours,
                "models": models,
                "lat_mean": args.lat_mean,
                "forecast_root": str(forecast_root),
                "init_start": init_start,
                "init_end": init_end,
                "model_curves": model_curves,
                "latitudes": latitudes,
                "nlon": nlon,
                "ref_init_names": ref_init_names,
                "gt_curves": gt_curves,
            },
        )

    for var in variables:
        for lead in lead_hours:
            if lead in gt_curves[var]:
                spectra[var][lead]["ERA5"] = gt_curves[var][lead]
            for model in models:
                curves = model_curves.get(model, {})
                if lead in curves.get(var, {}):
                    spectra[var][lead][model] = curves[var][lead]

    nrows = len(variables)
    ncols = len(lead_hours)
    # Per-panel inches: nearly square, slight landscape (not a perfect square).
    panel_w_in, panel_h_in = 3.92, 3.88
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(panel_w_in * ncols, panel_h_in * nrows),
        sharex=True,
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[None, :]
    elif ncols == 1:
        axes = axes[:, None]

    # Typography: panel titles & legend were 2× prior title size; shrink those + ticks by 2/5 (×3/5).
    base_fs = float(plt.rcParams["font.size"])
    try:
        title_base = float(plt.rcParams["axes.titlesize"])
    except (TypeError, ValueError):
        title_base = base_fs * 1.15
    shrink_plot_text = 3.0 / 5.0
    title_legend_boost = 1.25 * 1.25  # cumulative +1/4 on panel titles & legend (again vs prior pass)
    title_fs = title_base * 2.0 * shrink_plot_text * title_legend_boost
    label_fs = title_base
    legend_fs = (base_fs + 2.0) * 2.0 * shrink_plot_text * title_legend_boost
    tick_fs = base_fs * shrink_plot_text

    line_order = ["ERA5"] + models
    color_map = {
        "ERA5": "black",
        "aifs": "#1f77b4",
        "aurora": "#ff7f0e",
        "fuxi": "#2ca02c",
        "fengwu": "#d62728",
        "pangu": "#9467bd",
        "graphcast": "#8c564b",
        "stormer": "#e377c2",
        "neuralgcm": "#17becf",
    }

    for r, var in enumerate(variables):
        for c, lead in enumerate(lead_hours):
            ax = axes[r, c]
            d = spectra[var][lead]
            for name in line_order:
                if name not in d:
                    continue
                k, y = d[name]
                # Plot from k=1 to avoid singular large-scale component dominance.
                mask = k >= 1
                ax.plot(
                    k[mask],
                    y[mask],
                    label=model_legend_name(name),
                    lw=1.8,
                    color=color_map.get(name),
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", which="major", labelsize=tick_fs)
            ax.tick_params(axis="both", which="minor", labelsize=max(tick_fs - 1.0, 6.0))
            ax.set_title(
                f"{display_name(var)} - {lead_label(lead)}",
                fontsize=title_fs,
                fontweight="bold",
            )
            if c == 0:
                ax.set_ylabel("Mean power", fontsize=label_fs, fontweight="bold")
            if r == nrows - 1:
                ax.set_xlabel("Zonal wavenumber", fontsize=label_fs, fontweight="bold")

    # Figure-level legend: fixed order, names match model_legend_name; size/bold like subplot titles.
    label_to_handle: dict[str, object] = {}
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in label_to_handle:
                label_to_handle[ll] = hh
    legend_handles: list[object] = []
    legend_labels: list[str] = []
    for name in line_order:
        lab = model_legend_name(name)
        if lab in label_to_handle:
            legend_handles.append(label_to_handle[lab])
            legend_labels.append(lab)
    if legend_handles:
        # Two rows: ceil(n/2) columns → row-major fill; odd n gives balanced 5+4-style blocks, figure-centered.
        ncol = max(1, math.ceil(len(legend_handles) / 2))
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.26),
            ncol=ncol,
            frameon=False,
            columnspacing=1.15,
            handletextpad=0.38,
            labelspacing=0.55,
            prop={"size": legend_fs, "weight": "bold"},
        )
    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.96,
        bottom=0.37,
        wspace=0.19,
        hspace=0.21,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")
    save_cache(
        cache_file,
        {
            "variables": variables,
            "lead_hours": lead_hours,
            "models": models,
            "lat_mean": args.lat_mean,
            "forecast_root": str(forecast_root),
            "init_start": init_start,
            "init_end": init_end,
            "model_curves": model_curves,
            "latitudes": latitudes,
            "nlon": nlon,
            "ref_init_names": ref_init_names,
            "gt_curves": gt_curves,
        },
    )
    print(f"Saved cache: {cache_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -----------------------------------------------------------------------------
# Example commands (paths are illustrative; adjust to your environment)
# -----------------------------------------------------------------------------
#
# From repo root, after activating a conda env with matplotlib/xarray/numpy:
#
#   cd /path/to/NWP-Benchmark
#   python scripts/plot_wb_like_spectra.py \
#     --forecast-root "${REPO_ROOT}/nwp_outputs/era5_monthly_202506_v2/forecasts" \
#     --gt-pressure-root "/ecmwf-era5-datasets/era5_np.25/2025" \
#     --gt-single-root "/ecmwf-era5-datasets/era5_np.25/single/2025" \
#     --models "aifs,aurora,fuxi,fengwu,pangu,graphcast,stormer,neuralgcm" \
#     --variables "z500,q700,u850,t2m" \
#     --lead-hours "6,72,120,240" \
#     --init-start "2025070100" \
#     --init-end "2025123118" \
#     --output "${REPO_ROOT}/nwp_outputs/ifs_monthly_202506_v2/figures/wb_like_spectra_h2.png" \
#     --cache-file "${REPO_ROOT}/nwp_outputs/ifs_monthly_202506_v2/figures/wb_like_spectra_h2.pkl" \
#     --resume-cache
#
# Recompute only NeuralGCM (after changing its forecasts), keep other models from cache:
#
#   python scripts/plot_wb_like_spectra.py ...same as above... \
#     --resume-cache --refresh-models neuralgcm
#
# Omit t2m if a model set has no surface field:
#
#   python scripts/plot_wb_like_spectra.py \
#     --forecast-root "/path/to/forecasts" \
#     --models "neuralgcm" \
#     --variables "z500,q700,u850" \
#     --lead-hours "6,72,120,240" \
#     --init-start "2025010100" \
#     --init-end "2025123118" \
#     --gt-pressure-root "/ecmwf-era5-datasets/era5_np.25/2025" \
#     --gt-single-root "/ecmwf-era5-datasets/era5_np.25/single/2025" \
#     --output "./figures/wb_like_spectra_neuralgcm.png" \
#     --cache-file "./figures/wb_like_spectra_neuralgcm.pkl"

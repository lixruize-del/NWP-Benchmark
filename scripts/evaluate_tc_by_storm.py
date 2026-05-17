#!/usr/bin/env python3
"""Storm-centric tropical cyclone track evaluation using tracker.Tracker.

This script evaluates track error by looping:
model -> storm(SID) -> init_time(storm exists) -> lead_time(6..max_lead).

Design choices:
- Multi-model batch by default.
- Forecast paths are resolved lazily per (init, lead); no startup scan of all ``*/init/*.nc``
  on huge NAS trees (only one optional glob per init directory, cached).
- Optional ``--resume``: skip inits already marked finished in prior CSV output; with checkpointing
  after each init so crashes do not lose progress (see ``--no-checkpoint-every-init``). Prior raw CSV is
  loaded lazily only when new rows need merging; skipped inits avoid per-line logging by default (fast on SSH).
- Lagrangian continuity policy: break current init lead-loop on interruption.
  Interruption includes:
  - storm_end      : no IBTrACS truth at valid_time
  - missing_file   : forecast file for lead not found
  - tracker_fail   : tracker.step(...) raised exception
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracker import Tracker

LEAD_RE = re.compile(r".*-(\d+)(?:_[0-9]+)?\.nc$")

# stormer
DEFAULT_MODELS = ["aifs", "aurora", "fengwu", "fuxi", "graphcast", "pangu", "stormer"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Storm-centric typhoon track evaluation.")
    p.add_argument(
        "--forecast-root",
        type=Path,
        default=ROOT / "nwp_outputs" / "era5_monthly_202506_v2" / "forecasts",
        help="Forecast root: <forecast-root>/<model>/<YYYYMMDDHH>/*.nc",
    )
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model list to evaluate.")
    p.add_argument("--source", type=str, default="ERA5", help="Source tag for output path naming.")
    p.add_argument(
        "--ibtracs-csv",
        type=Path,
        default=ROOT / "data" / "tc" / "ibtracs.last3years.list.v04r01.csv",
        help="IBTrACS CSV file path.",
    )
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--start-date", type=str, default="2025-06-01", help="YYYY-MM-DD")
    p.add_argument("--end-date", type=str, default="2025-12-31", help="YYYY-MM-DD")
    p.add_argument("--basins", nargs="*", default=None, help="Optional BASIN filters (e.g. WP EP).")
    p.add_argument("--subbasins", nargs="*", default=None, help="Optional SUBBASIN filters.")
    p.add_argument("--sid", nargs="*", default=None, help="Optional SID whitelist.")
    p.add_argument("--storm-name", nargs="*", default=None, help="Optional storm NAME whitelist.")
    p.add_argument(
        "--single-init-time",
        type=str,
        default="",
        help="Optional single init time (YYYYMMDDHH) to run one trajectory only.",
    )
    p.add_argument("--lead-step", type=int, default=6)
    p.add_argument("--max-lead", type=int, default=240)
    p.add_argument("--distance-threshold", type=float, default=560.0, help="Tracker distance threshold (km).")
    p.add_argument("--wind-threshold", type=float, default=8.0, help="Tracker wind threshold (m/s).")
    p.add_argument("--resolution", type=float, default=0.25, help="Tracker grid resolution hint.")
    p.add_argument(
        "--forecast-nc-cache-size",
        type=int,
        default=0,
        help=(
            "LRU cache for open forecast NetCDF datasets keyed by resolved path (reuse when the same "
            "file is needed across storms/inits; reduces repeated NAS reads). 0 disables. "
            "Try 64–128 for slow NFS (e.g. GraphCast IFS)."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "tc_eval_results" / "storm_centric",
        help="Output root. Writes to <out-dir>/<source>/<model>_*.csv",
    )
    p.add_argument(
        "--plot-trajectory",
        action="store_true",
        help="When used with --single-init-time, plot predicted vs true trajectory.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Load existing per-model CSVs under --out-dir/<source>/ and skip inits already marked "
            "finished (see --resume-skip-reasons). By default also checkpoints CSVs after each init "
            "so a crash does not lose progress. That checkpoint rewrites the full raw+term+by_lead files "
            "and can be slow on large tables or slow storage; use --no-checkpoint-every-init to only "
            "write at the end of each model (faster, less crash-safe)."
        ),
    )
    p.add_argument(
        "--resume-skip-reasons",
        nargs="*",
        default=None,
        metavar="REASON",
        help=(
            "Terminate_Reason values that mean 'this init is done; skip on resume'. "
            "Default: max_lead_reached storm_end. Pass an empty list with --resume-skip-reasons "
            "(shell-dependent) to skip no inits by reason — prefer omitting --resume for full re-run."
        ),
    )
    p.add_argument(
        "--no-checkpoint-every-init",
        action="store_true",
        help=(
            "With --resume, do not write CSVs after each init (only at end of each model). "
            "Much faster for long runs because each init no longer re-merges and rewrites the full raw CSV."
        ),
    )
    p.add_argument(
        "--resume-print-each-init",
        action="store_true",
        help=(
            "With --resume, print one log line per skipped init (can be very slow on tmux/SSH). "
            "Default is quiet + periodic progress only."
        ),
    )
    p.add_argument(
        "--resume-heartbeat-every",
        type=int,
        default=250,
        metavar="N",
        help=(
            "With --resume and without --resume-print-each-init, print a progress line every N skipped "
            "inits. Use 0 for fully quiet skips. Default: 250."
        ),
    )
    return p


def _parse_lead_hours(path: Path) -> int | None:
    m = LEAD_RE.match(path.name)
    if not m:
        return None
    return int(m.group(1))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r, lat2r = np.deg2rad(lat1), np.deg2rad(lat2)
    lon1r, lon2r = np.deg2rad(lon1), np.deg2rad(lon2)
    inner = 1 - np.cos(lat2r - lat1r) + np.cos(lat1r) * np.cos(lat2r) * (1 - np.cos(lon2r - lon1r))
    # Guard against tiny negative values caused by floating-point precision.
    inner = np.clip(0.5 * inner, 0.0, 1.0)
    return float(2 * 6371.0 * np.arcsin(np.sqrt(inner)))


def _load_ibtracs(
    csv_path: Path,
    season: int,
    start_dt: datetime,
    end_dt: datetime,
    basins: list[str] | None,
    subbasins: list[str] | None,
    sid_whitelist: list[str] | None,
    storm_names: list[str] | None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    try:
        df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
    except TypeError:
        df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce", utc=True).dt.tz_localize(None)
    df["SEASON"] = pd.to_numeric(df.get("SEASON"), errors="coerce")
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce") % 360.0
    df["WMO_PRES"] = pd.to_numeric(df.get("WMO_PRES"), errors="coerce")
    df["WMO_WIND"] = pd.to_numeric(df.get("WMO_WIND"), errors="coerce")
    df["SID"] = df["SID"].astype(str).str.strip()
    df["NAME"] = df["NAME"].astype(str).str.strip()
    df["BASIN"] = df.get("BASIN", pd.Series(["UNKNOWN"] * len(df))).fillna("UNKNOWN").astype(str)
    df["SUBBASIN"] = df.get("SUBBASIN", pd.Series(["UNKNOWN"] * len(df))).fillna("UNKNOWN").astype(str)

    df = df.dropna(subset=["ISO_TIME", "LAT", "LON"]).copy()
    df = df[(df["SEASON"] == season) & (df["ISO_TIME"] >= start_dt) & (df["ISO_TIME"] <= end_dt)].copy()
    df = df[(df["ISO_TIME"].dt.hour.isin([0, 6, 12, 18])) & (df["ISO_TIME"].dt.minute == 0)].copy()

    if basins:
        basin_set = {b.upper() for b in basins}
        df = df[df["BASIN"].str.upper().isin(basin_set)].copy()
    if subbasins:
        sub_set = {s.upper() for s in subbasins}
        df = df[df["SUBBASIN"].str.upper().isin(sub_set)].copy()
    if sid_whitelist:
        sid_set = set(sid_whitelist)
        df = df[df["SID"].isin(sid_set)].copy()
    if storm_names:
        name_set = {n.upper() for n in storm_names}
        df = df[df["NAME"].str.upper().isin(name_set)].copy()

    return df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)


def _datepart_forecast_filename(init_dt: datetime) -> str:
    """Middle token in ``YYYY-MMDD-LLL.nc`` (same convention as monthly forecast saves / ``run_large_scale_ifs``)."""
    return init_dt.strftime("%Y-%m%d")


def _deterministic_nc_filenames(init_dt: datetime, lead_h: int) -> list[str]:
    dp = _datepart_forecast_filename(init_dt)
    names: list[str] = []
    for fmt in (str(lead_h), f"{lead_h:02d}", f"{lead_h:03d}"):
        names.append(f"{dp}-{fmt}.nc")
    return list(dict.fromkeys(names))


def _forecast_candidates_for_lead(
    model: str,
    init_dt: datetime,
    lead_h: int,
    root_chain: list[tuple[Path, str]],
    init_dir_nc_cache: dict[str, list[Path]],
) -> list[tuple[Path, str]]:
    """Resolve candidate NetCDF paths without scanning the entire forecast tree.

    1) Deterministic names under ``<root>/<model>/<YYYYMMDDHH>/``.
    2) If needed, a single cached ``*.nc`` listing per init directory (only when
    that directory is first seen), then match by parsed lead hours.

    This avoids walking all ``*/init/*.nc`` on multi-TB NAS trees at process start.
    """
    init_str = init_dt.strftime("%Y%m%d%H")
    det_names = _deterministic_nc_filenames(init_dt, lead_h)
    out: list[tuple[Path, str]] = []
    seen: set[str] = set()

    def _add(p: Path, tag: str) -> None:
        key = str(p.resolve())
        if key in seen:
            return
        seen.add(key)
        out.append((p, tag))

    for root, tag in root_chain:
        init_dir = root / model / init_str
        for fname in det_names:
            _add(init_dir / fname, tag)

    for root, tag in root_chain:
        init_dir = root / model / init_str
        cache_key = str(init_dir.resolve())
        if cache_key not in init_dir_nc_cache:
            init_dir_nc_cache[cache_key] = sorted(init_dir.glob("*.nc")) if init_dir.is_dir() else []
        for p in init_dir_nc_cache[cache_key]:
            lh = _parse_lead_hours(p)
            if lh == lead_h:
                _add(p, tag)
    return out


def _init_has_parseable_nc(
    model: str,
    init_key: str,
    root_chain: list[tuple[Path, str]],
    init_dir_nc_cache: dict[str, list[Path]],
) -> bool:
    """True iff some root has ``<root>/<model>/<init_key>/*.nc`` with a parseable lead (matches old index)."""
    for root, _ in root_chain:
        init_dir = root / model / init_key
        cache_key = str(init_dir.resolve())
        if cache_key not in init_dir_nc_cache:
            init_dir_nc_cache[cache_key] = sorted(init_dir.glob("*.nc")) if init_dir.is_dir() else []
        for p in init_dir_nc_cache[cache_key]:
            if _parse_lead_hours(p) is not None:
                return True
    return False


def _open_forecast_dataset(nc_path: Path) -> xr.Dataset:
    """Open a forecast NetCDF file.

    Prefer explicit engines (str path avoids rare Path backend matching issues).
    """
    path = Path(nc_path)
    if not path.is_file():
        raise FileNotFoundError(f"forecast file missing: {path}")

    errors: list[str] = []
    for engine in ("netcdf4", "h5netcdf", "scipy"):
        try:
            return xr.open_dataset(str(path), engine=engine)
        except Exception as exc:
            errors.append(f"{engine}:{type(exc).__name__}:{exc}")
            continue
    try:
        return xr.open_dataset(str(path))
    except Exception as exc:
        errors.append(f"auto:{type(exc).__name__}:{exc}")
    detail = " | ".join(errors)
    raise RuntimeError(f"could not open forecast NetCDF: {path}\n{detail}")


class ForecastNcLRUCache:
    """Keep a bounded number of open xarray.Dataset handles keyed by resolved path.

    Same spirit as avoiding repeated HDF5 opens on slow storage (cf. backfill scripts).
    Caller must not close datasets returned from open(); eviction closes evicted handles.
    """

    def __init__(self, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self._store: OrderedDict[str, xr.Dataset] = OrderedDict()

    def open(self, nc_path: Path) -> xr.Dataset:
        key = str(Path(nc_path).resolve())
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        ds = _open_forecast_dataset(nc_path)
        self._store[key] = ds
        while len(self._store) > self.max_entries:
            _, old_ds = self._store.popitem(last=False)
            old_ds.close()
        return ds

    def clear(self) -> None:
        for ds in self._store.values():
            ds.close()
        self._store.clear()


def _to_iso(t: Any) -> str:
    return pd.to_datetime(t).strftime("%Y-%m-%d %H:%M:%S")


DEFAULT_RESUME_SKIP_REASONS = frozenset({"max_lead_reached", "storm_end"})


def _norm_sid_val(s: Any) -> str:
    return str(s).strip()


def _init_completion_key(sid: Any, init_dt: datetime) -> tuple[str, str]:
    return (_norm_sid_val(sid), _to_iso(init_dt))


def _completed_init_keys_from_term(term_df: pd.DataFrame, skip_reasons: frozenset[str]) -> set[tuple[str, str]]:
    if term_df.empty or not skip_reasons:
        return set()
    out: set[tuple[str, str]] = set()
    for _, row in term_df.iterrows():
        reason = str(row.get("Terminate_Reason", "") or "")
        if reason not in skip_reasons:
            continue
        sid = _norm_sid_val(row.get("Storm_ID", ""))
        it = str(row.get("Init_Time", "")).strip()
        if sid and it:
            out.add((sid, it))
    return out


def _merge_session_outputs(
    existing_raw: pd.DataFrame,
    existing_term: pd.DataFrame,
    session_raw: list[dict[str, Any]],
    session_term: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_new = pd.DataFrame(session_raw) if session_raw else pd.DataFrame()
    term_new = pd.DataFrame(session_term) if session_term else pd.DataFrame()
    if existing_raw.empty:
        raw_all = raw_new
    else:
        raw_all = pd.concat([existing_raw, raw_new], ignore_index=True)
    if not raw_all.empty and "Eval_Key" in raw_all.columns:
        raw_all = raw_all.drop_duplicates(subset=["Eval_Key"], keep="last")

    if existing_term.empty:
        term_all = term_new
    else:
        term_all = pd.concat([existing_term, term_new], ignore_index=True)
    if not term_all.empty and "Storm_ID" in term_all.columns and "Init_Time" in term_all.columns:
        term_all = term_all.drop_duplicates(subset=["Storm_ID", "Init_Time"], keep="last")
    return raw_all, term_all


def _track_error_rmse_km(series: pd.Series) -> float:
    """RMSE of great-circle track error (km): sqrt(mean(e^2))."""
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(x))))


def _write_model_csv_bundle(
    out_root: Path, model: str, raw_df: pd.DataFrame, term_df: pd.DataFrame
) -> tuple[Path, Path, Path, pd.DataFrame]:
    raw_csv = out_root / f"{model}_storm_eval_raw.csv"
    term_csv = out_root / f"{model}_storm_eval_terminations.csv"
    by_lead_csv = out_root / f"{model}_by_lead.csv"
    raw_df.to_csv(raw_csv, index=False)
    term_df.to_csv(term_csv, index=False)
    if raw_df.empty:
        by_lead_df = pd.DataFrame(
            columns=[
                "Lead_Time",
                "N",
                "Track_Error_km_mean",
                "Track_Error_km_median",
                "Track_Error_km_RMSE",
                "Wind_MAE_ms",
                "Wind_Bias_ms",
                "MSL_MAE_hPa",
                "MSL_Bias_hPa",
            ]
        )
    else:
        _bl = raw_df.copy()
        if "Status" in _bl.columns:
            _bl = _bl[_bl["Status"] == "ok"].copy()
        if {"Pred_MSL_hPa", "True_MSL_hPa"}.issubset(_bl.columns):
            _bl["Pred_MSL_hPa"] = pd.to_numeric(_bl["Pred_MSL_hPa"], errors="coerce")
            _bl["True_MSL_hPa"] = pd.to_numeric(_bl["True_MSL_hPa"], errors="coerce")
            _bl["MSL_Abs_Error_hPa"] = (_bl["Pred_MSL_hPa"] - _bl["True_MSL_hPa"]).abs()
            _bl["MSL_Error_hPa"] = _bl["Pred_MSL_hPa"] - _bl["True_MSL_hPa"]
        else:
            _bl["MSL_Abs_Error_hPa"] = np.nan
            _bl["MSL_Error_hPa"] = np.nan
        by_lead_df = (
            _bl.groupby("Lead_Time", dropna=False)
            .agg(
                N=("Track_Error_km", "size"),
                Track_Error_km_mean=("Track_Error_km", "mean"),
                Track_Error_km_median=("Track_Error_km", "median"),
                Track_Error_km_RMSE=("Track_Error_km", _track_error_rmse_km),
                Wind_MAE_ms=("Wind_Abs_Error_ms", "mean"),
                Wind_Bias_ms=("Wind_Error_ms", "mean"),
                MSL_MAE_hPa=("MSL_Abs_Error_hPa", "mean"),
                MSL_Bias_hPa=("MSL_Error_hPa", "mean"),
            )
            .reset_index()
            .sort_values("Lead_Time")
        )
    by_lead_df.to_csv(by_lead_csv, index=False)
    return raw_csv, term_csv, by_lead_csv, by_lead_df


def _prepare_tracker_dataset(ds: xr.Dataset, model: str) -> xr.Dataset:
    """Prepare dataset for tracker input.

    - For stormer: bilinear interpolate to 0.25-degree global grid.
    - For others: keep native grid.
    """
    work = ds
    if "latitude" not in work.coords and "lat" in work.coords:
        work = work.rename({"lat": "latitude"})
    if "longitude" not in work.coords and "lon" in work.coords:
        work = work.rename({"lon": "longitude"})

    if model.lower() != "stormer":
        return work

    required = [v for v in ("z", "msl", "u10", "v10") if v in work]
    work = work[required]
    target_lat = np.linspace(-90.0, 90.0, 721, endpoint=True)
    target_lon = np.linspace(0.0, 359.75, 1440, endpoint=True)
    return work.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )


def _plot_trajectory_compare(traj_df: pd.DataFrame, out_png: Path) -> None:
    if traj_df.empty:
        return
    d = traj_df.sort_values("Lead_Time").copy()
    d = d.dropna(subset=["Pred_Lat", "Pred_Lon", "True_Lat", "True_Lon"])
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        d["True_Lon"].values,
        d["True_Lat"].values,
        "-o",
        color="black",
        linewidth=1.8,
        markersize=4,
        label="IBTrACS Truth",
    )
    ax.plot(
        d["Pred_Lon"].values,
        d["Pred_Lat"].values,
        "-o",
        color="#1f77b4",
        linewidth=1.8,
        markersize=4,
        label="Model Track",
    )

    for _, r in d.iterrows():
        lead = int(r["Lead_Time"])
        if lead % 24 == 0:
            ax.text(float(r["Pred_Lon"]) + 0.05, float(r["Pred_Lat"]) + 0.05, f"{lead}h", fontsize=8, color="#1f77b4")

    ax.scatter([d["True_Lon"].iloc[0]], [d["True_Lat"].iloc[0]], marker="s", s=40, color="black")
    ax.scatter([d["Pred_Lon"].iloc[0]], [d["Pred_Lat"].iloc[0]], marker="s", s=40, color="#1f77b4")

    model = str(d["Model"].iloc[0])
    sid = str(d["Storm_ID"].iloc[0])
    name = str(d["Storm_Name"].iloc[0])
    init_time = str(d["Init_Time"].iloc[0])
    ax.set_title(f"{model.upper()} {name} ({sid})\nInit: {init_time}")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S")
    leads = list(range(args.lead_step, args.max_lead + 1, args.lead_step))
    ib = _load_ibtracs(
        csv_path=args.ibtracs_csv,
        season=args.season,
        start_dt=start_dt,
        end_dt=end_dt,
        basins=args.basins,
        subbasins=args.subbasins,
        sid_whitelist=args.sid,
        storm_names=args.storm_name,
    )
    if ib.empty:
        raise SystemExit("No IBTrACS rows after filtering. Check filters/time window.")

    out_root = args.out_dir / args.source.lower()
    out_root.mkdir(parents=True, exist_ok=True)

    storm_groups = list(ib.groupby("SID", sort=True))
    print(f"[info] storms_selected={len(storm_groups)} rows={len(ib)} leads={len(leads)}")

    root_chain: list[tuple[Path, str]] = [(args.forecast_root, "primary")]

    for model in args.models:
        nc_cache: ForecastNcLRUCache | None = None
        if args.forecast_nc_cache_size > 0:
            nc_cache = ForecastNcLRUCache(args.forecast_nc_cache_size)
            print(f"[model={model}] forecast_nc_cache_size={args.forecast_nc_cache_size}")
        init_dir_nc_cache: dict[str, list[Path]] = {}
        print(
            f"[model={model}] forecast_index=lazy_no_tree_scan "
            f"roots={len(root_chain)} primary={args.forecast_root}"
        )

        checkpoint_every_init = bool(args.resume) and (not args.no_checkpoint_every_init)
        existing_raw_df = pd.DataFrame()
        existing_term_df = pd.DataFrame()
        existing_raw_path = out_root / f"{model}_storm_eval_raw.csv"
        term_p = out_root / f"{model}_storm_eval_terminations.csv"
        existing_raw_loaded = False

        def _ensure_existing_raw_loaded() -> None:
            """Load prior raw rows only when merging/checkpointing (skip-path avoids huge CSV read)."""
            nonlocal existing_raw_df, existing_raw_loaded
            if existing_raw_loaded:
                return
            existing_raw_loaded = True
            if existing_raw_path.exists():
                existing_raw_df = pd.read_csv(existing_raw_path)

        if args.resume and term_p.exists():
            existing_term_df = pd.read_csv(term_p)
        if args.resume_skip_reasons is not None:
            skip_reasons = frozenset(args.resume_skip_reasons)
        else:
            skip_reasons = DEFAULT_RESUME_SKIP_REASONS
        completed_inits = (
            _completed_init_keys_from_term(existing_term_df, skip_reasons) if args.resume else set()
        )
        print(
            f"[model={model}] resume={bool(args.resume)} completed_inits_skippable={len(completed_inits)} "
            f"checkpoint_every_init={checkpoint_every_init} resume_skip_reasons={sorted(skip_reasons)}"
        )

        raw_records: list[dict[str, Any]] = []
        term_records: list[dict[str, Any]] = []
        terminate_counts = {"storm_end": 0, "missing_file": 0, "tracker_fail": 0, "max_lead_reached": 0}
        skipped_resume = 0

        for sid, storm_df in storm_groups:
            storm_df = storm_df.sort_values("ISO_TIME").reset_index(drop=True)
            name = str(storm_df["NAME"].iloc[0])
            basin = str(storm_df["BASIN"].iloc[0])
            subbasin = str(storm_df["SUBBASIN"].iloc[0])
            obs_map = {pd.Timestamp(t): row for t, row in storm_df.set_index("ISO_TIME").iterrows()}
            obs_times = list(obs_map.keys())

            print(f"[model={model}] storm={name} sid={sid} init_count={len(obs_times)}")

            for init_time in obs_times:
                init_dt = pd.Timestamp(init_time).to_pydatetime()
                init_key = init_dt.strftime("%Y%m%d%H")
                if args.single_init_time and init_key != args.single_init_time:
                    continue
                if args.resume and _init_completion_key(sid, init_dt) in completed_inits:
                    skipped_resume += 1
                    if args.resume_print_each_init:
                        print(
                            f"[resume] skip model={model} storm={name} sid={sid} init={init_key} "
                            "(already finished per prior terminations CSV)"
                        )
                    else:
                        hb = args.resume_heartbeat_every
                        if hb > 0 and skipped_resume % hb == 0:
                            print(f"[resume] skipped {skipped_resume} inits so far...")
                    continue
                terminate_reason = "max_lead_reached"
                terminate_lead = args.max_lead
                detail = ""
                start_idx = len(raw_records)

                init_row = obs_map[init_time]
                tracker = Tracker(
                    init_lat=float(init_row["LAT"]),
                    init_lon=float(init_row["LON"]),
                    init_time=init_dt,
                    distance_threshold=args.distance_threshold,
                    wind_threshold=args.wind_threshold,
                    resolution=args.resolution,
                )

                if not _init_has_parseable_nc(model, init_key, root_chain, init_dir_nc_cache):
                    terminate_reason = "missing_file"
                    terminate_lead = 0
                    detail = f"missing init directory or no parseable nc for {init_key}"
                else:
                    for lead in leads:
                        valid_dt = init_dt + timedelta(hours=lead)
                        valid_ts = pd.Timestamp(valid_dt)
                        if valid_ts not in obs_map:
                            terminate_reason = "storm_end"
                            terminate_lead = lead
                            detail = "no IBTrACS truth at valid_time"
                            break

                        candidates = _forecast_candidates_for_lead(
                            model, init_dt, lead, root_chain, init_dir_nc_cache
                        )
                        if not candidates:
                            print(
                                f"[warn] skip lead model={model} sid={sid} init={init_key} "
                                f"lead={lead}: no forecast candidates"
                            )
                            continue

                        ds = None
                        nc_path: Path | None = None
                        file_source_tag = ""
                        open_errors: list[str] = []
                        for cand_path, cand_tag in candidates:
                            try:
                                if nc_cache is not None:
                                    ds = nc_cache.open(cand_path)
                                else:
                                    ds = _open_forecast_dataset(cand_path)
                                nc_path = cand_path
                                file_source_tag = cand_tag
                                break
                            except Exception as exc:
                                open_errors.append(f"{cand_path} ({cand_tag}): {type(exc).__name__}: {exc}")
                                continue
                        if ds is None:
                            print(
                                f"[warn] skip lead model={model} sid={sid} init={init_key} "
                                f"lead={lead}: all forecast candidates unreadable "
                                f"({len(open_errors)} tries)"
                            )
                            continue
                        close_base_ds = nc_cache is None
                        ds_tracker = ds
                        try:
                            ds_tracker = _prepare_tracker_dataset(ds, model)
                            n_before = len(tracker.tracked_times)
                            tracker.step(ds_tracker)
                            if getattr(tracker, "end_flag", False):
                                raise RuntimeError("tracker_end_flag_true")
                            if len(tracker.tracked_times) <= n_before:
                                raise RuntimeError("tracker_no_new_step")
                        except Exception as exc:
                            terminate_reason = "tracker_fail"
                            terminate_lead = lead
                            detail = f"{type(exc).__name__}: {exc}"
                            break
                        finally:
                            if ds_tracker is not ds:
                                ds_tracker.close()
                            if close_base_ds:
                                ds.close()

                        pred_lat = float(tracker.tracked_lats[-1])
                        pred_lon = float(tracker.tracked_lons[-1])
                        pred_msl = float(tracker.tracked_msls[-1]) if len(tracker.tracked_msls) else np.nan
                        pred_wind = float(tracker.tracked_winds[-1]) if len(tracker.tracked_winds) else np.nan

                        true_row = obs_map[valid_ts]
                        true_lat = float(true_row["LAT"])
                        true_lon = float(true_row["LON"])
                        true_msl = float(true_row["WMO_PRES"]) if pd.notna(true_row["WMO_PRES"]) else np.nan
                        true_wind = float(true_row["WMO_WIND"]) if pd.notna(true_row["WMO_WIND"]) else np.nan
                        if np.isfinite(true_wind):
                            true_wind *= 0.514444
                        wind_error = pred_wind - true_wind if np.isfinite(pred_wind) and np.isfinite(true_wind) else np.nan
                        wind_abs_error = abs(wind_error) if np.isfinite(wind_error) else np.nan

                        raw_records.append(
                            {
                                "Source": args.source,
                                "Model": model,
                                "Storm_ID": sid,
                                "Storm_Name": name,
                                "Basin": basin,
                                "Subbasin": subbasin,
                                "Init_Time": _to_iso(init_dt),
                                "Valid_Time": _to_iso(valid_dt),
                                "Lead_Time": int(lead),
                                "True_Lat": true_lat,
                                "True_Lon": true_lon,
                                "True_MSL_hPa": true_msl,
                                "True_Wind_ms": true_wind,
                                "Pred_Lat": pred_lat,
                                "Pred_Lon": pred_lon,
                                "Pred_MSL_hPa": pred_msl / 100.0 if np.isfinite(pred_msl) and pred_msl > 2000 else pred_msl,
                                "Pred_Wind_ms": pred_wind,
                                "Wind_Error_ms": wind_error,
                                "Wind_Abs_Error_ms": wind_abs_error,
                                "Track_Error_km": _haversine_km(pred_lat, pred_lon, true_lat, true_lon),
                                "Status": "ok",
                                "Terminate_Reason": "",
                                "Forecast_File": str(nc_path),
                                "Forecast_File_Source": file_source_tag,
                                "Eval_Key": f"{sid}|{init_dt.strftime('%Y%m%d%H')}|{int(lead)}",
                            }
                        )
                    else:
                        # Finished all leads without storm_end / tracker_fail break.
                        n_added = len(raw_records) - start_idx
                        if n_added == 0:
                            terminate_reason = "missing_file"
                            terminate_lead = 0
                            detail = "no readable forecast for any lead (all leads skipped or unavailable)"
                        else:
                            terminate_reason = "max_lead_reached"
                            terminate_lead = args.max_lead
                            detail = ""

                if terminate_reason not in terminate_counts:
                    terminate_counts[terminate_reason] = 0
                terminate_counts[terminate_reason] += 1
                for i in range(start_idx, len(raw_records)):
                    raw_records[i]["Terminate_Reason"] = terminate_reason

                term_records.append(
                    {
                        "Source": args.source,
                        "Model": model,
                        "Storm_ID": sid,
                        "Storm_Name": name,
                        "Init_Time": _to_iso(init_dt),
                        "Terminate_Reason": terminate_reason,
                        "Terminate_Lead_Time": int(terminate_lead),
                        "Detail": detail,
                    }
                )

                if checkpoint_every_init:
                    _ensure_existing_raw_loaded()
                    raw_ck, term_ck = _merge_session_outputs(
                        existing_raw_df, existing_term_df, raw_records, term_records
                    )
                    _write_model_csv_bundle(out_root, model, raw_ck, term_ck)

        if args.resume and not raw_records and not term_records:
            print(
                f"[done model={model}] skipped_resume={skipped_resume} no new rows this run; "
                "left existing CSVs unchanged"
            )
            if nc_cache is not None:
                nc_cache.clear()
            continue

        _ensure_existing_raw_loaded()
        raw_df, term_df = _merge_session_outputs(existing_raw_df, existing_term_df, raw_records, term_records)
        raw_csv, term_csv, by_lead_csv, by_lead_df = _write_model_csv_bundle(out_root, model, raw_df, term_df)
        low_n = by_lead_df[by_lead_df["N"] < 10][["Lead_Time", "N"]] if not by_lead_df.empty else pd.DataFrame()

        print(
            f"[done model={model}] raw={len(raw_df)} terminations={len(term_df)} "
            f"ok={len(raw_df)} terminate_counts={terminate_counts} skipped_resume={skipped_resume}"
        )
        if not low_n.empty:
            pairs = ", ".join([f"{int(r.Lead_Time)}h:{int(r.N)}" for _, r in low_n.iterrows()])
            print(f"  [qc] low sample leads (N<10): {pairs}")
        print(f"  wrote: {raw_csv}")
        print(f"  wrote: {term_csv}")
        print(f"  wrote: {by_lead_csv}")
        if nc_cache is not None:
            nc_cache.clear()
        if args.single_init_time:
            init_iso = datetime.strptime(args.single_init_time, "%Y%m%d%H").strftime("%Y-%m-%d %H:%M:%S")
            traj_df = raw_df[raw_df["Init_Time"] == init_iso].copy().sort_values("Lead_Time")
            traj_csv = out_root / f"{model}_trajectory_{args.single_init_time}.csv"
            traj_df.to_csv(traj_csv, index=False)
            print(f"  wrote: {traj_csv} ({len(traj_df)} rows)")
            if args.plot_trajectory:
                traj_png = out_root / f"{model}_trajectory_{args.single_init_time}.png"
                _plot_trajectory_compare(traj_df, traj_png)
                print(f"  wrote: {traj_png}")


if __name__ == "__main__":
    main()

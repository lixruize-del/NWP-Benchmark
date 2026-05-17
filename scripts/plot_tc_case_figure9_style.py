#!/usr/bin/env python3
"""Plot TC case figures in a Figure-9-like layout.

For each case (Storm_ID + Init_Time), generate:
- Top-left: ERA5 track map (truth + model tracks)
- Top-right: IFS track map (truth + model tracks)
- Bottom: lead-time track-error curves (models, ERA5 solid / IFS dashed)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.common.repo_paths import tc_eval_results_dir  # noqa: E402
from typing import Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


DEFAULT_MODELS = ["aifs", "aurora", "fengwu", "fuxi", "graphcast", "pangu", "stormer"]

# Case-study bottom panel + CSV: never use leads beyond this (hours).
CASE_PANEL_MAX_LEAD_H = 120

# Same hex mapping as scripts/plot_case_studies_v9.py MODEL_COLORS.
MODEL_COLORS = {
    "aifs": "#1f77b4",
    "aurora": "#ff7f0e",
    "fuxi": "#2ca02c",
    "fengwu": "#d62728",
    "pangu": "#9467bd",
    "graphcast": "#8c564b",
    "stormer": "#e377c2",
}

# Display names for legends (Pangu short form per paper style).
MODEL_DISPLAY_NAMES = {
    "aifs": "AIFS",
    "aurora": "Aurora",
    "fengwu": "FengWu",
    "fuxi": "FuXi",
    "graphcast": "GraphCast",
    "pangu": "Pangu",
    "stormer": "Stormer",
}


def _panel_c_series_label(model: str, source: str) -> str:
    """e.g. AIFS(ERA5), AIFS(oper.)."""
    base = MODEL_DISPLAY_NAMES.get(model, model.upper())
    if source == "era5":
        return f"{base}(ERA5)"
    if source == "ifs":
        return f"{base}(oper.)"
    return f"{base}({source.upper()})"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot Figure9-style TC case figures.")
    p.add_argument(
        "--storm-root",
        type=Path,
        default=tc_eval_results_dir() / "storm_centric",
        help="Root containing era5/ifs <model>_storm_eval_raw.csv.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=tc_eval_results_dir() / "final_report/case_figures",
        help="Directory for output case figures.",
    )
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--sources", nargs="+", default=["era5", "ifs"])
    p.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional explicit cases: SID@YYYY-MM-DD HH:MM:SS (quote each case).",
    )
    p.add_argument("--top-k", type=int, default=3, help="Auto-select top-k cases if --cases not provided.")
    p.add_argument(
        "--max-lead",
        type=int,
        default=120,
        help="Requested max lead for panel (c); capped at %d for plot and CSV." % CASE_PANEL_MAX_LEAD_H,
    )
    return p


def _load_raw(storm_root: Path, source: str, model: str) -> pd.DataFrame:
    p = storm_root / source / f"{model}_storm_eval_raw.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    df["Source"] = source
    df["Model"] = model
    df["Init_Time"] = pd.to_datetime(df["Init_Time"], errors="coerce")
    df["Valid_Time"] = pd.to_datetime(df["Valid_Time"], errors="coerce")
    df["Lead_Time"] = pd.to_numeric(df["Lead_Time"], errors="coerce")
    df["Track_Error_km"] = pd.to_numeric(df["Track_Error_km"], errors="coerce")
    return df


def _auto_cases(data: Dict[Tuple[str, str], pd.DataFrame], top_k: int) -> List[Tuple[str, pd.Timestamp, str]]:
    # Use pangu availability in ERA5+IFS as robust auto-case selector.
    e = data.get(("era5", "pangu"), pd.DataFrame())
    i = data.get(("ifs", "pangu"), pd.DataFrame())
    if e.empty or i.empty:
        return []

    g1 = (
        e.groupby(["Storm_ID", "Storm_Name", "Init_Time"], dropna=False)["Lead_Time"]
        .max()
        .reset_index(name="max_lead_era5")
    )
    g2 = (
        i.groupby(["Storm_ID", "Storm_Name", "Init_Time"], dropna=False)["Lead_Time"]
        .max()
        .reset_index(name="max_lead_ifs")
    )
    m = g1.merge(g2, on=["Storm_ID", "Storm_Name", "Init_Time"], how="inner")
    if m.empty:
        return []
    m["score"] = m[["max_lead_era5", "max_lead_ifs"]].min(axis=1)
    m = m.sort_values(["score", "Storm_ID", "Init_Time"], ascending=[False, True, True])
    out: List[Tuple[str, pd.Timestamp, str]] = []
    seen_sid = set()
    for _, r in m.iterrows():
        sid = str(r["Storm_ID"])
        if sid in seen_sid:
            continue
        out.append((sid, pd.Timestamp(r["Init_Time"]), str(r["Storm_Name"])))
        seen_sid.add(sid)
        if len(out) >= top_k:
            break
    return out


def _nice_major_step(span_deg: float, n_ticks: float = 5.5) -> float:
    """Pick a clean degree spacing for map gridlines (~n_ticks intervals across span)."""
    if span_deg <= 0 or not np.isfinite(span_deg):
        return 1.0
    raw = span_deg / n_ticks
    exp = 10 ** np.floor(np.log10(raw))
    m = raw / exp
    if m <= 1.0:
        nice = 1.0
    elif m <= 2.0:
        nice = 2.0
    elif m <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * exp)


def _extent_for_tracks(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> List[float]:
    """Square lon/lat extent: length along the longer storm axis (+padding) sets box side;
    the shorter axis is expanded symmetrically so margins match visually.
    """
    lon_r = float(lon_max - lon_min)
    lat_r = float(lat_max - lat_min)
    major_r = max(lon_r, lat_r, 1e-6)
    # Padding tied to along-track scale (same ° margin on both ends of the long side).
    pad = max(0.12 * major_r, 0.35)
    side_deg = major_r + 2.0 * pad
    mid_lon = 0.5 * (lon_min + lon_max)
    mid_lat = 0.5 * (lat_min + lat_max)
    half = 0.5 * side_deg
    return [mid_lon - half, mid_lon + half, mid_lat - half, mid_lat + half]


def _parse_cases(cases: List[str]) -> List[Tuple[str, pd.Timestamp]]:
    out: List[Tuple[str, pd.Timestamp]] = []
    for c in cases:
        if "@" not in c:
            raise ValueError(f"Invalid case format: {c}. Expect SID@YYYY-MM-DD HH:MM:SS")
        sid, t = c.split("@", 1)
        out.append((sid.strip(), pd.Timestamp(t.strip())))
    return out


def _format_map(ax, extent, title: str) -> None:
    lon0, lon1, lat0, lat1 = extent
    lon_span = lon1 - lon0
    lat_span = lat1 - lat0
    lon_step = _nice_major_step(lon_span)
    lat_step = _nice_major_step(lat_span)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="#BCE6E6", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="#555555", zorder=1)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor="blue", zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":", color="gray", zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.8, color="white", alpha=0.8, linestyle="-", zorder=2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 9, "color": "black"}
    gl.ylabel_style = {"size": 9, "color": "black"}
    gl.xlocator = mticker.MultipleLocator(lon_step)
    gl.ylocator = mticker.MultipleLocator(lat_step)
    # Title above map (default placement); avoids clipping when row spacing is tight.
    ax.set_title(title, fontsize=11, fontweight="bold", color="black", pad=10)


def _plot_case(
    sid: str,
    init_time: pd.Timestamp,
    storm_name: str,
    models: List[str],
    sources: List[str],
    data: Dict[Tuple[str, str], pd.DataFrame],
    out_png: Path,
    out_errors_csv: Path,
    max_lead: int,
) -> None:
    lead_cap = min(int(max_lead), CASE_PANEL_MAX_LEAD_H)
    # Gather tracks for extent calculation.
    lon_all: List[float] = []
    lat_all: List[float] = []
    truth_cache: Dict[str, pd.DataFrame] = {}

    for source in sources:
        for model in models:
            df = data.get((source, model), pd.DataFrame())
            if df.empty:
                continue
            cdf = df[(df["Storm_ID"] == sid) & (df["Init_Time"] == init_time)].copy()
            if cdf.empty:
                continue
            cdf = cdf.sort_values("Lead_Time")
            lon_all.extend(pd.to_numeric(cdf["Pred_Lon"], errors="coerce").dropna().tolist())
            lat_all.extend(pd.to_numeric(cdf["Pred_Lat"], errors="coerce").dropna().tolist())
            lon_all.extend(pd.to_numeric(cdf["True_Lon"], errors="coerce").dropna().tolist())
            lat_all.extend(pd.to_numeric(cdf["True_Lat"], errors="coerce").dropna().tolist())
            if source not in truth_cache and {"True_Lon", "True_Lat"}.issubset(cdf.columns):
                truth_cache[source] = cdf[["Lead_Time", "True_Lon", "True_Lat"]].dropna().drop_duplicates("Lead_Time")

    if not lon_all or not lat_all:
        return

    lon_min, lon_max = min(lon_all), max(lon_all)
    lat_min, lat_max = min(lat_all), max(lat_all)
    extent = _extent_for_tracks(lon_min, lon_max, lat_min, lat_max)

    fig = plt.figure(figsize=(12, 10))
    # Outer: maps row | error row — full-width alignment. Inner: two maps, zero gap.
    # Titles above maps — row gap slightly tighter than before but not clipping.
    gs_outer = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1.0], hspace=0.12)
    gs_maps = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[0], wspace=0.0)

    # (a) ERA5 map
    # (b) IFS map
    for idx, source in enumerate(sources[:2]):
        if HAS_CARTOPY:
            ax = fig.add_subplot(gs_maps[0, idx], projection=ccrs.PlateCarree())
            _format_map(ax, extent, f"({chr(ord('a') + idx)}) {source.upper()} {storm_name} ({sid})")
            trans = ccrs.PlateCarree()
        else:
            ax = fig.add_subplot(gs_maps[0, idx])
            ax.set_title(
                f"({chr(ord('a') + idx)}) {source.upper()} {storm_name} ({sid})",
                fontsize=11,
                fontweight="bold",
                color="black",
                pad=10,
            )
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            trans = None

        tdf = truth_cache.get(source)
        if tdf is not None and not tdf.empty:
            ax.plot(
                tdf["True_Lon"].values,
                tdf["True_Lat"].values,
                color="#3498db",
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=3.5,
                label="Ground Truth",
                transform=trans,
                zorder=4,
            )

        for model in models:
            df = data.get((source, model), pd.DataFrame())
            if df.empty:
                continue
            cdf = df[(df["Storm_ID"] == sid) & (df["Init_Time"] == init_time)].copy()
            if cdf.empty:
                continue
            cdf = cdf.sort_values("Lead_Time")
            ax.plot(
                cdf["Pred_Lon"].values,
                cdf["Pred_Lat"].values,
                color=MODEL_COLORS.get(model, "gray"),
                marker=".",
                linestyle="--",
                linewidth=1.1,
                markersize=3,
                label=MODEL_DISPLAY_NAMES.get(model, model.upper()),
                transform=trans,
                zorder=3,
            )
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9, prop={"weight": "bold"})

    # (c) error line chart — fixed 0–120 h; shorter series simply stop (no axis shrink).
    ax3 = fig.add_subplot(gs_outer[1])
    error_rows: List[dict] = []
    for model in models:
        color = MODEL_COLORS.get(model, "gray")
        for source in sources[:2]:
            df = data.get((source, model), pd.DataFrame())
            if df.empty:
                continue
            cdf = df[(df["Storm_ID"] == sid) & (df["Init_Time"] == init_time)].copy()
            if cdf.empty:
                continue
            cdf = cdf[cdf["Lead_Time"] <= lead_cap].sort_values("Lead_Time")
            if cdf.empty:
                continue
            ls = "-" if source == "era5" else "--"
            label = _panel_c_series_label(model, source)
            ax3.plot(
                cdf["Lead_Time"].values,
                cdf["Track_Error_km"].values,
                color=color,
                marker="o",
                linestyle=ls,
                linewidth=1.2,
                markersize=3,
                label=label,
                clip_on=False,
            )
            te = pd.to_numeric(cdf["Track_Error_km"], errors="coerce").dropna()
            if len(te):
                error_rows.append(
                    {
                        "Storm_ID": sid,
                        "Storm_Name": storm_name,
                        "Init_Time": init_time.isoformat(),
                        "Series": label,
                        "Model": model,
                        "Source": source.upper(),
                        "Mean_Track_Error_km": float(te.mean()),
                        "N_leads": int(len(te)),
                        "Max_Lead_h": int(lead_cap),
                    }
                )

    ax3.set_xlabel("Lead Time (hours)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Track Error (km)", fontsize=11, fontweight="bold")
    ax3.set_autoscalex_on(False)
    ax3.set_xlim(0.0, float(CASE_PANEL_MAX_LEAD_H))
    try:
        ax3.margins(x=0)
    except Exception:
        ax3.set_xmargin(0)
    ax3.set_xticks(np.arange(0, CASE_PANEL_MAX_LEAD_H + 1, 12))
    ax3.grid(True, linestyle="--", color="gray", alpha=0.5)
    ax3.tick_params(direction="in")
    ax3.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9, prop={"weight": "bold"})
    ax3.set_title(
        f"(c) {storm_name} ({sid}) Init {init_time.strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=11,
        y=-0.2,
        fontweight="bold",
        color="black",
    )

    # Manual layout: aligns map row with bottom panel; avoids tight_layout widening map gaps.
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.09, hspace=0.11)
    ax3.set_xlim(0.0, float(CASE_PANEL_MAX_LEAD_H))
    ax3.set_autoscalex_on(False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    if error_rows:
        pd.DataFrame(error_rows).sort_values(["Source", "Model"]).to_csv(out_errors_csv, index=False)


def main() -> None:
    args = _build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data: Dict[Tuple[str, str], pd.DataFrame] = {}
    for source in args.sources:
        for model in args.models:
            data[(source, model)] = _load_raw(args.storm_root, source, model)

    if args.cases:
        parsed = _parse_cases(args.cases)
        case_list: List[Tuple[str, pd.Timestamp, str]] = []
        # infer storm name from era5-pangu if possible
        ref = data.get(("era5", "pangu"), pd.DataFrame())
        for sid, init_t in parsed:
            name = sid
            if not ref.empty:
                sub = ref[(ref["Storm_ID"] == sid) & (ref["Init_Time"] == init_t)]
                if not sub.empty:
                    name = str(sub["Storm_Name"].iloc[0])
            case_list.append((sid, init_t, name))
    else:
        case_list = _auto_cases(data, args.top_k)

    if not case_list:
        raise SystemExit("No cases available for plotting. Provide --cases explicitly.")

    for sid, init_t, name in case_list:
        stem = f"figure9_case_{sid}_{init_t.strftime('%Y%m%d%H')}"
        out_png = args.output_dir / f"{stem}.png"
        out_csv = args.output_dir / f"{stem}_mean_track_errors.csv"
        _plot_case(
            sid=sid,
            init_time=init_t,
            storm_name=name,
            models=args.models,
            sources=args.sources,
            data=data,
            out_png=out_png,
            out_errors_csv=out_csv,
            max_lead=min(int(args.max_lead), CASE_PANEL_MAX_LEAD_H),
        )
        print(f"Wrote: {out_png}")
        if out_csv.exists():
            print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()


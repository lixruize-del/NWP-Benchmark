#!/usr/bin/env python3
"""Build storm-centric TC evaluation tables and bar charts.

Inputs:
  <input-root>/<source>/<model>_storm_eval_raw.csv

Outputs:
  <output-root>/intermediate/*.csv
  <output-root>/tables/*.csv
  <output-root>/figures/*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.common.repo_paths import tc_eval_results_dir  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MODELS = ["aifs", "aurora", "fengwu", "fuxi", "graphcast", "pangu", "stormer"]


def _track_error_rmse_km(series: pd.Series) -> float:
    """Strict RMSE of great-circle track error (km): sqrt(mean(e^2))."""
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(x))))


# Align with scripts/plot_case_studies_v9.py for cross-figure consistency.
MODEL_COLORS = {
    "aifs": "#1f77b4",
    "aurora": "#ff7f0e",
    "fuxi": "#2ca02c",
    "fengwu": "#d62728",
    "pangu": "#9467bd",
    "graphcast": "#8c564b",
    "stormer": "#e377c2",
}

MODEL_DISPLAY_NAMES = {
    "aifs": "AIFS",
    "aurora": "Aurora",
    "fengwu": "FengWu",
    "fuxi": "Fuxi",
    "graphcast": "GraphCast",
    "pangu": "Pangu-Weather",
    "stormer": "Stormer",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build TC report tables and figures.")
    p.add_argument(
        "--input-root",
        type=Path,
        default=tc_eval_results_dir() / "storm_centric",
        help="Root containing era5/ifs model raw CSVs.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=tc_eval_results_dir() / "final_report",
        help="Output root for intermediate/tables/figures.",
    )
    p.add_argument("--sources", nargs="+", default=["era5", "ifs"])
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--table-leads", nargs="+", type=int, default=[24, 48, 72, 96, 120])
    p.add_argument("--plot-max-lead", type=int, default=120)
    p.add_argument(
        "--no-dual-panel-png",
        action="store_true",
        help="Skip writing 1×2 ERA5|IFS combined figures (homogeneous/objective dual panels).",
    )
    p.add_argument("--figure-dpi", type=int, default=300, help="DPI for saved PNG figures.")
    return p


def _load_raw(input_root: Path, source: str, model: str) -> pd.DataFrame:
    p = input_root / source / f"{model}_storm_eval_raw.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    df["Model"] = model
    df["Source"] = source
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "Status" in out.columns:
        out = out[out["Status"] == "ok"].copy()
    out["Lead_Time"] = pd.to_numeric(out["Lead_Time"], errors="coerce")
    out["Track_Error_km"] = pd.to_numeric(out["Track_Error_km"], errors="coerce")
    out["Wind_Error_ms"] = pd.to_numeric(out.get("Wind_Error_ms"), errors="coerce")
    out["Wind_Abs_Error_ms"] = pd.to_numeric(out.get("Wind_Abs_Error_ms"), errors="coerce")
    if {"Pred_MSL_hPa", "True_MSL_hPa"}.issubset(out.columns):
        out["Pred_MSL_hPa"] = pd.to_numeric(out["Pred_MSL_hPa"], errors="coerce")
        out["True_MSL_hPa"] = pd.to_numeric(out["True_MSL_hPa"], errors="coerce")
        out["MSL_Abs_Error_hPa"] = (out["Pred_MSL_hPa"] - out["True_MSL_hPa"]).abs()
        out["MSL_Error_hPa"] = out["Pred_MSL_hPa"] - out["True_MSL_hPa"]
    else:
        out["MSL_Abs_Error_hPa"] = np.nan
        out["MSL_Error_hPa"] = np.nan
    out = out.dropna(subset=["Lead_Time", "Track_Error_km"]).copy()
    out["Lead_Time"] = out["Lead_Time"].astype(int)
    return out


def _objective_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Source",
                "Model",
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
    return (
        df.groupby(["Source", "Model", "Lead_Time"], dropna=False)
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
        .sort_values(["Source", "Model", "Lead_Time"])
    )


def _homogeneous_aggregate(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Source",
                "Model",
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

    out_frames: list[pd.DataFrame] = []
    for source, sdf in df.groupby("Source", sort=True):
        source_models = [m for m in models if m in set(sdf["Model"].unique())]
        if not source_models:
            continue
        for lead, ldf in sdf.groupby("Lead_Time", sort=True):
            key_sets = []
            for m in source_models:
                mdf = ldf[ldf["Model"] == m]
                key_sets.append(set(mdf["Eval_Key"].dropna().astype(str).tolist()))
            if not key_sets:
                continue
            common_keys = set.intersection(*key_sets)
            if not common_keys:
                continue
            hdf = ldf[ldf["Eval_Key"].astype(str).isin(common_keys)].copy()
            agg = (
                hdf.groupby(["Source", "Model", "Lead_Time"], dropna=False)
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
            )
            out_frames.append(agg)

    if not out_frames:
        return pd.DataFrame(
            columns=[
                "Source",
                "Model",
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
    return pd.concat(out_frames, ignore_index=True).sort_values(["Source", "Model", "Lead_Time"])


def _table_wide(df: pd.DataFrame, leads: list[int], objective: bool) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Source", "Model"])
    recs = []
    for (source, model), g in df.groupby(["Source", "Model"], sort=True):
        row: dict[str, float | str] = {"Source": source, "Model": model}
        for lead in leads:
            sub = g[g["Lead_Time"] == lead]
            if sub.empty:
                row[f"L{lead}_N"] = np.nan
                row[f"L{lead}_TrackErr_km"] = np.nan
                row[f"L{lead}_TrackErr_RMSE_km"] = np.nan
                row[f"L{lead}_MSL_MAE_hPa"] = np.nan
                row[f"L{lead}_MSL_Bias_hPa"] = np.nan
            else:
                row[f"L{lead}_N"] = float(sub["N"].iloc[0])
                row[f"L{lead}_TrackErr_km"] = float(sub["Track_Error_km_mean"].iloc[0])
                row[f"L{lead}_TrackErr_RMSE_km"] = float(sub["Track_Error_km_RMSE"].iloc[0])
                row[f"L{lead}_MSL_MAE_hPa"] = float(sub["MSL_MAE_hPa"].iloc[0])
                row[f"L{lead}_MSL_Bias_hPa"] = float(sub["MSL_Bias_hPa"].iloc[0])
        recs.append(row)
    out = pd.DataFrame(recs)
    if objective:
        return out.sort_values(["Source", "Model"]).reset_index(drop=True)
    return out.sort_values(["Source", "Model"]).reset_index(drop=True)


def _pivot_track_error(
    df: pd.DataFrame, source: str, plot_max_lead: int
):
    sdf = df[(df["Source"] == source) & (df["Lead_Time"] <= plot_max_lead)].copy()
    if sdf.empty:
        return None, None
    pivot = sdf.pivot_table(index="Lead_Time", columns="Model", values="Track_Error_km_mean", aggfunc="mean")
    pivot = pivot.sort_index().reindex(columns=DEFAULT_MODELS)
    leads = [int(x) for x in pivot.index.tolist()]
    return pivot, leads


def _grouped_bar_axes(
    ax,
    pivot: pd.DataFrame,
    leads: list[int],
    *,
    title: str,
    ymax: float | None,
    show_ylabel: bool,
) -> None:
    """Draw grouped bars; same x/y styling for paired panels."""
    models = [m for m in DEFAULT_MODELS if m in pivot.columns]
    if not leads:
        return
    x = np.arange(len(leads))
    nmod = len(models)
    width = 0.8 / max(nmod, 1)

    for i, m in enumerate(models):
        vals = pivot[m].values
        ax.bar(
            x + (i - (nmod - 1) / 2) * width,
            vals,
            width=width,
            color=MODEL_COLORS.get(m, "#888888"),
            label=MODEL_DISPLAY_NAMES.get(m, m),
            edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in leads])
    # Tighten horizontal margins (reduce gap before first lead / after last lead).
    ax.set_xlim(x[0] - 0.42, x[-1] + 0.42)
    ax.set_xlabel("Lead Time (h)", fontsize=15, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Mean Track Error (km)", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", direction="in")
    # Rotate x tick labels only; do not change tick font size or weight.
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    if ymax is not None:
        ax.set_ylim(0.0, ymax * 1.02)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        ncol=4,
        fontsize=8,
        framealpha=0.95,
        prop={"weight": "bold"},
    )


def _plot_grouped_bar(
    df: pd.DataFrame,
    source: str,
    out_png: Path,
    title_prefix: str,
    *,
    plot_max_lead: int,
    figure_dpi: int,
) -> None:
    pivot, leads = _pivot_track_error(df, source, plot_max_lead)
    if pivot is None or leads is None or pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    ymax = float(np.nanmax(pivot.to_numpy(dtype=float)))
    _grouped_bar_axes(
        ax,
        pivot,
        leads,
        title=f"{title_prefix} — {source.upper()}",
        ymax=ymax,
        show_ylabel=True,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_dual_era5_ifs(
    df: pd.DataFrame,
    out_png: Path,
    *,
    plot_max_lead: int,
    figure_dpi: int,
) -> None:
    """One row, two columns: ERA5 | IFS line charts, shared y-range, single legend (no figure suptitle)."""
    p_e5, leads_e5 = _pivot_track_error(df, "era5", plot_max_lead)
    p_ifs, leads_ifs = _pivot_track_error(df, "ifs", plot_max_lead)
    if p_e5 is None or p_ifs is None:
        return

    common_leads = sorted(set(leads_e5) & set(leads_ifs))
    if not common_leads:
        common_leads = sorted(set(leads_e5) | set(leads_ifs))
    p_e5 = p_e5.reindex(common_leads)
    p_ifs = p_ifs.reindex(common_leads)

    ymax = max(
        float(np.nanmax(p_e5.to_numpy(dtype=float))),
        float(np.nanmax(p_ifs.to_numpy(dtype=float))),
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.2),
        sharey=True,
        constrained_layout=False,
    )

    models = [m for m in DEFAULT_MODELS if m in p_e5.columns]
    x_leads = np.asarray(common_leads, dtype=float)
    span = float(x_leads[-1] - x_leads[0]) if len(x_leads) > 1 else 120.0
    x_pad = max(3.0, 0.02 * span)

    for ax, pivot, letter, src_title in zip(
        axes,
        (p_e5, p_ifs),
        ("a", "b"),
        ("ERA5", "IFS"),
    ):
        for m in models:
            vals = np.asarray(pivot[m].values, dtype=float)
            ax.plot(
                x_leads,
                vals,
                color=MODEL_COLORS.get(m, "#888888"),
                marker="o",
                markersize=4,
                linewidth=1.6,
                label=MODEL_DISPLAY_NAMES.get(m, m),
                clip_on=False,
            )
        ax.set_xticks(x_leads)
        ax.set_xticklabels([str(int(v)) for v in common_leads])
        ax.set_xlim(x_leads[0] - x_pad, x_leads[-1] + x_pad)
        ax.set_xlabel("Lead Time (h)", fontsize=15, fontweight="bold")
        ax.set_title(f"({letter}) {src_title}", fontsize=14, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="both", direction="in")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.set_ylim(0.0, ymax * 1.02)

    axes[0].set_ylabel("Mean Track Error (km)", fontsize=14, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    # Legend raised ~one letter height vs prior anchor (see bbox y).
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
        framealpha=0.95,
        prop={"weight": "bold"},
        fontsize=8,
    )

    plt.subplots_adjust(bottom=0.24, wspace=0.12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    out_intermediate = args.output_root / "intermediate"
    out_tables = args.output_root / "tables"
    out_figures = args.output_root / "figures"
    out_intermediate.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for source in args.sources:
        for model in args.models:
            df = _load_raw(args.input_root, source, model)
            if not df.empty:
                frames.append(df)
    if not frames:
        raise SystemExit("No input raw CSV found. Check --input-root and model/source filters.")

    raw_all = _clean(pd.concat(frames, ignore_index=True))
    objective = _objective_aggregate(raw_all)
    homogeneous = _homogeneous_aggregate(raw_all, args.models)

    objective.to_csv(out_intermediate / "objective_by_model_lead.csv", index=False)
    homogeneous.to_csv(out_intermediate / "homogeneous_by_model_lead.csv", index=False)

    table4 = _table_wide(objective, args.table_leads, objective=True)
    table3 = _table_wide(homogeneous, args.table_leads, objective=False)
    table4.to_csv(out_tables / "table4_objective.csv", index=False)
    table3.to_csv(out_tables / "table3_homogeneous.csv", index=False)

    for source in args.sources:
        _plot_grouped_bar(
            objective,
            source,
            out_figures / f"{source}_objective_track_error_bar.png",
            title_prefix="Objective sample: mean track error",
            plot_max_lead=args.plot_max_lead,
            figure_dpi=args.figure_dpi,
        )
        _plot_grouped_bar(
            homogeneous,
            source,
            out_figures / f"{source}_homogeneous_track_error_bar.png",
            title_prefix="Homogeneous sample: mean track error",
            plot_max_lead=args.plot_max_lead,
            figure_dpi=args.figure_dpi,
        )

    if (not args.no_dual_panel_png) and "era5" in args.sources and "ifs" in args.sources:
        _plot_dual_era5_ifs(
            homogeneous,
            out_figures / "homogeneous_era5_ifs_dual_panel.png",
            plot_max_lead=args.plot_max_lead,
            figure_dpi=args.figure_dpi,
        )
        _plot_dual_era5_ifs(
            objective,
            out_figures / "objective_era5_ifs_dual_panel.png",
            plot_max_lead=args.plot_max_lead,
            figure_dpi=args.figure_dpi,
        )

    print(f"Wrote intermediate: {out_intermediate}")
    print(f"Wrote tables: {out_tables}")
    print(f"Wrote figures: {out_figures}")
    print(f"Rows objective={len(objective)} homogeneous={len(homogeneous)}")


if __name__ == "__main__":
    main()


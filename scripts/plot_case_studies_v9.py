#!/usr/bin/env python3
"""V9 case-study figure: V8 data + compact rows, in-panel (a) labels, col-2 date ticks once-labeled, bias ylim -4..2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.common.repo_paths import nwp_outputs_dir  # noqa: E402
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import xarray as xr
from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D

DATA_DIR = (
    nwp_outputs_dir()
    / "era5_monthly_202506_v2/metrics/heatwave_object_v2/ifs/case_studies_v7"
)
META_PATH = DATA_DIR / "case_meta.json"
OUT_FILE_PNG = DATA_DIR / "extreme_events_case_studies_v9_final.png"
OUT_FILE_PDF = DATA_DIR / "extreme_events_case_studies_v9_final.pdf"

CASE_ORDER = ["North_America", "East_Asia", "Australia"]
MODELS = ["aifs", "aurora", "fuxi", "fengwu", "pangu", "graphcast", "stormer"]

MODEL_COLORS = {
    "aifs": "#1f77b4",
    "aurora": "#ff7f0e",
    "fuxi": "#2ca02c",
    "fengwu": "#d62728",
    "pangu": "#9467bd",
    "graphcast": "#8c564b",
    "stormer": "#e377c2",
}

MODEL_NAMES = {
    "aifs": "AIFS",
    "aurora": "Aurora",
    "fuxi": "FuXi",
    "fengwu": "FengWu",
    "pangu": "Pangu-Weather",
    "graphcast": "GraphCast",
    "stormer": "Stormer",
}

_LETTER_BBOX = dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3)

_ROW0_TITLES = (
    "ERA5 Tmax & P90 boundary",
    "Tmax time evolution",
    "Model bias vs. lead time",
)


def load_cfg() -> dict:
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


def contour_field_masked_to_analysis_box(
    diff: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    a_lat: tuple[float, float],
    a_lon: tuple[float, float],
    *,
    sigma: float = 0.8,
) -> np.ndarray:
    exceed = (diff > 0).astype(np.uint8)
    labeled, _ = ndimage.label(exceed)

    lat_min, lat_max = min(a_lat), max(a_lat)
    lon_min, lon_max = min(a_lon), max(a_lon)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    in_box = (lat_grid >= lat_min) & (lat_grid <= lat_max) & (lon_grid >= lon_min) & (lon_grid <= lon_max)

    touching = np.unique(labeled[in_box & (labeled > 0)])
    if touching.size == 0:
        valid_blobs = exceed.astype(bool)
    else:
        valid_blobs = np.isin(labeled, touching)

    diff_smooth = ndimage.gaussian_filter(diff.astype(np.float64), sigma=sigma).astype(np.float32)
    return np.where(valid_blobs, diff_smooth, -999.0)


def plot() -> None:
    cfg_all = load_cfg()

    fig = plt.figure(figsize=(18, 15))
    # Slim spacer columns (~0.1 em) between map|TS and TS|bias; keeps inner gaps uniform.
    _gap_w = 0.020
    gs = gridspec.GridSpec(
        3,
        5,
        width_ratios=(1.0, _gap_w, 1.0, _gap_w, 1.0),
        wspace=0.11,
        hspace=0.22,
    )

    letters = "abcdefghi"
    im_last = None
    row0_axes: Optional[tuple] = None

    for i, case in enumerate(CASE_ORDER):
        cfg = cfg_all[case]

        ax1 = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
        ax_gap1 = fig.add_subplot(gs[i, 1])
        ax_gap1.set_axis_off()

        ds_map = xr.open_dataset(DATA_DIR / f"case_{case}_map_data.nc")
        tmax = ds_map["tmax_gt"].values.astype(np.float32)
        p90 = ds_map["p90"].values.astype(np.float32)
        lons = ds_map["longitude"].values.astype(np.float64)
        lats = ds_map["latitude"].values.astype(np.float64)

        diff = tmax - p90
        levels = np.linspace(15, 45, 16)
        im_last = ax1.contourf(
            lons,
            lats,
            tmax,
            levels=levels,
            cmap="Spectral_r",
            extend="both",
            transform=ccrs.PlateCarree(),
        )

        a_lat = tuple(cfg["analysis_box"]["lat"])
        a_lon = tuple(cfg["analysis_box"]["lon"])
        diff_plot = contour_field_masked_to_analysis_box(diff, lons, lats, a_lat, a_lon, sigma=0.8)
        ax1.contour(
            lons,
            lats,
            diff_plot,
            levels=[0],
            colors="darkred",
            linewidths=2.5,
            transform=ccrs.PlateCarree(),
        )

        ax1.add_patch(
            patches.Rectangle(
                (a_lon[0], a_lat[0]),
                a_lon[1] - a_lon[0],
                a_lat[1] - a_lat[0],
                linewidth=2.5,
                edgecolor="blue",
                facecolor="none",
                transform=ccrs.PlateCarree(),
                zorder=12,
            )
        )

        ax1.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax1.add_feature(cfeature.BORDERS, linestyle="-", linewidth=0.4, alpha=0.4)

        m_lat = cfg["map_box"]["lat"]
        m_lon = cfg["map_box"]["lon"]
        ax1.set_extent([m_lon[0], m_lon[1], m_lat[0], m_lat[1]], crs=ccrs.PlateCarree())

        gl = ax1.gridlines(draw_labels=True, linestyle="--", alpha=0.2)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 10, "weight": "bold"}
        gl.ylabel_style = {"size": 10, "weight": "bold"}

        ax1.text(
            -0.22,
            0.5,
            case.replace("_", " "),
            transform=ax1.transAxes,
            fontsize=22,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
        )
        ax1.text(
            0.03,
            0.96,
            f"({letters[i * 3]})",
            transform=ax1.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            bbox=_LETTER_BBOX,
            zorder=20,
        )

        ax2 = fig.add_subplot(gs[i, 2])
        ax_gap2 = fig.add_subplot(gs[i, 3])
        ax_gap2.set_axis_off()
        df_ts = pd.read_csv(DATA_DIR / f"case_{case}_timeseries.csv")
        df_ts["time"] = pd.to_datetime(df_ts["time"])

        ax2.axvspan(
            pd.to_datetime(cfg["hw_dates"][0]),
            pd.to_datetime(cfg["hw_dates"][1]),
            color="lightgray",
            alpha=0.4,
            zorder=0,
        )

        ax2.plot(df_ts["time"], df_ts["era5"], color="black", marker="o", markersize=4, linewidth=2.5, zorder=5)
        for model in MODELS:
            ax2.plot(df_ts["time"], df_ts[model], color=MODEL_COLORS[model], linewidth=1.8, alpha=0.8)

        ax2.xaxis.set_major_formatter(DateFormatter("%m-%d"))
        ax2.set_ylabel("Temperature (°C)", fontsize=14, fontweight="bold")
        ax2.tick_params(axis="both", labelsize=12)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", weight="bold")
        plt.setp(ax2.yaxis.get_majorticklabels(), weight="bold")

        if i == 2:
            ax2.set_xlabel("Date (MM-DD)", fontsize=15, fontweight="bold")
        else:
            ax2.set_xlabel("")

        ax2.text(
            0.03,
            0.96,
            f"({letters[i * 3 + 1]})",
            transform=ax2.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            bbox=_LETTER_BBOX,
            zorder=20,
        )
        ax3 = fig.add_subplot(gs[i, 4])
        df_bias = pd.read_csv(DATA_DIR / f"case_{case}_bias.csv")

        ax3.axhline(0, color="black", linewidth=1.5, zorder=1)
        ax3.axhline(2, color="black", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
        ax3.axhline(-2, color="black", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)

        for model in MODELS:
            ax3.plot(
                df_bias["lead_day"],
                df_bias[model],
                color=MODEL_COLORS[model],
                marker="s",
                markersize=5,
                linewidth=2.0,
            )

        ax3.set_xticks([1, 3, 7, 10])
        ax3.set_ylabel("Temperature bias (°C)", fontsize=14, fontweight="bold")
        ax3.set_ylim(-4, 2)
        ax3.tick_params(axis="both", labelsize=12)
        plt.setp(ax3.yaxis.get_majorticklabels(), weight="bold")

        if i != 2:
            ax3.set_xticklabels([])
            ax3.set_xlabel("")
        else:
            ax3.set_xticklabels([1, 3, 7, 10], weight="bold")
            ax3.set_xlabel("Lead time (days)", fontsize=15, fontweight="bold")

        ax3.text(
            0.03,
            0.96,
            f"({letters[i * 3 + 2]})",
            transform=ax3.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            bbox=_LETTER_BBOX,
            zorder=20,
        )
        ds_map.close()

        if i == 0:
            row0_axes = (ax1, ax2, ax3)

    # Larger top margin for column titles; larger bottom margin so legend sits below (h) x-labels.
    fig.subplots_adjust(left=0.108, right=0.98, bottom=0.19, top=0.88)

    if row0_axes is not None:
        fig.canvas.draw()
        # GeoAxes vs plain axes: align titles to the highest row-1 panel top edge.
        title_y = max(ax.get_position().y1 for ax in row0_axes) + 0.014
        for ax, title in zip(row0_axes, _ROW0_TITLES):
            pos = ax.get_position()
            cx = pos.x0 + 0.5 * pos.width
            fig.text(
                cx,
                title_y,
                title,
                ha="center",
                va="bottom",
                fontsize=20,
                fontweight="bold",
                transform=fig.transFigure,
            )

    pos_map = ax1.get_position()
    # Slightly above previous slot (~1/4 em) so cbar does not sit too low under (g).
    cax = fig.add_axes([pos_map.x0, pos_map.y0 - 0.038, pos_map.width, 0.015])
    cb = fig.colorbar(im_last, cax=cax, orientation="horizontal")
    cb.set_label("Tmax (°C)", fontsize=16, fontweight="bold")
    cb.ax.tick_params(labelsize=13)
    plt.setp(cb.ax.get_xticklabels(), weight="bold")

    era5_legend = Line2D([0], [0], color="black", marker="o", linewidth=2.5, markersize=6, label="ERA5 (GT)")
    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS[m], marker="s", linewidth=2.0, markersize=7, label=MODEL_NAMES[m])
        for m in MODELS
    ]
    fig.legend(
        handles=[era5_legend] + model_handles,
        loc="lower center",
        bbox_to_anchor=(0.545, 0.055),
        ncol=4,
        frameon=False,
        prop={"weight": "bold", "size": 16},
    )

    fig.savefig(OUT_FILE_PNG, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_FILE_PDF, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print(f"✅ Saved:\n  - {OUT_FILE_PNG}\n  - {OUT_FILE_PDF}")


if __name__ == "__main__":
    plot()

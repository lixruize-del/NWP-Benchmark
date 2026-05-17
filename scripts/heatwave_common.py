#!/usr/bin/env python3
"""Shared utilities for heatwave baseline build and object-v2 lead-timeseries (forecast scan, °C)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

CLIMATE_FILENAME_PATTERN = re.compile(r"era5\.2_metre_temperature\.(\d{8})\.nc$")
FORECAST_FILENAME_PATTERN = re.compile(r"(\d{4})-(\d{4})-(\d+)\.nc$")


@dataclass(frozen=True)
class DatedFile:
    dt: date
    path: Path


def parse_ymd(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def iter_climate_dated_files(climate_dir: Path) -> Iterable[DatedFile]:
    for path in sorted(climate_dir.glob("*.nc")):
        m = CLIMATE_FILENAME_PATTERN.match(path.name)
        if not m:
            continue
        dt = datetime.strptime(m.group(1), "%Y%m%d").date()
        yield DatedFile(dt=dt, path=path)


def select_dated_files(
    files: Sequence[DatedFile], start_dt: date, end_dt: date
) -> list[DatedFile]:
    return [f for f in files if start_dt <= f.dt <= end_dt]


def detect_t2m_var(ds: xr.Dataset, preferred: str = "") -> str:
    if preferred:
        if preferred not in ds.data_vars:
            raise ValueError(
                f"Variable '{preferred}' not found. Available: {list(ds.data_vars)}"
            )
        return preferred
    if "t2m" in ds.data_vars:
        return "t2m"
    if "temperature_2m_max" in ds.data_vars:
        return "temperature_2m_max"
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise ValueError(
        "Cannot auto-detect t2m variable name. "
        f"Available variables: {list(ds.data_vars)}"
    )


def to_celsius(arr: xr.DataArray) -> xr.DataArray:
    units = str(arr.attrs.get("units", "")).strip().lower()
    if units in {"k", "kelvin"}:
        out = arr - 273.15
        out.attrs["units"] = "degC"
        return out
    return arr


def compute_lat_weights(lat: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(lat.astype(np.float64)))
    w = np.where(np.isfinite(w), w, 0.0)
    return w


def area_fraction(mask: np.ndarray, lat_weights: np.ndarray) -> float:
    # mask shape: [lat, lon], lat_weights shape: [lat]
    valid = np.isfinite(mask)
    if not np.any(valid):
        return float("nan")
    weighted = np.where(valid, mask, 0.0) * lat_weights[:, None]
    denom = np.where(valid, 1.0, 0.0) * lat_weights[:, None]
    d = np.nansum(denom)
    if d <= 0:
        return float("nan")
    return float(np.nansum(weighted) / d)


def weighted_mean(field: np.ndarray, lat_weights: np.ndarray) -> float:
    valid = np.isfinite(field)
    if not np.any(valid):
        return float("nan")
    weighted = np.where(valid, field, 0.0) * lat_weights[:, None]
    denom = np.where(valid, 1.0, 0.0) * lat_weights[:, None]
    d = np.nansum(denom)
    if d <= 0:
        return float("nan")
    return float(np.nansum(weighted) / d)


def centroid_from_weights(
    weights_2d: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[float, float]:
    valid = np.isfinite(weights_2d) & (weights_2d > 0)
    if not np.any(valid):
        return float("nan"), float("nan")
    w = np.where(valid, weights_2d, 0.0)
    total = np.nansum(w)
    if total <= 0:
        return float("nan"), float("nan")
    lat_grid = np.repeat(lat[:, None], lon.size, axis=1)
    lon_grid = np.repeat(lon[None, :], lat.size, axis=0)
    c_lat = float(np.nansum(lat_grid * w) / total)
    c_lon = float(np.nansum(lon_grid * w) / total)
    return c_lat, c_lon


def bbox_from_mask(
    mask_2d: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[float, float, float, float]:
    valid = np.asarray(mask_2d).astype(bool)
    if not np.any(valid):
        return float("nan"), float("nan"), float("nan"), float("nan")
    lat_idx, lon_idx = np.where(valid)
    return (
        float(np.nanmin(lat[lat_idx])),
        float(np.nanmax(lat[lat_idx])),
        float(np.nanmin(lon[lon_idx])),
        float(np.nanmax(lon[lon_idx])),
    )


def extract_events_from_daily(
    df: pd.DataFrame,
    *,
    date_col: str,
    active_col: str,
    min_duration_days: int,
    strategy: str,
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Extract contiguous active-day events from a daily table.

    Returns per-event rows with event_id, start/end/duration and grouping keys.
    """
    group_cols = list(group_cols or [])
    base_cols = group_cols + [date_col, active_col]
    work = df[base_cols].copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.date
    work = work.sort_values(group_cols + [date_col]).reset_index(drop=True)

    out: list[dict] = []
    if not group_cols:
        groups = [((), work)]
    else:
        groups = list(work.groupby(group_cols, dropna=False))

    for key, g in groups:
        g = g.sort_values(date_col)
        if group_cols:
            if not isinstance(key, tuple):
                key = (key,)
            key_map = {group_cols[i]: key[i] for i in range(len(group_cols))}
        else:
            key_map = {}

        active = g[active_col].astype(bool).to_numpy()
        dates = g[date_col].to_numpy()
        event_start_idx: int | None = None
        for i, is_on in enumerate(active):
            if is_on and event_start_idx is None:
                event_start_idx = i
            if (not is_on or i == len(active) - 1) and event_start_idx is not None:
                end_idx = i if is_on and i == len(active) - 1 else i - 1
                start_dt = dates[event_start_idx]
                end_dt = dates[end_idx]
                duration = (end_dt - start_dt).days + 1
                if duration >= min_duration_days:
                    out.append(
                        {
                            **key_map,
                            "strategy": strategy,
                            "start_time": start_dt.isoformat(),
                            "end_time": end_dt.isoformat(),
                            "duration_days": int(duration),
                        }
                    )
                event_start_idx = None

    events = pd.DataFrame(out)
    if events.empty:
        return events
    events = events.sort_values(group_cols + ["start_time", "end_time"]).reset_index(drop=True)
    events["event_id"] = [
        f"{strategy}_{i+1:06d}" for i in range(events.shape[0])
    ]
    return events


def _connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity == 4:
        return ndimage.generate_binary_structure(2, 1)
    if connectivity == 8:
        return ndimage.generate_binary_structure(2, 2)
    raise ValueError("--connectivity must be one of {4, 8}")


def _label_components(mask_2d: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    structure = _connectivity_structure(connectivity)
    labels, ncomp = ndimage.label(mask_2d.astype(bool), structure=structure)
    return labels, int(ncomp)


def _match_components(
    prev_components: list[dict[str, Any]],
    cur_components: list[dict[str, Any]],
    *,
    min_overlap_frac: float,
) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by IoU and overlap fraction."""
    candidates: list[tuple[float, float, int, int]] = []
    for i_prev, p in enumerate(prev_components):
        pm = p["mask"]
        pcount = max(int(p["pixel_count"]), 1)
        for i_cur, c in enumerate(cur_components):
            cm = c["mask"]
            ccount = max(int(c["pixel_count"]), 1)
            inter = int(np.count_nonzero(pm & cm))
            if inter <= 0:
                continue
            union = pcount + ccount - inter
            iou = inter / union if union > 0 else 0.0
            overlap_frac = inter / min(pcount, ccount)
            if overlap_frac < min_overlap_frac:
                continue
            candidates.append((iou, overlap_frac, i_prev, i_cur))

    candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
    used_prev: set[int] = set()
    used_cur: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _, _, i_prev, i_cur in candidates:
        if i_prev in used_prev or i_cur in used_cur:
            continue
        used_prev.add(i_prev)
        used_cur.add(i_cur)
        matched.append((i_prev, i_cur))
    return matched


def extract_connected_events_from_daily_masks(
    daily_items: Sequence[dict[str, Any]],
    *,
    strategy: str,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_weights: np.ndarray,
    min_duration_days: int,
    connectivity: int,
    min_overlap_frac: float,
    min_component_pixels: int = 1,
    group_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[date, np.ndarray]]]:
    """Extract connected-component events from daily masks.

    Each `daily_items` row must contain:
      - `date`: date | YYYY-MM-DD
      - `mask`: 2D bool array [lat, lon]
    Optional fields:
      - `field`: daily t2m field (degC)
      - `exceed`: exceedance field (degC), e.g., max(field-p90, 0)
      - any keys from `group_cols` (e.g., init_time)
    """
    group_cols = list(group_cols or [])
    if min_duration_days <= 0:
        raise ValueError("--min-duration-days must be positive")
    if min_overlap_frac < 0:
        raise ValueError("--min-overlap-frac must be >= 0")
    if min_component_pixels <= 0:
        raise ValueError("--min-component-pixels must be positive")

    if not daily_items:
        return pd.DataFrame(), {}

    normalized: list[dict[str, Any]] = []
    for row in daily_items:
        if "date" not in row or "mask" not in row:
            raise ValueError("daily_items rows must contain keys: date, mask")
        d = row["date"]
        if not isinstance(d, date):
            d = pd.to_datetime(d).date()
        mask = np.asarray(row["mask"]).astype(bool)
        normalized.append({**row, "date": d, "mask": mask})

    if group_cols:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in normalized:
            gkey = tuple(row.get(c) for c in group_cols)
            grouped.setdefault(gkey, []).append(row)
    else:
        grouped = {(): normalized}

    interim_rows: list[dict[str, Any]] = []
    interim_masks: dict[tuple[Any, ...], dict[date, np.ndarray]] = {}

    for group_key, items in grouped.items():
        items = sorted(items, key=lambda r: r["date"])
        track_series: dict[int, list[dict[str, Any]]] = {}
        next_track_id = 1
        prev_date: date | None = None
        prev_components: list[dict[str, Any]] = []

        for item in items:
            cur_date: date = item["date"]
            mask: np.ndarray = item["mask"]
            field: np.ndarray | None = item.get("field")
            exceed: np.ndarray | None = item.get("exceed")

            labels, ncomp = _label_components(mask, connectivity)
            cur_components: list[dict[str, Any]] = []
            for cid in range(1, ncomp + 1):
                cmask = labels == cid
                pcount = int(np.count_nonzero(cmask))
                if pcount < min_component_pixels:
                    continue
                comp_area_frac = area_fraction(cmask.astype(np.float32), lat_weights)
                comp_center = centroid_from_weights(cmask.astype(np.float32), lat, lon)
                comp_bbox = bbox_from_mask(cmask, lat, lon)
                if field is not None:
                    field_vals = np.where(cmask, field, np.nan)
                    comp_mean_t2m = float(np.nanmean(field_vals))
                    comp_max_t2m = float(np.nanmax(field_vals))
                else:
                    comp_mean_t2m = float("nan")
                    comp_max_t2m = float("nan")

                if exceed is not None:
                    ex_vals = np.where(cmask & np.isfinite(exceed), exceed, np.nan)
                    comp_mean_ex = float(np.nanmean(ex_vals))
                    comp_max_ex = float(np.nanmax(ex_vals))
                else:
                    comp_mean_ex = float("nan")
                    comp_max_ex = float("nan")

                cur_components.append(
                    {
                        "date": cur_date,
                        "mask": cmask,
                        "pixel_count": pcount,
                        "area_fraction": comp_area_frac,
                        "center_lat": float(comp_center[0]),
                        "center_lon": float(comp_center[1]),
                        "bbox_lat_min": float(comp_bbox[0]),
                        "bbox_lat_max": float(comp_bbox[1]),
                        "bbox_lon_min": float(comp_bbox[2]),
                        "bbox_lon_max": float(comp_bbox[3]),
                        "mean_t2m_c": comp_mean_t2m,
                        "max_t2m_c": comp_max_t2m,
                        "mean_exceedance_p90_c": comp_mean_ex,
                        "max_exceedance_p90_c": comp_max_ex,
                    }
                )

            if prev_date is None or (cur_date - prev_date).days > 1:
                prev_components = []

            matched = _match_components(
                prev_components, cur_components, min_overlap_frac=min_overlap_frac
            )
            for i_prev, i_cur in matched:
                cur_components[i_cur]["track_id"] = prev_components[i_prev]["track_id"]

            for comp in cur_components:
                if "track_id" not in comp:
                    comp["track_id"] = next_track_id
                    next_track_id += 1
                tid = int(comp["track_id"])
                track_series.setdefault(tid, []).append(comp)

            prev_components = cur_components
            prev_date = cur_date

        key_map = {group_cols[i]: group_key[i] for i in range(len(group_cols))}
        for tid, days in track_series.items():
            days = sorted(days, key=lambda x: x["date"])
            duration = len(days)
            if duration < min_duration_days:
                continue
            start_dt = days[0]["date"]
            end_dt = days[-1]["date"]
            stack_weights = np.stack(
                [d["mask"].astype(np.float32) for d in days], axis=0
            )
            union_weights = np.nansum(stack_weights, axis=0)
            center_lat, center_lon = centroid_from_weights(union_weights, lat, lon)
            bbox_lat_min, bbox_lat_max, bbox_lon_min, bbox_lon_max = bbox_from_mask(
                union_weights > 0, lat, lon
            )
            area_fracs = [float(d["area_fraction"]) for d in days]
            row = {
                **key_map,
                "strategy": strategy,
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "duration_days": int(duration),
                "hot_day_count": int(duration),
                "center_lat": float(center_lat),
                "center_lon": float(center_lon),
                "mean_area_fraction": float(np.nanmean(area_fracs)),
                "max_area_fraction": float(np.nanmax(area_fracs)),
                "bbox_lat_min": float(bbox_lat_min),
                "bbox_lat_max": float(bbox_lat_max),
                "bbox_lon_min": float(bbox_lon_min),
                "bbox_lon_max": float(bbox_lon_max),
                "mean_t2m_c": float(np.nanmean([d["mean_t2m_c"] for d in days])),
                "max_t2m_c": float(np.nanmax([d["max_t2m_c"] for d in days])),
                "mean_exceedance_p90_c": float(
                    np.nanmean([d["mean_exceedance_p90_c"] for d in days])
                ),
                "max_exceedance_p90_c": float(
                    np.nanmax([d["max_exceedance_p90_c"] for d in days])
                ),
                "_group_key": group_key,
                "_track_id": int(tid),
            }
            interim_rows.append(row)
            interim_masks[(group_key, int(tid))] = {d["date"]: d["mask"] for d in days}

    events = pd.DataFrame(interim_rows)
    if events.empty:
        return events, {}
    events = events.sort_values(group_cols + ["start_time", "end_time"]).reset_index(drop=True)
    events["event_id"] = [f"{strategy}_{i+1:06d}" for i in range(events.shape[0])]

    event_masks: dict[str, dict[date, np.ndarray]] = {}
    for _, row in events.iterrows():
        key = (row["_group_key"], int(row["_track_id"]))
        event_masks[str(row["event_id"])] = interim_masks[key]
    events = events.drop(columns=["_group_key", "_track_id"])
    return events, event_masks

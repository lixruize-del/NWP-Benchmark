"""GraphCast operational runner on IFS np.25 inputs."""

from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path
from typing import Dict, List

import jax
import numpy as np
import pandas as pd
import xarray as xr

from src.common.data_reader_ifs import DEFAULT_ERA5_NPY_ROOT, Era5NpyLayout, load_npy_2d
from src.common.repo_paths import static_nc_path
from src.graphcast import inference_operational as op

DEFAULT_WEIGHTS_ROOT = Path(
    os.environ.get("NWP_WEIGHTS_ROOT", "/ecmwf-era5-datasets/nwp_bench/assets/weights")
)

_MODEL_BUNDLE_CACHE: dict[str, tuple] = {}
_PREDICTOR_CACHE: dict[str, object] = {}
_CHANNEL_NAMES_CACHE: dict[str, list[str]] = {}
_STATIC_CACHE: tuple[np.ndarray, np.ndarray] | None = None

DEFAULT_STATIC_NC = static_nc_path()

_SURFACE_LONG_TO_SHORT = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
}
_UPPER_LONG_TO_SHORT = {
    "geopotential": "z",
    "specific_humidity": "q",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "vertical_velocity": "w",
}


def _load_model_bundle(weights_root: Path):
    key = str(weights_root)
    if key in _MODEL_BUNDLE_CACHE:
        return _MODEL_BUNDLE_CACHE[key]

    weights_path = weights_root / "graphcast" / op.OPERATIONAL_CKPT
    with open(weights_path, "rb") as f:
        ckpt = op.checkpoint.load(f, op.graphcast.CheckPoint)

    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

    mean_path = op.NORM_DIR / "graphcast_stats_mean_by_level.nc"
    std_path = op.NORM_DIR / "graphcast_stats_stddev_by_level.nc"
    diff_path = op.NORM_DIR / "graphcast_stats_diffs_stddev_by_level.nc"
    diffs_raw = xr.load_dataset(diff_path, engine="netcdf4")
    mean_raw = xr.load_dataset(mean_path, engine="netcdf4")
    std_raw = xr.load_dataset(std_path, engine="netcdf4")
    levels = task_config.pressure_levels
    diffs_stddev = op._subset_stats_to_levels(diffs_raw, levels)
    mean_by_level = op._subset_stats_to_levels(mean_raw, levels)
    stddev_by_level = op._subset_stats_to_levels(std_raw, levels)

    bundle = (params, state, model_config, task_config, diffs_stddev, mean_by_level, stddev_by_level)
    _MODEL_BUNDLE_CACHE[key] = bundle
    return bundle


def _get_predictor(weights_root: Path):
    key = str(weights_root)
    if key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[key]
    (
        params,
        state,
        model_config,
        task_config,
        diffs_stddev,
        mean_by_level,
        stddev_by_level,
    ) = _load_model_bundle(weights_root)
    _PREDICTOR_CACHE[key] = op.build_predictor(
        params,
        state,
        model_config,
        task_config,
        diffs_stddev,
        mean_by_level,
        stddev_by_level,
    )
    return _PREDICTOR_CACHE[key]


def graphcast_operational_channel_names(*, weights_root: Path = DEFAULT_WEIGHTS_ROOT) -> List[str]:
    key = str(weights_root)
    if key in _CHANNEL_NAMES_CACHE:
        return list(_CHANNEL_NAMES_CACHE[key])
    (_, _, _, task_config, _, _, _) = _load_model_bundle(weights_root)
    _CHANNEL_NAMES_CACHE[key] = op.build_channel_mapping(task_config)
    return list(_CHANNEL_NAMES_CACHE[key])


def _load_static_fields(layout: Era5NpyLayout, init_time: datetime) -> tuple[np.ndarray, np.ndarray]:
    global _STATIC_CACHE
    if _STATIC_CACHE is not None:
        return _STATIC_CACHE

    if DEFAULT_STATIC_NC.exists():
        ds_static = xr.open_dataset(DEFAULT_STATIC_NC)
        rename_map = {"valid_time": "time", "latitude": "lat", "longitude": "lon"}
        ds_static = ds_static.rename(
            {k: v for k, v in rename_map.items() if k in ds_static.dims or k in ds_static.coords}
        )
        lat_name = "lat" if "lat" in ds_static.coords else "latitude"
        if ds_static[lat_name][0] < ds_static[lat_name][-1]:
            ds_static = ds_static.isel({lat_name: slice(None, None, -1)})
        lon_name = "lon" if "lon" in ds_static.coords else "longitude"
        lon = ds_static[lon_name].values
        if (lon < 0).any():
            lon_positive = lon % 360
            sort_idx = np.argsort(lon_positive)
            ds_static = ds_static.isel({lon_name: sort_idx})
            ds_static = ds_static.assign_coords({lon_name: lon_positive[sort_idx]})
        z_da = ds_static["z"].isel(time=0) if "time" in ds_static["z"].dims else ds_static["z"]
        lsm_da = ds_static["lsm"].isel(time=0) if "time" in ds_static["lsm"].dims else ds_static["lsm"]
        z_sfc = z_da.transpose("lat", "lon").astype(np.float32).values
        lsm = lsm_da.transpose("lat", "lon").astype(np.float32).values
        _STATIC_CACHE = (z_sfc, lsm)
        return _STATIC_CACHE

    lsm = load_npy_2d(layout.single_path(init_time, "lsm"), flip_north_south=False).astype(np.float32)
    z1000 = load_npy_2d(layout.pressure_path(init_time, "z", 1000.0), flip_north_south=False).astype(
        np.float32
    )
    t1000 = load_npy_2d(layout.pressure_path(init_time, "t", 1000.0), flip_north_south=False).astype(
        np.float32
    )
    sp = load_npy_2d(layout.single_path(init_time, "sp"), flip_north_south=False).astype(np.float32)
    rd = np.float32(287.05)
    p1000 = np.float32(100000.0)
    sp_safe = np.clip(sp, 1000.0, p1000)
    z_sfc = z1000 + rd * t1000 * np.log(p1000 / sp_safe)
    _STATIC_CACHE = (z_sfc.astype(np.float32), lsm.astype(np.float32))
    return _STATIC_CACHE


def _build_operational_input(init_time: datetime, era5_root: Path, task_config) -> xr.Dataset:
    layout = Era5NpyLayout(era5_root)
    times = [init_time - timedelta(hours=6), init_time]
    time_coords = np.array([np.datetime64(pd.Timestamp(t)) for t in times], dtype="datetime64[ns]")
    levels = [float(x) for x in task_config.pressure_levels]

    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon = np.linspace(0.0, 360.0, 1440, endpoint=False, dtype=np.float32)
    data_vars: Dict[str, xr.DataArray] = {}

    for long_name, short_name in _SURFACE_LONG_TO_SHORT.items():
        arr = np.stack(
            [load_npy_2d(layout.single_path(t, short_name), flip_north_south=False) for t in times], axis=0
        ).astype(np.float32)
        data_vars[long_name] = xr.DataArray(
            arr,
            dims=["time", "lat", "lon"],
            coords={"time": time_coords, "lat": lat, "lon": lon},
        )

    for long_name, short_name in _UPPER_LONG_TO_SHORT.items():
        arr = np.stack(
            [
                np.stack(
                    [
                        load_npy_2d(layout.pressure_path(t, short_name, lev), flip_north_south=False)
                        for lev in levels
                    ],
                    axis=0,
                )
                for t in times
            ],
            axis=0,
        ).astype(np.float32)
        data_vars[long_name] = xr.DataArray(
            arr,
            dims=["time", "level", "lat", "lon"],
            coords={
                "time": time_coords,
                "level": np.asarray(levels, dtype=np.float32),
                "lat": lat,
                "lon": lon,
            },
        )

    z_sfc, lsm = _load_static_fields(layout, init_time)
    data_vars["geopotential_at_surface"] = xr.DataArray(
        z_sfc.astype(np.float32), dims=["lat", "lon"], coords={"lat": lat, "lon": lon}
    )
    data_vars["land_sea_mask"] = xr.DataArray(
        lsm.astype(np.float32), dims=["lat", "lon"], coords={"lat": lat, "lon": lon}
    )

    ds = xr.Dataset(data_vars)
    ds = ds.assign_coords(datetime=("time", time_coords))
    op.data_utils.add_derived_vars(ds)
    return ds


def _pack_output(pred_ds: xr.Dataset, task_config) -> np.ndarray:
    pred = pred_ds.isel(batch=0, drop=True)
    atmo = set(op.graphcast.TARGET_ATMOSPHERIC_VARS)
    data_list: List[np.ndarray] = []
    for var in task_config.target_variables:
        da = pred[var]
        if "time" in da.dims:
            da = da.isel(time=-1)
        if var in atmo:
            for i in range(len(task_config.pressure_levels)):
                data_list.append(da.isel(level=i).values.astype(np.float32))
        else:
            data_list.append(da.values.astype(np.float32))
    return np.stack(data_list, axis=0).astype(np.float32)


def run_graphcast_operational_forecast(
    init_time: datetime,
    lead_times_hours: List[int],
    *,
    era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
    weights_root: Path = DEFAULT_WEIGHTS_ROOT,
) -> Dict[int, np.ndarray]:
    wanted = sorted({int(h) for h in lead_times_hours})
    if not wanted:
        return {}
    if any(h % op.STEP_HOURS != 0 for h in wanted):
        raise ValueError(f"lead_time_hours must be multiples of {op.STEP_HOURS}: {wanted}")

    steps = max(wanted) // op.STEP_HOURS
    (_, _, _, task_config, _, _, _) = _load_model_bundle(weights_root)
    ds_long = _build_operational_input(init_time, era5_root, task_config)
    init_str = init_time.strftime("%Y%m%d%H")
    inputs, forcings, targets_template = op.prepare_forcings_and_targets(
        ds_long, init_str, steps, task_config
    )
    run_forward_jitted = _get_predictor(weights_root)
    predictions = op.rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
    )

    out: Dict[int, np.ndarray] = {}
    for step_idx in range(steps):
        lead_hour = (step_idx + 1) * op.STEP_HOURS
        if lead_hour in wanted:
            step_ds = predictions.isel(time=step_idx, drop=False)
            out[lead_hour] = _pack_output(step_ds, task_config)
    return out


class GraphcastOperationalForecastRunner:
    def __init__(
        self,
        *,
        era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
        weights_root: Path = DEFAULT_WEIGHTS_ROOT,
    ) -> None:
        self.era5_root = era5_root
        self.weights_root = weights_root

    def run(self, init_time: datetime, lead_times_hours: List[int]) -> Dict[int, np.ndarray]:
        return run_graphcast_operational_forecast(
            init_time,
            lead_times_hours,
            era5_root=self.era5_root,
            weights_root=self.weights_root,
        )

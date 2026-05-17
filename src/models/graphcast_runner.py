"""GraphCast np.25 runner, strict home-script structure."""  # src:/home/NWP-Benchmark/src/graphcast/inference.py:1-7

from __future__ import annotations  # src:/home/NWP-Benchmark/src/graphcast/inference.py:1

import functools  # src:/home/NWP-Benchmark/src/graphcast/inference.py:5
import logging  # src:/home/NWP-Benchmark/src/graphcast/inference.py:3
import os  # src:/home/NWP-Benchmark/src/graphcast/inference.py:1
from datetime import datetime, timedelta  # src:/home/NWP-Benchmark/src/graphcast/inference.py:8
from pathlib import Path  # src:/home/NWP-Benchmark/src/graphcast/inference.py:7
from typing import Dict, List  # src:/home/NWP-Benchmark/src/graphcast/inference.py:1-8

import haiku as hk  # src:/home/NWP-Benchmark/src/graphcast/inference.py:14
import jax  # src:/home/NWP-Benchmark/src/graphcast/inference.py:13
import numpy as np  # src:/home/NWP-Benchmark/src/graphcast/inference.py:10
import pandas as pd  # src:/home/NWP-Benchmark/src/graphcast/inference.py:12
import xarray as xr  # src:/home/NWP-Benchmark/src/graphcast/inference.py:11
from graphcast import autoregressive, casting, checkpoint, data_utils, graphcast, normalization, rollout, solar_radiation  # src:/home/NWP-Benchmark/src/graphcast/inference.py:16-25

from src.common.data_reader import (  # src:/vepfs-dev/.../src/common/data_reader.py
    DEFAULT_ERA5_NPY_ROOT,
    Era5NpyLayout,
    load_era5_tp6h_depth_m,
    load_npy_2d,
)

logger = logging.getLogger(__name__)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:49

DEFAULT_WEIGHTS_ROOT = Path(os.environ.get("NWP_WEIGHTS_ROOT", "/ecmwf-era5-datasets/nwp_bench/assets/weights"))  # src:/home/NWP-Benchmark/src/graphcast/inference.py:35 adapted
NORM_DIR = Path(__file__).resolve().parents[1] / "graphcast" / "normalization_constants"  # src:/home/NWP-Benchmark/src/graphcast/inference.py:36
from src.common.repo_paths import static_nc_path

DEFAULT_STATIC_NC = static_nc_path()  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:91 static_file adapted
STEP_HOURS = 6  # src:/home/NWP-Benchmark/src/graphcast/inference.py:52
PRESSURE_LEVELS = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]  # src:/home/NWP-Benchmark/src/graphcast/inference.py:53-55
SURFACE_VARS_LONG_ORDERED = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "total_precipitation_6hr"]  # src:/home/NWP-Benchmark/src/graphcast/inference.py:57
UPPER_VARS_LONG_ORDERED = ["geopotential", "specific_humidity", "u_component_of_wind", "v_component_of_wind", "temperature", "vertical_velocity"]  # src:/home/NWP-Benchmark/src/graphcast/inference.py:58
LONG_TO_SAVER_SHORT = {"2m_temperature": "t2m", "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10", "mean_sea_level_pressure": "msl", "total_precipitation_6hr": "tp6h", "geopotential": "z", "specific_humidity": "q", "u_component_of_wind": "u", "v_component_of_wind": "v", "temperature": "t", "vertical_velocity": "w"}  # src:/home/NWP-Benchmark/src/graphcast/inference.py:60
SURFACE_SHORT = ["t2m", "u10", "v10", "msl", "tp6h"]  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:36-42 inverse mapping
UPPER_SHORT = ["z", "q", "u", "v", "t", "w"]  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:43-50 inverse mapping

_MODEL_BUNDLE_CACHE: dict[str, tuple] = {}
_PREDICTOR_CACHE: dict[str, object] = {}
_STATIC_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def graphcast_channel_names() -> List[str]:  # src:/home/NWP-Benchmark/src/graphcast/inference.py:200-212
    mapping: List[str] = []  # src:/home/NWP-Benchmark/src/graphcast/inference.py:202
    mapping.extend(["t2m", "u10", "v10", "msl", "tp6h"])  # src:/home/NWP-Benchmark/src/graphcast/inference.py:203-204
    for short in ["z", "q", "u", "v", "t", "w"]:  # src:/home/NWP-Benchmark/src/graphcast/inference.py:205-206
        for lev in PRESSURE_LEVELS:  # src:/home/NWP-Benchmark/src/graphcast/inference.py:207
            mapping.append(f"{short}_{lev}")  # src:/home/NWP-Benchmark/src/graphcast/inference.py:208
    return mapping  # src:/home/NWP-Benchmark/src/graphcast/inference.py:212


def _load_model(weights_root: Path):  # src:/home/NWP-Benchmark/src/graphcast/inference.py:68-91
    weights_path = weights_root / "graphcast" / "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"  # src:/home/NWP-Benchmark/src/graphcast/inference.py:70
    with open(weights_path, "rb") as f:  # src:/home/NWP-Benchmark/src/graphcast/inference.py:72
        ckpt = checkpoint.load(f, graphcast.CheckPoint)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:73
    params = ckpt.params  # src:/home/NWP-Benchmark/src/graphcast/inference.py:75
    state = {}  # src:/home/NWP-Benchmark/src/graphcast/inference.py:76
    model_config = ckpt.model_config  # src:/home/NWP-Benchmark/src/graphcast/inference.py:77
    task_config = ckpt.task_config  # src:/home/NWP-Benchmark/src/graphcast/inference.py:78
    diffs_stddev = xr.load_dataset(NORM_DIR / "graphcast_stats_diffs_stddev_by_level.nc", engine="netcdf4").compute()  # src:/home/NWP-Benchmark/src/graphcast/inference.py:87
    mean_by_level = xr.load_dataset(NORM_DIR / "graphcast_stats_mean_by_level.nc", engine="netcdf4").compute()  # src:/home/NWP-Benchmark/src/graphcast/inference.py:88
    stddev_by_level = xr.load_dataset(NORM_DIR / "graphcast_stats_stddev_by_level.nc", engine="netcdf4").compute()  # src:/home/NWP-Benchmark/src/graphcast/inference.py:89
    return params, state, model_config, task_config, diffs_stddev, mean_by_level, stddev_by_level  # src:/home/NWP-Benchmark/src/graphcast/inference.py:91


def _load_static_fields(static_nc: Path) -> tuple[np.ndarray, np.ndarray]:  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:91-123 static handling adapted
    key = str(static_nc)
    if key in _STATIC_CACHE:
        return _STATIC_CACHE[key]
    ds_static = xr.open_dataset(static_nc)  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:102
    rename_map = {"valid_time": "time", "latitude": "lat", "longitude": "lon"}  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:114-117
    ds_static = ds_static.rename({k: v for k, v in rename_map.items() if k in ds_static.dims or k in ds_static.coords})  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:117
    lat_name = "lat" if "lat" in ds_static.coords else "latitude"  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:68
    if ds_static[lat_name][0] < ds_static[lat_name][-1]:  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:69-70
        ds_static = ds_static.isel({lat_name: slice(None, None, -1)})  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:70
    lon_name = "lon" if "lon" in ds_static.coords else "longitude"  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:73
    lon = ds_static[lon_name].values  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:74
    if (lon < 0).any():  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:75
        lon_positive = lon % 360  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:76
        sort_idx = np.argsort(lon_positive)  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:77
        ds_static = ds_static.isel({lon_name: sort_idx})  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:78
        ds_static = ds_static.assign_coords({lon_name: lon_positive[sort_idx]})  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:79
    z_da = ds_static["z"].isel(time=0) if "time" in ds_static["z"].dims else ds_static["z"]  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:142
    lsm_da = ds_static["lsm"].isel(time=0) if "time" in ds_static["lsm"].dims else ds_static["lsm"]  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:142
    z = z_da.transpose("lat", "lon").astype(np.float32).values  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:143
    lsm = lsm_da.transpose("lat", "lon").astype(np.float32).values  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:143
    _STATIC_CACHE[key] = (z, lsm)
    return _STATIC_CACHE[key]  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:144


def _get_model_bundle(weights_root: Path):
    key = str(weights_root)
    if key not in _MODEL_BUNDLE_CACHE:
        _MODEL_BUNDLE_CACHE[key] = _load_model(weights_root)
    return _MODEL_BUNDLE_CACHE[key]


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
    ) = _get_model_bundle(weights_root)
    _PREDICTOR_CACHE[key] = _build_predictor(
        params,
        state,
        model_config,
        task_config,
        diffs_stddev,
        mean_by_level,
        stddev_by_level,
    )
    return _PREDICTOR_CACHE[key]


def _build_np25_input(init_time: datetime, era5_root: Path) -> xr.Dataset:  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:85-165 adapted to np.25
    layout = Era5NpyLayout(era5_root)  # src:/vepfs-dev/.../src/common/data_reader.py:60
    times = [init_time - timedelta(hours=6), init_time]  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:87
    time_coords = np.array([np.datetime64(pd.Timestamp(t)) for t in times], dtype="datetime64[ns]")  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:148,155
    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:163
    lon = np.linspace(0.0, 360.0, 1440, endpoint=False, dtype=np.float32)  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:164
    data_vars: Dict[str, xr.DataArray] = {}  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:129
    for short, long in zip(SURFACE_SHORT, SURFACE_VARS_LONG_ORDERED):  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:132-134
        if short == "tp6h":  # src:ERA5 6h accum in metres — not FuXi mm scaling
            arr = np.stack([load_era5_tp6h_depth_m(layout, t, flip_north_south=False) for t in times], axis=0).astype(np.float32)  # src:/data_reader.load_era5_tp6h_depth_m
        else:
            arr = np.stack([load_npy_2d(layout.single_path(t, short), flip_north_south=False) for t in times], axis=0).astype(np.float32)  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:133
        data_vars[long] = xr.DataArray(arr, dims=["time", "lat", "lon"], coords={"time": time_coords, "lat": lat, "lon": lon})  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:133
    for short, long in zip(UPPER_SHORT, UPPER_VARS_LONG_ORDERED):  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:136-139
        arr = np.stack([np.stack([load_npy_2d(layout.pressure_path(t, short, float(lev)), flip_north_south=False) for lev in PRESSURE_LEVELS], axis=0) for t in times], axis=0).astype(np.float32)  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:137-138
        data_vars[long] = xr.DataArray(arr, dims=["time", "level", "lat", "lon"], coords={"time": time_coords, "level": np.asarray(PRESSURE_LEVELS, dtype=np.int32), "lat": lat, "lon": lon})  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:138
    if not DEFAULT_STATIC_NC.exists():
        raise FileNotFoundError(
            f"GraphCast requires static fields at {DEFAULT_STATIC_NC}; "
            "place static.nc in the repository root."
        )
    z_sfc, lsm = _load_static_fields(DEFAULT_STATIC_NC)  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:102,141-144 adapted
    data_vars["geopotential_at_surface"] = xr.DataArray(z_sfc.astype(np.float32), dims=["lat", "lon"], coords={"lat": lat, "lon": lon})  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:141-144 static var add
    data_vars["land_sea_mask"] = xr.DataArray(lsm.astype(np.float32), dims=["lat", "lon"], coords={"lat": lat, "lon": lon})  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:141-144 static var add
    ds = xr.Dataset(data_vars)  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:152
    ds = ds.assign_coords(datetime=("time", time_coords))  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:155-156
    data_utils.add_derived_vars(ds)  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:159
    return ds  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:161-165


def _prepare_forcings_and_targets(ds: xr.Dataset, init_time: datetime, lead_steps: int, task_config):  # src:/home/NWP-Benchmark/src/graphcast/inference.py:101-167
    base_time = pd.Timestamp(init_time)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:110
    lead_times = pd.timedelta_range(start="6h", periods=lead_steps, freq="6h")  # src:/home/NWP-Benchmark/src/graphcast/inference.py:111
    target_times = [base_time + lt for lt in lead_times]  # src:/home/NWP-Benchmark/src/graphcast/inference.py:113
    target_time_coords = np.array([np.datetime64(t) for t in target_times])  # src:/home/NWP-Benchmark/src/graphcast/inference.py:114
    lat = ds.lat.values  # src:/home/NWP-Benchmark/src/graphcast/inference.py:116
    lon = ds.lon.values  # src:/home/NWP-Benchmark/src/graphcast/inference.py:117
    forcings_ds = xr.Dataset(coords={"time": target_time_coords, "lat": lat, "lon": lon})  # src:/home/NWP-Benchmark/src/graphcast/inference.py:119
    forcings_ds = forcings_ds.assign_coords(datetime=("time", target_time_coords))  # src:/home/NWP-Benchmark/src/graphcast/inference.py:120
    data_utils.add_derived_vars(forcings_ds)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:122
    toa_rad = solar_radiation.get_toa_incident_solar_radiation_for_xarray(forcings_ds, use_jit=True)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:124
    forcings_ds["toa_incident_solar_radiation"] = toa_rad.astype(np.float32)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:125
    forcing_vars = list(task_config.forcing_variables)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:127
    forcings = forcings_ds[forcing_vars].expand_dims(batch=1).astype(np.float32)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:128
    if "toa_incident_solar_radiation" not in ds:  # src:/home/NWP-Benchmark/src/graphcast/inference.py:130
        if "datetime" not in ds.coords:  # src:/home/NWP-Benchmark/src/graphcast/inference.py:131
            ds = ds.assign_coords(datetime=("time", ds.time.values))  # src:/home/NWP-Benchmark/src/graphcast/inference.py:132
        ds_toa = solar_radiation.get_toa_incident_solar_radiation_for_xarray(ds, use_jit=True)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:133
        ds["toa_incident_solar_radiation"] = ds_toa.astype(np.float32)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:134
    input_vars = list(task_config.input_variables)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:136
    ds = ds[input_vars]  # src:/home/NWP-Benchmark/src/graphcast/inference.py:137
    inputs = ds.expand_dims(batch=1)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:139
    target_vars = task_config.target_variables  # src:/home/NWP-Benchmark/src/graphcast/inference.py:142
    target_templates: Dict[str, xr.DataArray] = {}  # src:/home/NWP-Benchmark/src/graphcast/inference.py:143
    pressure_levels_array = np.array(PRESSURE_LEVELS)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:145
    for var in target_vars:  # src:/home/NWP-Benchmark/src/graphcast/inference.py:147
        if var in UPPER_VARS_LONG_ORDERED:  # src:/home/NWP-Benchmark/src/graphcast/inference.py:148
            da = xr.DataArray(np.full((len(target_time_coords), len(pressure_levels_array), len(lat), len(lon)), np.nan, dtype=np.float32), dims=["time", "level", "lat", "lon"], coords={"time": target_time_coords, "level": pressure_levels_array, "lat": lat, "lon": lon})  # src:/home/NWP-Benchmark/src/graphcast/inference.py:149-153
        else:  # src:/home/NWP-Benchmark/src/graphcast/inference.py:154
            da = xr.DataArray(np.full((len(target_time_coords), len(lat), len(lon)), np.nan, dtype=np.float32), dims=["time", "lat", "lon"], coords={"time": target_time_coords, "lat": lat, "lon": lon})  # src:/home/NWP-Benchmark/src/graphcast/inference.py:155-159
        target_templates[var] = da  # src:/home/NWP-Benchmark/src/graphcast/inference.py:160
    targets_template = xr.Dataset(target_templates).expand_dims(batch=1)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:162
    return inputs, forcings, targets_template  # src:/home/NWP-Benchmark/src/graphcast/inference.py:167


def _build_predictor(params, state, model_config, task_config, diffs_stddev, mean_by_level, stddev_by_level):  # src:/home/NWP-Benchmark/src/graphcast/inference.py:169-199
    def _construct_wrapped_graphcast(mc, tc):  # src:/home/NWP-Benchmark/src/graphcast/inference.py:171
        predictor = graphcast.GraphCast(mc, tc)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:172
        predictor = casting.Bfloat16Cast(predictor)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:173
        predictor = normalization.InputsAndResiduals(predictor, diffs_stddev_by_level=diffs_stddev, mean_by_level=mean_by_level, stddev_by_level=stddev_by_level)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:174-179
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:180
        return predictor  # src:/home/NWP-Benchmark/src/graphcast/inference.py:181
    @hk.transform_with_state  # src:/home/NWP-Benchmark/src/graphcast/inference.py:183
    def _run_forward(model_config, task_config, inputs, targets_template, forcings):  # src:/home/NWP-Benchmark/src/graphcast/inference.py:184
        predictor = _construct_wrapped_graphcast(model_config, task_config)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:185
        return predictor(inputs, targets_template=targets_template, forcings=forcings)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:186
    def _with_configs(fn):  # src:/home/NWP-Benchmark/src/graphcast/inference.py:188
        return functools.partial(fn, model_config=model_config, task_config=task_config)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:189
    def _with_params(fn):  # src:/home/NWP-Benchmark/src/graphcast/inference.py:191
        return functools.partial(fn, params=params, state=state)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:192
    def _drop_state(fn):  # src:/home/NWP-Benchmark/src/graphcast/inference.py:194
        return lambda **kw: fn(**kw)[0]  # src:/home/NWP-Benchmark/src/graphcast/inference.py:195
    run_forward_jitted = _drop_state(_with_params(jax.jit(_with_configs(_run_forward.apply))))  # src:/home/NWP-Benchmark/src/graphcast/inference.py:197
    return run_forward_jitted  # src:/home/NWP-Benchmark/src/graphcast/inference.py:198


def _pack_output(pred_ds: xr.Dataset) -> np.ndarray:  # src:/home/NWP-Benchmark/src/graphcast/inference.py:214-257
    pred = pred_ds.isel(batch=0, drop=True)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:226
    data_list: List[np.ndarray] = []  # src:/home/NWP-Benchmark/src/graphcast/inference.py:228
    for long_name in SURFACE_VARS_LONG_ORDERED:  # src:/home/NWP-Benchmark/src/graphcast/inference.py:231
        da = pred[long_name]  # src:/home/NWP-Benchmark/src/graphcast/inference.py:232
        if "time" in da.dims:  # src:/home/NWP-Benchmark/src/graphcast/inference.py:233
            da = da.isel(time=-1)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:234
        data_list.append(da.values.astype(np.float32))  # src:/home/NWP-Benchmark/src/graphcast/inference.py:235
    for long_name in UPPER_VARS_LONG_ORDERED:  # src:/home/NWP-Benchmark/src/graphcast/inference.py:238
        da = pred[long_name]  # src:/home/NWP-Benchmark/src/graphcast/inference.py:239
        if "time" in da.dims:  # src:/home/NWP-Benchmark/src/graphcast/inference.py:240
            da = da.isel(time=-1)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:241
        for i, _lev in enumerate(PRESSURE_LEVELS):  # src:/home/NWP-Benchmark/src/graphcast/inference.py:242-244
            data_list.append(da.isel(level=i).values.astype(np.float32))  # src:/home/NWP-Benchmark/src/graphcast/inference.py:243
    out_array = np.stack(data_list, axis=0)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:247
    return out_array.astype(np.float32)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:247


def run_graphcast_forecast(init_time: datetime, lead_times_hours: List[int], *, era5_root: Path = DEFAULT_ERA5_NPY_ROOT) -> Dict[int, np.ndarray]:  # src:/vepfs-dev/.../run_large_scale.py:163 contract
    wanted = sorted({int(h) for h in lead_times_hours})  # src:/home/NWP-Benchmark/src/graphcast/inference.py:263-264
    if not wanted:  # src:empty call compatibility
        return {}  # src:empty output
    if any(h % STEP_HOURS != 0 for h in wanted):  # src:/home/NWP-Benchmark/src/graphcast/inference.py:261-262
        raise ValueError(f"lead_time_hours must be multiples of {STEP_HOURS}: {wanted}")  # src:/home/NWP-Benchmark/src/graphcast/inference.py:262
    steps = max(wanted) // STEP_HOURS  # src:/home/NWP-Benchmark/src/graphcast/inference.py:263
    (
        _params,
        _state,
        _model_config,
        task_config,
        _diffs_stddev,
        _mean_by_level,
        _stddev_by_level,
    ) = _get_model_bundle(DEFAULT_WEIGHTS_ROOT)
    ds_long = _build_np25_input(init_time, era5_root)  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:85-165 adapted to np.25
    inputs, forcings, targets_template = _prepare_forcings_and_targets(ds_long, init_time, steps, task_config)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:275
    run_forward_jitted = _get_predictor(DEFAULT_WEIGHTS_ROOT)
    predictions = rollout.chunked_prediction(run_forward_jitted, rng=jax.random.PRNGKey(0), inputs=inputs, targets_template=targets_template, forcings=forcings)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:283-289
    out: Dict[int, np.ndarray] = {}  # src:adapter output
    for step_idx in range(steps):  # src:/home/NWP-Benchmark/src/graphcast/inference.py:295
        lead_hour = (step_idx + 1) * STEP_HOURS  # src:/home/NWP-Benchmark/src/graphcast/inference.py:296
        if lead_hour in wanted:  # src:select requested leads
            step_ds = predictions.isel(time=step_idx, drop=False)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:298
            out[lead_hour] = _pack_output(step_ds)  # src:/home/NWP-Benchmark/src/graphcast/inference.py:299
    return out  # src:/vepfs adapter contract


class GraphcastForecastRunner:
    """Class-style GraphCast runner with process-level cached model bundle/predictor."""

    def __init__(self, *, era5_root: Path = DEFAULT_ERA5_NPY_ROOT) -> None:
        self.era5_root = era5_root

    def run(self, init_time: datetime, lead_times_hours: List[int]) -> Dict[int, np.ndarray]:
        return run_graphcast_forecast(init_time, lead_times_hours, era5_root=self.era5_root)

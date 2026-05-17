"""
GraphCast operational inference (13 WeatherBench levels, HRES-prepared input).
Uses checkpoint task_config for pressure levels and target order.
Normalization stats: same NetCDF bytes as gs://dm_graphcast/graphcast/stats/
(mean_by_level.nc etc.); this repo stores them as graphcast_stats_*.nc in
normalization_constants/ (identical content, different filenames).
"""
import functools
import logging
import argparse
import sys
import traceback
from pathlib import Path
import datetime as dt

import numpy as np
import xarray as xr
import pandas as pd
import jax
import haiku as hk

from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
    rollout,
    solar_radiation,
)

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from src.common.saver import Saver

WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "graphcast"
# Official GCS names are mean_by_level.nc; repo uses graphcast_stats_* (same files, verified identical MD5).
NORM_DIR = CURRENT_DIR / "normalization_constants"
PROCESSED_DATA_DIR = BASE_DIR / "assets" / "data" / "processed_graphcast_operational"
OUTPUT_DIR = BASE_DIR / "outputs" / "graphcast_operational"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

OPERATIONAL_CKPT = (
    "GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - "
    "pressure levels 13 - mesh 2to6 - precipitation output only.npz"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GraphCast.InferenceOperational")

STEP_HOURS = 6

LONG_TO_SAVER_SHORT = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "total_precipitation_6hr": "tp6h",
    "geopotential": "z",
    "specific_humidity": "q",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "vertical_velocity": "w",
}


def get_target_date():
    if DATE_FILE.exists():
        return DATE_FILE.read_text().strip()
    logger.warning("DATE_FILE not found, using current time.")
    return dt.datetime.now().strftime("%Y%m%d%H")


def _subset_stats_to_levels(ds: xr.Dataset, levels: tuple) -> xr.Dataset:
    """Colab stats have level dim 37; operational model uses 13 — subset."""
    lev_list = list(levels)
    out = xr.Dataset(attrs=ds.attrs)
    for v in ds.data_vars:
        da = ds[v]
        if "level" in da.dims:
            out[v] = da.sel(level=lev_list)
        else:
            out[v] = da
    return out.compute()


def load_model():
    weights_path = WEIGHTS_DIR / OPERATIONAL_CKPT
    logger.info("Loading weights %s", weights_path)
    with open(weights_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)

    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

    logger.info("input_variables: %s", task_config.input_variables)
    logger.info("target_variables: %s", task_config.target_variables)
    logger.info("pressure_levels: %s", task_config.pressure_levels)

    mean_path = NORM_DIR / "graphcast_stats_mean_by_level.nc"
    std_path = NORM_DIR / "graphcast_stats_stddev_by_level.nc"
    diff_path = NORM_DIR / "graphcast_stats_diffs_stddev_by_level.nc"
    if not mean_path.exists():
        raise FileNotFoundError(
            f"Missing {mean_path}. Same bytes as GCS mean_by_level.nc; use graphcast_stats_* names in repo."
        )

    diffs_raw = xr.load_dataset(diff_path, engine="netcdf4")
    mean_raw = xr.load_dataset(mean_path, engine="netcdf4")
    std_raw = xr.load_dataset(std_path, engine="netcdf4")

    levels = task_config.pressure_levels
    diffs_stddev = _subset_stats_to_levels(diffs_raw, levels)
    mean_by_level = _subset_stats_to_levels(mean_raw, levels)
    stddev_by_level = _subset_stats_to_levels(std_raw, levels)

    nlev = len(diffs_stddev.level) if "level" in diffs_stddev.dims else None
    if nlev != len(levels):
        raise ValueError(f"After subset, expected {len(levels)} levels, got {nlev}")

    return params, state, model_config, task_config, diffs_stddev, mean_by_level, stddev_by_level


def load_data(target_date: str):
    path = PROCESSED_DATA_DIR / f"input_{target_date}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    return xr.load_dataset(path, decode_timedelta=False)


def prepare_forcings_and_targets(ds: xr.Dataset, target_date: str, lead_steps: int, task_config):
    base_time = pd.to_datetime(target_date, format="%Y%m%d%H")
    lead_times = pd.timedelta_range(start="6h", periods=lead_steps, freq="6h")
    target_times = [base_time + lt for lt in lead_times]
    target_time_coords = np.array([np.datetime64(t) for t in target_times])

    lat = ds.lat.values
    lon = ds.lon.values
    pressure_levels_array = np.array(task_config.pressure_levels)

    forcings_ds = xr.Dataset(coords={"time": target_time_coords, "lat": lat, "lon": lon})
    forcings_ds = forcings_ds.assign_coords(datetime=("time", target_time_coords))
    data_utils.add_derived_vars(forcings_ds)
    toa_rad = solar_radiation.get_toa_incident_solar_radiation_for_xarray(forcings_ds, use_jit=True)
    forcings_ds["toa_incident_solar_radiation"] = toa_rad.astype(np.float32)
    forcing_vars = list(task_config.forcing_variables)
    forcings = forcings_ds[forcing_vars].expand_dims(batch=1).astype(np.float32)

    if "toa_incident_solar_radiation" not in ds:
        if "datetime" not in ds.coords:
            ds = ds.assign_coords(datetime=("time", ds.time.values))
        ds_toa = solar_radiation.get_toa_incident_solar_radiation_for_xarray(ds, use_jit=True)
        ds["toa_incident_solar_radiation"] = ds_toa.astype(np.float32)

    input_vars = list(task_config.input_variables)
    ds_in = ds[input_vars]
    inputs = ds_in.expand_dims(batch=1)

    target_vars = task_config.target_variables
    target_templates = {}
    atmo = set(graphcast.TARGET_ATMOSPHERIC_VARS)

    for var in target_vars:
        if var in atmo:
            da = xr.DataArray(
                np.full(
                    (len(target_time_coords), len(pressure_levels_array), len(lat), len(lon)),
                    np.nan,
                    dtype=np.float32,
                ),
                dims=["time", "level", "lat", "lon"],
                coords={
                    "time": target_time_coords,
                    "level": pressure_levels_array,
                    "lat": lat,
                    "lon": lon,
                },
            )
        else:
            da = xr.DataArray(
                np.full((len(target_time_coords), len(lat), len(lon)), np.nan, dtype=np.float32),
                dims=["time", "lat", "lon"],
                coords={"time": target_time_coords, "lat": lat, "lon": lon},
            )
        target_templates[var] = da

    targets_template = xr.Dataset(target_templates).expand_dims(batch=1)
    return inputs, forcings, targets_template


def build_predictor(params, state, model_config, task_config, diffs_stddev, mean_by_level, stddev_by_level):
    def _construct_wrapped_graphcast(mc, tc):
        predictor = graphcast.GraphCast(mc, tc)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def _run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = _construct_wrapped_graphcast(model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def _with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)

    def _with_params(fn):
        return functools.partial(fn, params=params, state=state)

    def _drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    return _drop_state(_with_params(jax.jit(_with_configs(_run_forward.apply))))


def build_channel_mapping(task_config) -> list:
    atmo = set(graphcast.TARGET_ATMOSPHERIC_VARS)
    mapping = []
    for var in task_config.target_variables:
        if var in atmo:
            short = LONG_TO_SAVER_SHORT[var]
            for lev in task_config.pressure_levels:
                mapping.append(f"{short}_{lev}")
        else:
            mapping.append(LONG_TO_SAVER_SHORT[var])
    return mapping


def process_and_save_output(
    pred_ds: xr.Dataset,
    init_time_str: str,
    lead_time_hours: int,
    saver: Saver,
    task_config,
    channel_mapping: list,
):
    pred = pred_ds.isel(batch=0, drop=True)
    atmo = set(graphcast.TARGET_ATMOSPHERIC_VARS)
    data_list = []

    for var in task_config.target_variables:
        da = pred[var]
        if "time" in da.dims:
            da = da.isel(time=-1)
        if var in atmo:
            for i in range(len(task_config.pressure_levels)):
                data_list.append(da.isel(level=i).values.astype(np.float32))
        else:
            data_list.append(da.values.astype(np.float32))

    out_array = np.stack(data_list, axis=0)
    saver.save(
        data=out_array,
        channel_mapping=channel_mapping,
        init_time_str=init_time_str,
        lead_time_hours=lead_time_hours,
        lat_values=pred.lat.values,
        lon_values=pred.lon.values,
    )


def run_inference(target_date: str, lead_time_hours: int):
    if lead_time_hours % STEP_HOURS != 0:
        raise ValueError(f"lead_time_hours must be a multiple of {STEP_HOURS}.")
    steps = lead_time_hours // STEP_HOURS

    logger.info("GraphCast operational: date %s, %s steps (%sh)", target_date, steps, lead_time_hours)

    params, state, model_config, task_config, diffs_stddev, mean_by_level, stddev_by_level = load_model()
    ds_long = load_data(target_date)
    logger.info("Dataset vars: %s", list(ds_long.data_vars))

    inputs, forcings, targets_template = prepare_forcings_and_targets(ds_long, target_date, steps, task_config)
    run_forward_jitted = build_predictor(
        params, state, model_config, task_config, diffs_stddev, mean_by_level, stddev_by_level
    )

    logger.info("Rollout…")
    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
    )

    saver = Saver(save_root=str(OUTPUT_DIR))
    channel_mapping = build_channel_mapping(task_config)

    for step_idx in range(steps):
        lead_hour = (step_idx + 1) * STEP_HOURS
        logger.info("Saving +%sh", lead_hour)
        step_ds = predictions.isel(time=step_idx, drop=False)
        process_and_save_output(step_ds, target_date, lead_hour, saver, task_config, channel_mapping)

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphCast operational inference")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--lead-time", type=int, default=6)
    args = parser.parse_args()
    target = args.date if args.date else get_target_date()
    try:
        run_inference(target, args.lead_time)
    except Exception as e:
        logger.error("%s", e)
        logger.debug(traceback.format_exc())
        sys.exit(1)

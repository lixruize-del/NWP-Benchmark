import os
import sys
import logging
import argparse
import functools
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

# --- Path Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from src.common.saver import Saver

# --- Configuration ---
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "graphcast"
NORM_DIR = CURRENT_DIR / "normalization_constants"
PROCESSED_DATA_DIR = BASE_DIR / "assets" / "data" / "processed_graphcast"
OUTPUT_DIR = BASE_DIR / "outputs" / "graphcast"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GraphCast.Inference")

# --- Constants ---
STEP_HOURS = 6
PRESSURE_LEVELS = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
                   225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
                   775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]

SURFACE_VARS_LONG_ORDERED = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'total_precipitation_6hr']
UPPER_VARS_LONG_ORDERED = ['geopotential', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'vertical_velocity']

LONG_TO_SAVER_SHORT = {'2m_temperature': 't2m', '10m_u_component_of_wind': 'u10', '10m_v_component_of_wind': 'v10', 'mean_sea_level_pressure': 'msl', 'total_precipitation_6hr': 'tp6h', 'geopotential': 'z', 'specific_humidity': 'q', 'u_component_of_wind': 'u', 'v_component_of_wind': 'v', 'temperature': 't', 'vertical_velocity': 'w'}

def get_target_date():
    """Read target date from DATE_FILE (format YYYYMMDDHH)."""
    if DATE_FILE.exists(): return DATE_FILE.read_text().strip()
    logger.warning("DATE_FILE not found, using current time.")
    return dt.datetime.now().strftime("%Y%m%d%H")

def load_model():
    """Load GraphCast checkpoint and normalisation statistics."""
    weights_path = WEIGHTS_DIR / "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
    logger.info(f"Loading model weights from {weights_path}")
    with open(weights_path, 'rb') as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)

    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

    logger.info(f"input_variables: {task_config.input_variables}")
    logger.info(f"forcing_variables: {task_config.forcing_variables}")
    logger.info(f"target_variables: {task_config.target_variables}")
    # logger.info(f"pressure_levels: {task_config.pressure_levels}")
    logger.info(f"pressure_levels: {PRESSURE_LEVELS}")

    logger.info("Loading normalisation statistics...")
    diffs_stddev = xr.load_dataset(NORM_DIR / "graphcast_stats_diffs_stddev_by_level.nc", engine="netcdf4").compute()
    mean_by_level = xr.load_dataset(NORM_DIR / "graphcast_stats_mean_by_level.nc", engine="netcdf4").compute()
    stddev_by_level = xr.load_dataset(NORM_DIR / "graphcast_stats_stddev_by_level.nc", engine="netcdf4").compute()

    return params, state, model_config, task_config, diffs_stddev, mean_by_level, stddev_by_level

def load_data(target_date: str):
    """Load the processed input NetCDF (two time steps)."""
    input_file = PROCESSED_DATA_DIR / f"input_{target_date}.nc"
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    ds = xr.load_dataset(input_file)
    return ds

def prepare_forcings_and_targets(ds: xr.Dataset, target_date: str, lead_steps: int, task_config):
    """
    Build forcings and targets_template for the required lead times.
    The input `ds` already uses long variable names.
    Returns:
        inputs: Dataset with batch dimension added.
        forcings: Dataset with batch dimension.
        targets_template: Dataset with NaNs for future steps, with batch dimension.
    """
    base_time = pd.to_datetime(target_date, format="%Y%m%d%H")
    lead_times = pd.timedelta_range(start='6h', periods=lead_steps, freq='6h')

    target_times = [base_time + lt for lt in lead_times]
    target_time_coords = np.array([np.datetime64(t) for t in target_times])

    lat = ds.lat.values
    lon = ds.lon.values

    forcings_ds = xr.Dataset(coords={'time': target_time_coords, 'lat': lat, 'lon': lon})
    forcings_ds = forcings_ds.assign_coords(datetime=('time', target_time_coords))

    data_utils.add_derived_vars(forcings_ds)

    toa_rad = solar_radiation.get_toa_incident_solar_radiation_for_xarray(forcings_ds, use_jit=True)
    forcings_ds['toa_incident_solar_radiation'] = toa_rad.astype(np.float32)

    forcing_vars = list(task_config.forcing_variables)
    forcings = forcings_ds[forcing_vars].expand_dims(batch=1).astype(np.float32)

    if 'toa_incident_solar_radiation' not in ds:
        if 'datetime' not in ds.coords:
            ds = ds.assign_coords(datetime=('time', ds.time.values))
        ds_toa = solar_radiation.get_toa_incident_solar_radiation_for_xarray(ds, use_jit=True)
        ds['toa_incident_solar_radiation'] = ds_toa.astype(np.float32)

    input_vars = list(task_config.input_variables)
    ds = ds[input_vars]

    inputs = ds.expand_dims(batch=1)

    # --- Targets template (filled with NaN) ---
    target_vars = task_config.target_variables
    target_templates = {}
    # pressure_levels_array = np.array(task_config.pressure_levels)
    pressure_levels_array = np.array(PRESSURE_LEVELS)

    for var in target_vars:
        if var in UPPER_VARS_LONG_ORDERED:
            da = xr.DataArray(
                np.full((len(target_time_coords), len(pressure_levels_array), len(lat), len(lon)), np.nan, dtype=np.float32),
                dims=['time', 'level', 'lat', 'lon'],
                coords={'time': target_time_coords, 'level': pressure_levels_array, 'lat': lat, 'lon': lon}
            )
        else:
            da = xr.DataArray(
                np.full((len(target_time_coords), len(lat), len(lon)), np.nan, dtype=np.float32),
                dims=['time', 'lat', 'lon'],
                coords={'time': target_time_coords, 'lat': lat, 'lon': lon}
            )
        target_templates[var] = da
        
    targets_template = xr.Dataset(target_templates).expand_dims(batch=1)

    # Add batch dimension (size 1)
    inputs = ds.expand_dims(batch=1)

    return inputs, forcings, targets_template

def build_predictor(params, state, model_config, task_config, diffs_stddev, mean_by_level, stddev_by_level):
    """Construct the jitted prediction function (similar to original prediction.py)."""
    def _construct_wrapped_graphcast(mc, tc):
        predictor = graphcast.GraphCast(mc, tc)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level
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

    run_forward_jitted = _drop_state(_with_params(jax.jit(_with_configs(_run_forward.apply))))
    return run_forward_jitted

def generate_channel_mapping() -> list:
    """Create channel names in the order expected by Saver (surface then upper with levels)."""
    mapping =[]
    for long_name in SURFACE_VARS_LONG_ORDERED:
        mapping.append(LONG_TO_SAVER_SHORT[long_name])
    for long_name in UPPER_VARS_LONG_ORDERED:
        short_name = LONG_TO_SAVER_SHORT[long_name]
        for lev in PRESSURE_LEVELS:
            mapping.append(f"{short_name}_{lev}")
            
    if len(mapping) != 5 + 6 * 37:
        raise RuntimeError(f"Channel mapping length mismatch: {len(mapping)}")
    return mapping

def process_and_save_output(pred_ds: xr.Dataset,
                            init_time_str: str,
                            lead_time_hours: int,
                            saver: Saver,
                            channel_mapping: list):
    """
    Extract a single forecast step from the prediction Dataset and save via Saver.
    pred_ds: xarray Dataset with dimensions (batch, time, level, lat, lon) or (batch, time, lat, lon)
             (batch dimension is 1, time dimension is the lead time index).
    The variables in pred_ds are long names; we map them back to short order for saving.
    """
    # Remove batch dimension
    pred = pred_ds.isel(batch=0, drop=True)   # now time is the forecast step

    data_list = []

    # Surface variables
    for long_name in SURFACE_VARS_LONG_ORDERED:
        da = pred[long_name]
        if 'time' in da.dims:
            da = da.isel(time=-1)
        data_list.append(da.values.astype(np.float32))

    # Upper variables
    for long_name in UPPER_VARS_LONG_ORDERED:
        da = pred[long_name]
        if 'time' in da.dims:
            da = da.isel(time=-1)
        for i, lev in enumerate(PRESSURE_LEVELS):
            level_slice = da.isel(level=i).values.astype(np.float32)
            data_list.append(level_slice)

    # Stack into [C, H, W]
    out_array = np.stack(data_list, axis=0)

    # Save using Saver (channel_mapping already contains short names with levels)
    saver.save(
        data=out_array,
        channel_mapping=channel_mapping,
        init_time_str=init_time_str,
        lead_time_hours=lead_time_hours,
        lat_values=pred.lat.values,
        lon_values=pred.lon.values,
    )

def run_inference(target_date: str, lead_time_hours: int):
    """Main inference routine."""
    if lead_time_hours % STEP_HOURS != 0:
        raise ValueError(f"lead_time_hours must be a multiple of {STEP_HOURS}.")
    steps = lead_time_hours // STEP_HOURS

    logger.info(f"🚀 GraphCast inference for date {target_date}, {steps} steps ({lead_time_hours}h)")

    # 1. Load model and stats
    params, state, model_config, task_config, diffs_stddev, mean_by_level, stddev_by_level = load_model()

    # 2. Load input data (short names)
    ds_long = load_data(target_date)
    logger.info(f"Using Input variables: {list(ds_long.data_vars)}")

    # 3. Prepare forcings and targets_template
    inputs, forcings, targets_template = prepare_forcings_and_targets(ds_long, target_date, steps, task_config)

    # 4. Build predictor
    run_forward_jitted = build_predictor(params, state, model_config, task_config,
                                         diffs_stddev, mean_by_level, stddev_by_level)

    # 5. Run autoregressive rollout
    logger.info("Starting autoregressive rollout...")
    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings
    )

    # 6. Save each lead step using Saver
    saver = Saver(save_root=str(OUTPUT_DIR))
    channel_mapping = generate_channel_mapping()

    for step_idx in range(steps):
        lead_hour = (step_idx + 1) * STEP_HOURS
        logger.info(f"Saving step +{lead_hour}h")
        step_ds = predictions.isel(time=step_idx, drop=False)
        process_and_save_output(step_ds, target_date, lead_hour, saver, channel_mapping)

    logger.info("✅ Inference completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphCast Inference (Saver-compatible)")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYYMMDDHH")
    parser.add_argument("--lead-time", type=int, default=6, help="Total forecast hours (multiple of 6)")
    args = parser.parse_args()

    target = args.date if args.date else get_target_date()
    try:
        run_inference(target, args.lead_time)
    except Exception as e:
        logger.error(f"Program terminated with error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
import os
import sys
import logging
import argparse
import pickle
import numpy as np
import xarray as xr
import jax
import neuralgcm
from pathlib import Path
import pandas as pd
from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("NeuralGCM.Inference")

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
PROCESSED_DIR = BASE_DIR / "assets" / "data" / "processed_neuralgcm"
RAW_DATA_DIR = BASE_DIR / "assets" / "data" / "raw"
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "neuralgcm"
OUTPUT_DIR = BASE_DIR / "outputs" / "neuralgcm"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.append(str(BASE_DIR))

from src.common.saver import Saver


grib_para = {
    "z": {"Unit": "m^2 s^-2"},
    "t": {"Unit": "K"},
    "u": {"Unit": "m s^-1"},
    "v": {"Unit": "m s^-1"},
}
var_map_inv = {
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
}

def standardize_latlon_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize latitude/longitude coordinates for interpolation/regridding.

    - Ensures latitude is monotonically increasing (-90 -> 90).
    - Ensures longitude is within [0, 360) and monotonically increasing.
    """
    if "longitude" in ds.coords:
        lon = ds["longitude"]
        if float(lon.min()) < 0.0:
            ds = ds.assign_coords(longitude=((lon + 360.0) % 360.0))
        ds = ds.sortby("longitude")

    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")

    return ds


# ==============================================================================
# Metric Calculation Utilities
# ==============================================================================
def compute_latitude_weights(latitudes: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(latitudes))
    weights = weights / weights.mean()
    weights.name = "latitude_weights"
    return weights


def calculate_wrmse(pred: xr.DataArray, gt: xr.DataArray) -> float:
    lat_coord_name = "latitude" if "latitude" in gt.coords else "lat"
    weights = compute_latitude_weights(gt[lat_coord_name])
    diff_sq = (pred - gt) ** 2
    return np.sqrt(diff_sq.weighted(weights).mean()).item()


# ==============================================================================
# Core Evaluation Function
# ==============================================================================
def evaluate_errors(
    pred_native_res: xr.Dataset,
    model: neuralgcm.PressureLevelModel,
    target_date: str,
    lead_time_hours: int,
):
    logger.info("=" * 80)
    logger.info("Starting Pre- and Post-Interpolation WRMSE Evaluation")
    logger.info("=" * 80)

    valid_time = pd.to_datetime(target_date, format="%Y%m%d%H") + pd.Timedelta(hours=lead_time_hours)
    valid_time_str = valid_time.strftime("%Y%m%d%H")

    gt_upper_path = RAW_DATA_DIR / f"upper_{valid_time_str}.nc"
    if not gt_upper_path.exists():
        logger.warning(f"Ground truth file not found: {gt_upper_path}. Skipping evaluation.")
        return

    ds_gt_hr = xr.open_dataset(gt_upper_path)

    gt_rename_map = {}
    if "valid_time" in ds_gt_hr.dims:
        gt_rename_map["valid_time"] = "time"
    if "pressure_level" in ds_gt_hr.dims:
        gt_rename_map["pressure_level"] = "level"

    var_map = {
        "z": "geopotential",
        "t": "temperature",
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
    }
    for short_name, long_name in var_map.items():
        if short_name in ds_gt_hr.data_vars:
            gt_rename_map[short_name] = long_name

    if gt_rename_map:
        logger.info(f"Standardizing GT data with renames: {gt_rename_map}")
        ds_gt_hr = ds_gt_hr.rename(gt_rename_map)

    # Critical: conservative regridding assumes monotonic coordinates.
    ds_gt_hr = standardize_latlon_coords(ds_gt_hr)

    ds_gt_hr = ds_gt_hr.sel(time=valid_time, method="nearest")
    if "longitude" in ds_gt_hr.dims and "latitude" in ds_gt_hr.dims:
        ds_gt_hr = ds_gt_hr.transpose(..., "longitude", "latitude")

    logger.info("Creating low-resolution ground truth...")
    ds_gt_hr = standardize_latlon_coords(ds_gt_hr)
    source_grid = spherical_harmonic.Grid(
        latitude_nodes=ds_gt_hr.sizes["latitude"],
        longitude_nodes=ds_gt_hr.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(ds_gt_hr.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(ds_gt_hr.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(source_grid, model.data_coords.horizontal)
    ds_gt_native = xarray_utils.regrid(ds_gt_hr, regridder)
    ds_gt_native = xarray_utils.fill_nan_with_nearest(ds_gt_native)

    vars_to_check = {
        "temperature": 850,
        "u_component_of_wind": 850,
        "geopotential": 500,
    }

    logger.info("\n--- PRE-INTERPOLATION WRMSE (Native Resolution) ---")
    for var, level in vars_to_check.items():
        if var in pred_native_res and var in ds_gt_native:
            da_pred = pred_native_res[var].sel(level=level).squeeze()
            da_gt = ds_gt_native[var].sel(level=level).squeeze()
            wrmse = calculate_wrmse(da_pred, da_gt)
            if var == "geopotential":
                wrmse_gpm = wrmse / 9.80665
                logger.info(
                    f"WRMSE for {var} at {level}hPa: {wrmse:.4f} ({grib_para[var_map_inv[var]]['Unit']}), "
                    f"equiv. {wrmse_gpm:.4f} gpm"
                )
            else:
                logger.info(f"WRMSE for {var} at {level}hPa: {wrmse:.4f} ({grib_para[var_map_inv[var]]['Unit']})")

    logger.info("\n--- POST-INTERPOLATION WRMSE (High Resolution 0.25°) ---")
    pred_hr = pred_native_res.interp_like(ds_gt_hr, method="linear")
    for var, level in vars_to_check.items():
        if var in pred_hr and var in ds_gt_hr:
            da_pred = pred_hr[var].sel(level=level).squeeze()
            da_gt = ds_gt_hr[var].sel(level=level).squeeze()
            wrmse = calculate_wrmse(da_pred, da_gt)
            if var == "geopotential":
                wrmse_gpm = wrmse / 9.80665
                logger.info(
                    f"WRMSE for {var} at {level}hPa: {wrmse:.4f} ({grib_para[var_map_inv[var]]['Unit']}), "
                    f"equiv. {wrmse_gpm:.4f} gpm"
                )
            else:
                logger.info(f"WRMSE for {var} at {level}hPa: {wrmse:.4f} ({grib_para[var_map_inv[var]]['Unit']})")

    logger.info("=" * 80 + "\n")


# ==============================================================================
# Helper & Main Functions
# ==============================================================================
def get_target_date():
    if DATE_FILE.exists():
        with open(DATE_FILE) as f:
            return f.read().strip()
    return "2023010112"


def setup_device():
    logging.getLogger("jax").setLevel(logging.ERROR)
    logger.info(f"JAX Devices: {jax.devices()}")


def load_model(weights_path):
    if not weights_path.exists():
        logger.error(f"FATAL: Model weights missing at {weights_path}")
        sys.exit(1)
    logger.info(f"Loading model: {weights_path.name}")
    with open(weights_path, "rb") as f:
        ckpt = pickle.load(f)
    return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)


def load_input_data(data_path):
    if not data_path.exists():
        logger.error(f"FATAL: Input data missing at {data_path}")
        sys.exit(1)
    logger.info(f"Loading input: {data_path.name}")
    return xr.open_dataset(data_path)

def load_shifted_forcings_if_available(
    target_date: str,
    model: neuralgcm.PressureLevelModel,
    init_time: pd.Timestamp,
) -> dict:
    """
    If available, load pre-shifted Gaussian-grid forcings produced by
    `prepare_forcings_shifted.py`:
      assets/data/processed_neuralgcm/forcings_shifted_{date}_gaussian.nc
    and convert to `model.unroll()` forcings dict via `forcings_from_xarray()`.
    """
    forcing_path = PROCESSED_DIR / f"forcings_shifted_{target_date}_gaussian.nc"
    if not forcing_path.exists():
        return {}

    logger.info(f"Loading shifted forcings: {forcing_path.name}")
    ds_f = xr.open_dataset(forcing_path)
    if "valid_time" in ds_f.dims:
        ds_f = ds_f.rename({"valid_time": "time"})
    if "pressure_level" in ds_f.dims:
        ds_f = ds_f.rename({"pressure_level": "level"})

    # Ensure we have a time axis of length 1 at init time (or nearest).
    ds_f = ds_f.sel(time=[np.datetime64(init_time)], method="nearest")
    return model.forcings_from_xarray(ds_f)


def maybe_regrid_input_to_model_grid(ds_input: xr.Dataset, model: neuralgcm.PressureLevelModel) -> xr.Dataset:
    ds_input = standardize_latlon_coords(ds_input)
    src_lat = ds_input.sizes.get("latitude")
    src_lon = ds_input.sizes.get("longitude")
    tgt_lat = model.data_coords.horizontal.latitude_nodes
    tgt_lon = model.data_coords.horizontal.longitude_nodes

    if (src_lat, src_lon) == (tgt_lat, tgt_lon):
        logger.info(f"Input grid already matches model grid: {src_lon}x{src_lat}")
        return ds_input

    logger.warning(
        f"Input grid ({src_lon}x{src_lat}) does not match model grid ({tgt_lon}x{tgt_lat}); "
        "regridding input to model native grid before encode."
    )

    source_grid = spherical_harmonic.Grid(
        latitude_nodes=ds_input.sizes["latitude"],
        longitude_nodes=ds_input.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(ds_input.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(ds_input.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        source_grid, model.data_coords.horizontal, skipna=True
    )
    ds_regridded = xarray_utils.regrid(ds_input, regridder)
    ds_regridded = xarray_utils.fill_nan_with_nearest(ds_regridded)
    return ds_regridded


def save_output(ds_native: xr.Dataset, target_date: str, lead_time_hours: int):
    logger.info("Preparing data for final save (Standardized Grid)...")

    ds_native = ds_native.drop_vars("sim_time", errors="ignore")
    ds_native = standardize_latlon_coords(ds_native)

    if "longitude" in ds_native.dims:
        ds_0 = ds_native.isel(longitude=0)
        ds_360 = ds_0.assign_coords(longitude=360.0)
        ds_cyclic = xr.concat([ds_native, ds_360], dim="longitude")
    else:
        ds_cyclic = ds_native

    # xarray interpolation expects monotonic coordinates. Interpolate on
    # increasing latitude, then flip back to benchmark's common 90->-90 ordering.
    lat_target = np.linspace(-90.0, 90.0, 721)
    lon_target = np.linspace(0, 360, 1440, endpoint=False)

    ds_latlon = ds_cyclic.interp(
        latitude=lat_target,
        longitude=lon_target,
        method="linear",
    )

    # Restore descending latitude if desired by downstream tools.
    ds_latlon = ds_latlon.sortby("latitude", ascending=False)

    if bool(ds_latlon.to_array().isnull().any()):
        logger.info("Filling remaining NaNs with nearest-neighbor values...")
        ds_latlon = xarray_utils.fill_nan_with_nearest(ds_latlon)

    saver = Saver(save_root=str(OUTPUT_DIR))

    name_map = {
        "geopotential": "z",
        "temperature": "t",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
        "specific_humidity": "q",
        "specific_cloud_ice_water_content": "ciwc",
        "specific_cloud_liquid_water_content": "clwc",
    }
    channel_names = []
    data_slices = []

    for var_name, data_array in ds_latlon.data_vars.items():
        short_var_name = name_map.get(var_name, var_name)

        if "latitude" in data_array.dims and "longitude" in data_array.dims:
            data_array = data_array.transpose(..., "latitude", "longitude")

        if "level" in data_array.dims:
            for level in data_array.level.values:
                channel_name = f"{short_var_name}_{int(level)}"
                data_slices.append(data_array.sel(level=level).squeeze().values)
                channel_names.append(channel_name)
        else:
            channel_name = short_var_name
            data_slices.append(data_array.squeeze().values)
            channel_names.append(channel_name)

    if not data_slices:
        raise ValueError("No data found.")

    final_data_array = np.stack(data_slices, axis=0)

    saver.save(
        data=final_data_array,
        channel_mapping=channel_names,
        init_time_str=target_date,
        lead_time_hours=lead_time_hours,
        # Use post-processed coordinate order (after sorting) to avoid latitude flips
        # between the dataset and the saved NetCDF.
        lat_values=ds_latlon.latitude.values,
        lon_values=ds_latlon.longitude.values,
    )
    logger.info("✅ Output successfully saved (Standardized Grid).")


def main():
    parser = argparse.ArgumentParser(description="Run NeuralGCM Inference and Evaluation")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYYMMDDHH)")
    parser.add_argument("--steps", type=int, default=6, help="Forecast hours")
    parser.add_argument(
        "--model-res",
        type=str,
        default="0.7",
        choices=["0.7", "1.4", "2.8"],
        help="NeuralGCM checkpoint resolution; must match prepared input grid for best skill.",
    )
    args = parser.parse_args()

    setup_device()
    target_date = args.date or get_target_date()
    logger.info(f"Starting inference for Target Date: {target_date}, Forecast: {args.steps} hours")

    model_name_map = {
        "0.7": "models_v1_deterministic_0_7_deg.pkl",
        "1.4": "models_v1_deterministic_1_4_deg.pkl",
        "2.8": "models_v1_deterministic_2_8_deg.pkl",
    }
    model_path = WEIGHTS_DIR / model_name_map[args.model_res]
    data_path = PROCESSED_DIR / f"input_{target_date}_gaussian.nc"
    model = load_model(model_path)
    ds_input = load_input_data(data_path)
    ds_input = maybe_regrid_input_to_model_grid(ds_input, model)

    if "valid_time" in ds_input.dims:
        ds_input = ds_input.rename({"valid_time": "time"})
    if "pressure_level" in ds_input.dims:
        ds_input = ds_input.rename({"pressure_level": "level"})

    var_map = {
        "z": "geopotential",
        "t": "temperature",
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
        "q": "specific_humidity",
    }
    rename_dict = {k: v for k, v in var_map.items() if k in ds_input.data_vars}
    if rename_dict:
        ds_input = ds_input.rename(rename_dict)

    if "level" in ds_input.coords and ds_input.level.size < 37:
        logger.warning(
            f"Input pressure levels = {ds_input.level.size}. NeuralGCM is typically trained with 37 levels; "
            "reduced levels can significantly degrade Z500 skill."
        )

    logger.info("Encoding initial state...")
    init_time = pd.to_datetime(target_date, format="%Y%m%d%H")
    # Explicitly select the initialization time (avoid accidental isel mismatch).
    ds_init = ds_input.sel(time=np.datetime64(init_time), method="nearest")
    rng_key = jax.random.key(42)
    inputs = model.inputs_from_xarray(ds_init)
    input_forcings = model.forcings_from_xarray(ds_init)
    encoded_state = model.encode(inputs, input_forcings, rng_key)

    logger.info(f"Running inference for {args.steps} hours...")
    dt_hours = model.timestep / np.timedelta64(1, "h")
    n_steps = int(args.steps / dt_hours)
    if n_steps <= 0:
        logger.error(f"Invalid forecast length: steps={args.steps}, model dt={dt_hours}h")
        sys.exit(1)

    # Prefer pre-shifted forcings (official 24h temporal shift), if present.
    all_forcings = load_shifted_forcings_if_available(target_date, model, init_time)
    if not all_forcings:
        # Fallback: persistence forcings from the init-time slice (time axis length 1).
        # `forcings_from_xarray()` injects `sim_time` automatically from the `time` coordinate.
        ds_forcings = ds_input.sel(time=[np.datetime64(init_time)], method="nearest")
        all_forcings = model.forcings_from_xarray(ds_forcings)

    # Output timestamps for +1h .. +steps*h (aligns to +6h target at last frame).
    out_times = np.array(
        [init_time + pd.Timedelta(hours=(i + 1) * float(dt_hours)) for i in range(n_steps)],
        dtype="datetime64[ns]",
    )

    logger.info(
        f"Forcing shapes - sst: {np.shape(all_forcings.get('sea_surface_temperature'))}, "
        f"sic: {np.shape(all_forcings.get('sea_ice_cover'))}, "
        f"sim_time: {np.shape(all_forcings.get('sim_time'))}"
    )

    _, predictions = model.unroll(
        encoded_state,
        all_forcings,
        steps=n_steps,
        timedelta=model.timestep,
        # We want outputs at +timestep .. +steps*timestep (last frame == +6h).
        start_with_input=False,
    )

    decoded_ds_native = model.data_to_xarray(predictions, times=out_times).isel(time=-1, drop=False)

    evaluate_errors(decoded_ds_native, model, target_date, args.steps)
    save_output(decoded_ds_native, target_date, args.steps)

    logger.info("✅ Inference and Evaluation complete.")


if __name__ == "__main__":
    main()
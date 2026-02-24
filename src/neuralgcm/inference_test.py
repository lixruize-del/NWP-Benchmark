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
RAW_DATA_DIR = BASE_DIR / "assets" / "data" / "era5_neuralgcm"
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "neuralgcm"
OUTPUT_DIR = BASE_DIR / "outputs" / "neuralgcm"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.append(str(BASE_DIR))

try:
    from src.common.saver import Saver
except ImportError:
    logger.error("FATAL: Could not import the Saver class from src.common.saver.")
    Saver = None

# ==============================================================================
# Metric Calculation Utilities (Adapted from your Evaluator)
# ==============================================================================
def compute_latitude_weights(latitudes: xr.DataArray) -> xr.DataArray:
    """Computes cosine latitude weights for global grids."""
    weights = np.cos(np.deg2rad(latitudes))
    weights = weights / weights.mean()  # Normalize weights
    weights.name = "latitude_weights"
    return weights

def calculate_wrmse(pred: xr.DataArray, gt: xr.DataArray) -> float:
    """Computes Weighted Root Mean Square Error (WRMSE)."""
    # Ensure latitude coordinates are named the same for weight calculation
    lat_coord_name = 'latitude'
    if 'lat' in gt.coords:
        lat_coord_name = 'lat'
        
    weights = compute_latitude_weights(gt[lat_coord_name])
    diff_sq = (pred - gt) ** 2
    
    # Use xarray's weighted mean capability
    wrmse = np.sqrt(diff_sq.weighted(weights).mean()).item()
    return wrmse

# ==============================================================================
# Core Evaluation Function
# ==============================================================================
def evaluate_errors(
    pred_native_res: xr.Dataset,
    model: neuralgcm.PressureLevelModel,
    target_date: str,
    lead_time_hours: int,
):
    """
    Calculates and prints both pre- and post-interpolation WRMSE in NATIVE MODEL UNITS.
    This function serves as the SINGLE SOURCE OF TRUTH for evaluation.
    - Geopotential ('z') is evaluated in m^2/s^2.
    - Temperature ('t') is evaluated in K.
    - Wind components ('u', 'v') are evaluated in m/s.
    """
    logger.info("="*80)
    logger.info("Starting Final, Unified Pre- and Post-Interpolation WRMSE Evaluation")
    logger.info("="*80)

    # 1. Load High-Resolution Ground Truth (GT)
    valid_time = pd.to_datetime(target_date, format="%Y%m%d%H") + pd.Timedelta(hours=lead_time_hours)
    valid_time_str = valid_time.strftime("%Y%m%d%H")
    
    gt_upper_path = RAW_DATA_DIR / f"upper_{valid_time_str}.nc"
    if not gt_upper_path.exists():
        logger.warning(f"Ground truth file not found: {gt_upper_path}. Skipping evaluation.")
        return
    ds_gt_hr = xr.open_dataset(gt_upper_path)

    # 2. DEFENSIVE DATA STANDARDIZATION of GT
    # This block ensures GT data format is consistent, regardless of its source (CDS API, etc.)
    gt_rename_map = {}
    # Dimensions
    if 'valid_time' in ds_gt_hr.dims: gt_rename_map['valid_time'] = 'time'
    if 'pressure_level' in ds_gt_hr.dims: gt_rename_map['pressure_level'] = 'level'
    # Variables (short names from CDS to long names used by NeuralGCM)
    var_map = {'z': 'geopotential', 't': 'temperature', 'u': 'u_component_of_wind', 'v': 'v_component_of_wind'}
    for short_name, long_name in var_map.items():
        if short_name in ds_gt_hr.data_vars:
            gt_rename_map[short_name] = long_name
    
    if gt_rename_map:
        logger.info(f"Standardizing GT data with renames: {gt_rename_map}")
        ds_gt_hr = ds_gt_hr.rename(gt_rename_map)
    
    ds_gt_hr = ds_gt_hr.sel(time=valid_time, method='nearest')

    # 3. Create Low-Resolution GT for pre-interpolation comparison
    logger.info("Creating low-resolution ground truth...")
    ds_gt_hr = ds_gt_hr.sortby("latitude", ascending=True) # Regridder requires sorted coords
    source_grid = spherical_harmonic.Grid(
        latitude_nodes=ds_gt_hr.sizes['latitude'],
        longitude_nodes=ds_gt_hr.sizes['longitude'],
        latitude_spacing=xarray_utils.infer_latitude_spacing(ds_gt_hr.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(ds_gt_hr.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(source_grid, model.data_coords.horizontal)
    ds_gt_native = xarray_utils.regrid(ds_gt_hr, regridder)
    ds_gt_native = xarray_utils.fill_nan_with_nearest(ds_gt_native)

    # 4. Define variables and levels to check
    vars_to_check = {
        "temperature": 850,
        "u_component_of_wind": 850,
        "geopotential": 500
    }
    
    # 5. Perform Evaluations
    logger.info("\n--- PRE-INTERPOLATION WRMSE (Native Resolution ~1.4°) ---")
    for var, level in vars_to_check.items():
        if var in pred_native_res and var in ds_gt_native:
            da_pred = pred_native_res[var].sel(level=level).squeeze()
            da_gt = ds_gt_native[var].sel(level=level).squeeze()
            wrmse = calculate_wrmse(da_pred, da_gt)
            logger.info(f"WRMSE for {var} at {level}hPa: {wrmse:.4f} ({grib_para[var_map_inv[var]]['Unit']})")

    logger.info("\n--- POST-INTERPOLATION WRMSE (High Resolution 0.25°) ---")
    pred_hr = pred_native_res.interp_like(ds_gt_hr, method="linear")
    for var, level in vars_to_check.items():
        if var in pred_hr and var in ds_gt_hr:
            da_pred = pred_hr[var].sel(level=level).squeeze()
            da_gt = ds_gt_hr[var].sel(level=level).squeeze()
            wrmse = calculate_wrmse(da_pred, da_gt)
            logger.info(f"WRMSE for {var} at {level}hPa: {wrmse:.4f} ({grib_para[var_map_inv[var]]['Unit']})")
    
    logger.info("="*80 + "\n")

# You will also need to add `calculate_wrmse` and `var_map_inv` to your script
def calculate_wrmse(pred: xr.DataArray, gt: xr.DataArray) -> float:
    """Computes Weighted Root Mean Square Error (WRMSE)."""
    weights = np.cos(np.deg2rad(gt.latitude))
    weights.name = "weights"
    diff_sq = (pred - gt)**2
    wrmse = np.sqrt(diff_sq.weighted(weights).mean()).item()
    return wrmse

var_map_inv = {'geopotential': 'z', 'temperature': 't', 'u_component_of_wind': 'u'}

# And ensure grib_para is available or defined in the script
grib_para = { "z": {"Unit": "m^2 s^-2"}, "t": {"Unit": "K"}, "u": {"Unit": "m s^-1"} }

# ==============================================================================
# Helper & Main Functions (largely unchanged, but with calls to new evaluator)
# ==============================================================================
# ... [Your existing get_target_date, setup_device, load_model, etc. functions here] ...
def get_target_date():
    """Reads target date from a file, with a fallback default."""
    if DATE_FILE.exists():
        with open(DATE_FILE) as f:
            return f.read().strip()
    return "2023010112"

def setup_device():
    """Initializes JAX, reporting device used and handling GPU errors gracefully."""
    try:
        logging.getLogger("jax").setLevel(logging.ERROR) # Suppress verbose JAX logs
        devices = jax.devices()
        logger.info(f"JAX Devices: {devices}")
    except Exception as e:
        logger.warning(f"JAX GPU detection failed, will fall back to CPU. Error: {e}")

def load_model(weights_path):
    """Loads the NeuralGCM model from the specified checkpoint file."""
    if not weights_path.exists():
        logger.error(f"FATAL: Model weights missing at {weights_path}")
        sys.exit(1)
    logger.info(f"Loading model: {weights_path.name}")
    with open(weights_path, 'rb') as f:
        ckpt = pickle.load(f)
    return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

def load_input_data(data_path):
    """Loads the input dataset from the specified NetCDF file."""
    if not data_path.exists():
        logger.error(f"FATAL: Input data missing at {data_path}")
        sys.exit(1)
    logger.info(f"Loading input: {data_path.name}")
    return xr.open_dataset(data_path)

def save_output(ds: xr.Dataset, target_date: str, lead_time_hours: int):
    """Coordinates the full data processing and saving pipeline."""
    logger.info("Preparing data for final save...")

    # For saving, we need to interpolate to the standard 0.25 degree grid
    lat_target = np.linspace(90, -90, 721)
    lon_target = np.linspace(0, 360, 1440, endpoint=False)
    
    # Create a dummy DataArray with the target grid to use interp_like
    target_grid = xr.DataArray(
        coords={'latitude': lat_target, 'longitude': lon_target}, 
        dims=['latitude', 'longitude']
    )


    # 2. 循环填充 (Cyclic Padding) - 解决经度 0/360 缝隙的核心
    # NeuralGCM 输出是 [0, ..., <360]。线性插值在末端需要 360 度的数据。
    # 我们把 0 度的数据复制一份，坐标设为 360，拼接到末尾。
    if 'longitude' in ds.dims:
        ds_0 = ds.isel(longitude=0)
        ds_360 = ds_0.assign_coords(longitude=360.0)
        ds_cyclic = xr.concat([ds, ds_360], dim='longitude')
    else:
        ds_cyclic = ds

    # 3. 线性插值
    # 现在有了 360 度的数据，interp 就能完美算出 359.75 右边的值，不会产生 NaN
    ds_latlon = ds_cyclic.interp(
        latitude=lat_target, 
        longitude=lon_target, 
        method="linear"
    )

    # 4. [已删除] 致命的 fillna(0.0)
    # ds_latlon = ds_latlon.where(np.isfinite(ds_latlon), 0.0)  <-- DELETE THIS

    # 5. 极点微调 (可选)
    # 极少数情况下极点可能还有 NaN，用最近邻填充，绝对不能填 0
    if ds_latlon.isnull().any():
        logger.info("Filling remaining NaNs (poles) with nearest values...")
        ds_latlon = ds_latlon.bfill("latitude").ffill("latitude")

    try:
        if not Saver:
            raise ImportError("Saver class is not available to use.")
        saver = Saver(save_root=str(OUTPUT_DIR))

        name_map = {
            'geopotential': 'z', 'temperature': 't', 'u_component_of_wind': 'u',
            'v_component_of_wind': 'v', 'specific_humidity': 'q',
            'specific_cloud_ice_water_content': 'ciwc',
            'specific_cloud_liquid_water_content': 'clwc',
        }
        channel_names = []
        data_slices = []

        for var_name, data_array in ds_latlon.data_vars.items():
            if 'time' in data_array.dims and data_array.dims != ('time',):
                data_array = data_array.squeeze('time', drop=True)

            if var_name == 'sim_time': continue
            
            short_var_name = name_map.get(var_name, var_name)

            # 确保维度顺序 (Lat, Lon)
            # 你的 evaluate.py 期望 (latitude, longitude)
            if 'latitude' in data_array.dims and 'longitude' in data_array.dims:
                data_array = data_array.transpose(..., 'latitude', 'longitude')

            if 'level' in data_array.dims:
                for level in data_array.level.values:
                    channel_name = f"{short_var_name}_{int(level)}"
                    data_slices.append(data_array.sel(level=level).squeeze().values)
                    channel_names.append(channel_name)
            else:
                channel_name = short_var_name
                data_slices.append(data_array.squeeze().values)
                channel_names.append(channel_name)

        if not data_slices:
            raise ValueError("No valid data slices found for saving.")

        final_data_array = np.stack(data_slices, axis=0)
        
        saver.save(
            data=final_data_array,
            channel_mapping=channel_names,
            init_time_str=target_date,
            lead_time_hours=lead_time_hours,
            lat_values=lat_target,
            lon_values=lon_target
        )
        logger.info(f"✅ Output successfully saved.")

    except Exception as e:
        logger.error(f"Saver failed: {e}", exc_info=True)
        fallback_path = OUTPUT_DIR / f"pred_{target_date}_{lead_time_hours}h_FALLBACK.nc"
        ds_latlon.to_netcdf(fallback_path)











































"""



    ds_latlon = ds.interp_like(target_grid, method="linear")

    # Clean NaNs that might appear after interpolation
    ds_latlon = ds_latlon.where(np.isfinite(ds_latlon), 0.0)

    try:
        if not Saver:
            raise ImportError("Saver class is not available to use.")
        saver = Saver(save_root=str(OUTPUT_DIR))

        name_map = {
            'geopotential': 'z', 'temperature': 't', 'u_component_of_wind': 'u',
            'v_component_of_wind': 'v', 'specific_humidity': 'q',
            'specific_cloud_ice_water_content': 'ciwc',
            'specific_cloud_liquid_water_content': 'clwc',
        }
        channel_names = []
        data_slices = []

        for var_name, data_array in ds_latlon.data_vars.items():
            if 'time' in data_array.dims and data_array.dims != ('time',): # Skip scalar time coord
                data_array = data_array.squeeze('time', drop=True)

            if var_name == 'sim_time': continue

            if 'latitude' in data_array.dims and 'longitude' in data_array.dims:
                # Identify non-spatial dimensions (like 'level')
                other_dims = [d for d in data_array.dims if d not in ['latitude', 'longitude']]
                # Define the correct final order
                correct_order = other_dims + ['latitude', 'longitude']
                data_array = data_array.transpose(*correct_order)
            
            short_var_name = name_map.get(var_name, var_name)

            if 'level' in data_array.dims:
                for level in data_array.level.values:
                    channel_name = f"{short_var_name}_{int(level)}"
                    slice_data = data_array.sel(level=level).values
                    channel_names.append(channel_name)
                    data_slices.append(slice_data)
            else:
                channel_name = short_var_name
                slice_data = data_array.values
                channel_names.append(channel_name)
                data_slices.append(slice_data)

        if not data_slices:
            raise ValueError("No valid data slices found for saving.")

        final_data_array = np.stack(data_slices, axis=0)
        
        saver.save(
            data=final_data_array,
            channel_mapping=channel_names,
            init_time_str=target_date,
            lead_time_hours=lead_time_hours,
            lat_values=ds_latlon.latitude.values,
            lon_values=ds_latlon.longitude.values
        )

    except Exception as e:
        logger.error(f"The primary Saver utility failed: {e}", exc_info=True)
        fallback_path = OUTPUT_DIR / f"pred_{target_date}_{lead_time_hours}h_FALLBACK.nc"
        logger.info(f"Attempting fallback save to: {fallback_path}")
        try:
            ds_latlon.to_netcdf(fallback_path)
            logger.warning(f"Fallback save successful: {fallback_path}")
        except Exception as fallback_e:
            logger.error(f"FATAL: Fallback save also failed: {fallback_e}")






"""











def main():
    parser = argparse.ArgumentParser(description="Run NeuralGCM Inference and Evaluation")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYYMMDDHH)")
    parser.add_argument("--steps", type=int, default=6, help="Forecast hours")
    args = parser.parse_args()

    setup_device()
    target_date = args.date or get_target_date()
    logger.info(f"Starting inference for Target Date: {target_date}, Forecast: {args.steps} hours")

    # Load model and data
    model_path = WEIGHTS_DIR / "models_v1_deterministic_1_4_deg.pkl"
    data_path = PROCESSED_DIR / f"input_{target_date}_gaussian.nc"
    model = load_model(model_path)
    ds_input = load_input_data(data_path)
    
    # --- Data format fix for CDS data ---
    if 'valid_time' in ds_input.dims:
        ds_input = ds_input.rename({'valid_time': 'time'})
    var_map = {
        'z': 'geopotential', 't': 'temperature', 'u': 'u_component_of_wind',
        'v': 'v_component_of_wind', 'q': 'specific_humidity'
    }
    rename_dict = {k: v for k, v in var_map.items() if k in ds_input.data_vars}
    if rename_dict: ds_input = ds_input.rename(rename_dict)
    if 'pressure_level' in ds_input.dims:
        ds_input = ds_input.rename({'pressure_level': 'level'})
    # ---

    # Encode
    logger.info("Encoding initial state...")
    ds_slice = ds_input.isel(time=-1)
    rng_key = jax.random.key(42)
    inputs = model.inputs_from_xarray(ds_slice)
    input_forcings = model.forcings_from_xarray(ds_slice)
    encoded_state = model.encode(inputs, input_forcings, rng_key)

    # Run inference loop
    logger.info(f"Running inference for {args.steps} hours...")
    dt_hours = model.timestep / np.timedelta64(1, 'h')
    n_steps = int(args.steps / dt_hours)
    
    current_state = encoded_state
    step_fn = jax.jit(model.advance)
    for _ in range(n_steps):
        current_state = step_fn(current_state, input_forcings)

    # Decode
    logger.info("Decoding final state...")
    valid_time = pd.to_datetime(target_date, format="%Y%m%d%H") + pd.Timedelta(hours=args.steps)
    times = np.array([np.datetime64(valid_time)])
    decoded_dict = model.decode(current_state, input_forcings)
    decoded_ds_native = model.data_to_xarray(
        {k: v[np.newaxis, ...] for k, v in decoded_dict.items()},
        times=times
    )

    # --- Call Error Evaluation ---
    evaluate_errors(decoded_ds_native, model, target_date, args.steps)

    # --- Save the final output ---
    save_output(decoded_ds_native, target_date, args.steps)
    
    logger.info("✅ Inference and Evaluation complete.")

if __name__ == "__main__":
    main()
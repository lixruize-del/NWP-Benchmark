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

# ==============================================================================
# Configuration
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("NeuralGCM.Inference")

# Define paths relative to the script location
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
PROCESSED_DIR = BASE_DIR / "assets" / "data" / "processed_neuralgcm"
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "neuralgcm"
OUTPUT_DIR = BASE_DIR / "outputs" / "neuralgcm"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

# Ensure output directory and system path are set up 
os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.append(str(BASE_DIR))

# Dynamically import the Saver, handling potential failure
try:
    from src.common.saver import Saver
except ImportError:
    logger.error("FATAL: Could not import the Saver class from src.common.saver.")
    Saver = None

# ==============================================================================
# Helper Functions
# ==============================================================================
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

from dinosaur import spherical_harmonic
from dinosaur import horizontal_interpolation
from dinosaur import xarray_utils

def save_output(ds: xr.Dataset, target_date: str, lead_time_hours: int):
    """
    Coordinates the full data processing and saving pipeline using Dinosaur's native regridder.
    This guarantees correct handling of periodic boundaries (0/360) and poles.
    """
    logger.info("Regridding output to 0.25 deg Lat/Lon using Dinosaur ConservativeRegridder...")

    # 1. 定义目标网格 (ERA5 标准: 0.25度, 包含极点)
    # dinosaur 使用弧度制，所以这里不用手动 linspace
    target_grid = spherical_harmonic.Grid(
        latitude_nodes=721,   # 180/0.25 + 1
        longitude_nodes=1440, # 360/0.25
        latitude_spacing='equiangular_with_poles', 
        longitude_offset=0.0
    )

    # 2. 推断源网格 (NeuralGCM 的高斯网格)
    # 我们利用 xarray_utils 自动从 dataset 推断网格信息
    try:
        source_grid = xarray_utils.coordinate_system_from_dataset(ds).horizontal
    except Exception:
        # 如果推断失败 (例如 ds 丢失了某些属性)，手动构建
        logger.info("Auto-inference of source grid failed, constructing manually...")
        source_grid = spherical_harmonic.Grid(
            latitude_nodes=ds.sizes['latitude'],
            longitude_nodes=ds.sizes['longitude'],
            latitude_spacing='gauss', # NeuralGCM 原生是 Gauss
            longitude_offset=xarray_utils.infer_longitude_offset(ds.longitude)
        )

    # 3. 构建 Regridder
    # ConservativeRegridder 会自动处理经度周期性 (Periodicity)，不会产生接缝 NaN
    regridder = horizontal_interpolation.ConservativeRegridder(
        source_grid=source_grid,
        target_grid=target_grid
    )

    # 4. 执行重采样 (Regridding)
    # dinosaur 会返回一个新的 Dataset，坐标已经是转换好的
    ds_for_regridding = ds.drop_vars('sim_time', errors='ignore')
    # ================================================
    ds_latlon = xarray_utils.regrid(ds_for_regridding, regridder)
    
    # 5. 极少量的极点 NaN 修复
    ds_latlon = xarray_utils.fill_nan_with_nearest(ds_latlon)

    # 6. 保存逻辑
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

        logger.info("Slicing variables for saving...")
        for var_name, data_array in ds_latlon.data_vars.items():
            if var_name == 'sim_time': continue
            
            short_var_name = name_map.get(var_name, var_name)

            # 强制调整维度顺序为 (..., latitude, longitude) 以适配 Saver
            if 'latitude' in data_array.dims and 'longitude' in data_array.dims:
                data_array = data_array.transpose(..., 'latitude', 'longitude')

            # 提取数据 .values
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
        
        # 注意：dinosaur grid 存储的是弧度，Saver 需要角度
        saver.save(
            data=final_data_array,
            channel_mapping=channel_names,
            init_time_str=target_date,
            lead_time_hours=lead_time_hours,
            lat_values=np.rad2deg(target_grid.latitudes), 
            lon_values=np.rad2deg(target_grid.longitudes)
        )
        logger.info("✅ Output successfully saved (using Dinosaur Regridder).")

    except Exception as e:
        logger.error(f"The primary Saver utility failed: {e}", exc_info=True)
        # Fallback
        fallback_path = OUTPUT_DIR / f"pred_{target_date}_{lead_time_hours}h_FALLBACK.nc"
        ds_latlon.to_netcdf(fallback_path)
        logger.warning(f"Fallback save successful: {fallback_path}")
            
# ==============================================================================
# Main Execution Block
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run NeuralGCM Inference")
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
    step_fn = jax.jit(model.advance) # JIT compile the step function for speed

    for _ in range(n_steps):
        current_state = step_fn(current_state, input_forcings) 

    # Decode
    logger.info("Decoding final state...")
    valid_time = pd.to_datetime(target_date, format="%Y%m%d%H") + pd.Timedelta(hours=args.steps)
    times = np.array([np.datetime64(valid_time)])
    decoded_dict = model.decode(current_state, input_forcings)
    decoded_ds = model.data_to_xarray(
        {k: v[np.newaxis, ...] for k, v in decoded_dict.items()},
        times=times
    )

    # Save the final output
    save_output(decoded_ds, target_date, args.steps)
    logger.info("✅ Inference task complete.")

if __name__ == "__main__":
    main()
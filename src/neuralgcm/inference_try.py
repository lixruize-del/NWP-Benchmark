import os
import sys
import logging
import argparse
import pickle
import numpy as np
import xarray as xr
import jax
import pandas as pd
from pathlib import Path

import neuralgcm
from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils

# ==============================================================================
# Configuration
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("NeuralGCM.Inference")

# Paths
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
PROCESSED_DIR = BASE_DIR / "assets" / "data" / "processed_neuralgcm"
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "neuralgcm"
OUTPUT_DIR = BASE_DIR / "outputs" / "neuralgcm"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.append(str(BASE_DIR))

try:
    from src.common.saver import Saver
except ImportError:
    logger.error("FATAL: Saver class missing.")
    Saver = None

# ==============================================================================
# Helper Functions
# ==============================================================================
def get_target_date():
    if DATE_FILE.exists():
        with open(DATE_FILE) as f:
            return f.read().strip()
    return "2023010112"

def setup_device():
    try:
        logger.info(f"JAX Devices: {jax.devices()}")
    except Exception as e:
        logger.warning(f"JAX GPU detection failed: {e}")

def load_model(weights_path):
    if not weights_path.exists():
        logger.error(f"FATAL: Model weights missing at {weights_path}")
        sys.exit(1)
    logger.info(f"Loading model: {weights_path.name}")
    with open(weights_path, 'rb') as f:
        ckpt = pickle.load(f)
    return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

def load_input_data(data_path):
    if not data_path.exists():
        logger.error(f"FATAL: Input data missing at {data_path}")
        sys.exit(1)
    logger.info(f"Loading input: {data_path.name}")
    return xr.open_dataset(data_path)

# ==============================================================================
# Core Function: Output Regridding (Official WB2 Style)
# ==============================================================================
def save_output_official_wb2(ds_native: xr.Dataset, model: neuralgcm.PressureLevelModel, target_date: str, lead_time_hours: int):
    """
    使用 dinosaur ConservativeRegridder 重采样到 ERA5 0.25度网格。
    这是最稳健的方法，只要输入物理量正确，RMSE 应该是正常的。
    """
    logger.info("Regridding to ERA5 0.25° grid using Conservative Regridding...")

    # 1. 定义目标网格 (ERA5 0.25度)
    target_grid = spherical_harmonic.Grid(
        latitude_nodes=721,
        longitude_nodes=1440,
        latitude_spacing='equiangular_with_poles', 
        longitude_offset=0.0,
    )

    # 2. 源网格
    source_grid = model.data_coords.horizontal

    # 3. Regridder
    regridder = horizontal_interpolation.ConservativeRegridder(
        source_grid=source_grid,
        target_grid=target_grid,
        skipna=True
    )

    # 4. 执行重采样
    ds_for_regridding = ds_native.drop_vars('sim_time', errors='ignore')
    ds_latlon = xarray_utils.regrid(ds_for_regridding, regridder)
    ds_latlon = xarray_utils.fill_nan_with_nearest(ds_latlon)

    # 5. 保存
    try:
        if not Saver: raise ImportError("Saver missing")
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
            if var_name == 'sim_time': continue
            short_var_name = name_map.get(var_name, var_name)

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

        if not data_slices: raise ValueError("No data found.")
        final_data_array = np.stack(data_slices, axis=0)
        
        saver.save(
            data=final_data_array,
            channel_mapping=channel_names,
            init_time_str=target_date,
            lead_time_hours=lead_time_hours,
            lat_values=np.rad2deg(target_grid.latitudes),
            lon_values=np.rad2deg(target_grid.longitudes)
        )
        logger.info("✅ Output saved.")

    except Exception as e:
        logger.error(f"Saver failed: {e}", exc_info=True)
        ds_latlon.to_netcdf(OUTPUT_DIR / f"pred_{target_date}_{lead_time_hours}h_FALLBACK.nc")

# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    setup_device()
    target_date = args.date or get_target_date()
    
    # 1. 加载
    model_path = WEIGHTS_DIR / "models_v1_deterministic_1_4_deg.pkl"
    data_path = PROCESSED_DIR / f"input_{target_date}_gaussian.nc"
    
    model = load_model(model_path)
    ds_input = load_input_data(data_path)

    # 重命名
    if 'valid_time' in ds_input.dims: ds_input = ds_input.rename({'valid_time': 'time'})
    if 'pressure_level' in ds_input.dims: ds_input = ds_input.rename({'pressure_level': 'level'})
    var_map = {'z': 'geopotential', 't': 'temperature', 'u': 'u_component_of_wind', 'v': 'v_component_of_wind', 'q': 'specific_humidity'}
    ds_input = ds_input.rename({k: v for k, v in var_map.items() if k in ds_input.data_vars})

    # 2. 编码
    logger.info("Encoding initial state...")
    ds_slice = ds_input.isel(time=-1)
    inputs = model.inputs_from_xarray(ds_slice)
    input_forcings = model.forcings_from_xarray(ds_slice)
    rng_key = jax.random.key(42)
    encoded_state = model.encode(inputs, input_forcings, rng_key)

    # 3. 准备 Forcings (FIXED PHYSICS)
    logger.info(f"Running inference for {args.steps} hours...")
    dt_hours = model.timestep / np.timedelta64(1, 'h')
    n_steps = int(args.steps / dt_hours)

    # --- 关键修复：正确构造 Forcing ---
    # A. 复制 SST/海冰 (Persistence)
    all_forcings = jax.tree_util.tree_map(
        lambda x: np.repeat(x[np.newaxis, ...], n_steps, axis=0), 
        input_forcings
    )
    
    # B. 修正 sim_time (Dynamic Update)
    # 我们必须为未来的每一步生成正确的 sim_time
    start_time = pd.to_datetime(target_date, format="%Y%m%d%H")
    future_times = [start_time + pd.Timedelta(hours=(i+1)*int(dt_hours)) for i in range(n_steps)]
    future_times_np = np.array(future_times).astype("datetime64[ns]")
    
    # 使用模型自带函数转为无量纲时间
    future_sim_times = model.datetime64_to_sim_time(future_times_np)
    
    # 覆盖错误的重复时间
    all_forcings['sim_time'] = future_sim_times

    # 4. 预测 (Unroll)
    _, predictions = model.unroll(
        encoded_state,
        all_forcings,
        steps=n_steps,
        timedelta=model.timestep,
        start_with_input=False 
    )

    # 5. 转换
    ds_predictions = model.data_to_xarray(predictions, times=future_times_np)
    ds_output_native = ds_predictions.isel(time=-1)

    # 6. 保存
    save_output_official_wb2(ds_output_native, model, target_date, args.steps)
    
    logger.info("✅ Inference Task Complete.")

if __name__ == "__main__":
    main()
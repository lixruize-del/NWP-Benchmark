import os
import sys
import logging
import argparse
import datetime
import numpy as np
import xarray as xr
import dask
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GraphCast.Prepare")

# === 路径配置 ===
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
# 必须把 utils 加入路径才能导入 era5_mirror
sys.path.append(str(CURRENT_DIR))

try:
    from utils.era5_mirror import ERA5Mirror
except ImportError:
    logger.error("无法导入 utils.era5_mirror，请检查文件是否存在")
    sys.exit(1)

# 数据存储路径
DATA_DIR = BASE_DIR / "assets" / "data" / "graphcast"
ZARR_PATH = DATA_DIR / "zarr"
HDF5_PATH = DATA_DIR / "hdf5"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

# === 34 个变量定义 (来自 config_34var.yaml) ===
# 格式: 字符串 或 (变量名, 压力层级)
VARIABLES_34 = [
  "10m_u_component_of_wind",
  "10m_v_component_of_wind",
  "2m_temperature",
  "surface_pressure",
  "mean_sea_level_pressure",
  "total_column_water_vapour",
  "100m_u_component_of_wind",
  "100m_v_component_of_wind",
  ("temperature", 850),
  ("u_component_of_wind", 1000),
  ("v_component_of_wind", 1000),
  ("geopotential", 1000),
  ("u_component_of_wind", 850),
  ("v_component_of_wind", 850),
  ("geopotential", 850),
  ("u_component_of_wind", 500),
  ("v_component_of_wind", 500),
  ("geopotential", 500),
  ("temperature", 500),
  ("geopotential", 50),
  ("relative_humidity", 500),
  ("relative_humidity", 850),
  ("u_component_of_wind", 250),
  ("v_component_of_wind", 250),
  ("geopotential", 250),
  ("temperature", 250),
  ("u_component_of_wind", 100),
  ("v_component_of_wind", 100),
  ("geopotential", 100),
  ("temperature", 100),
  ("u_component_of_wind", 900),
  ("v_component_of_wind", 900),
  ("geopotential", 900),
  ("temperature", 900),
]

def get_target_date():
    if DATE_FILE.exists():
        with open(DATE_FILE) as f:
            return f.read().strip()
    return "2023010112"

def prepare_data(target_date):
    logger.info(f"🚀 开始准备 GraphCast 数据，目标日期 (T0): {target_date}")
    
    # 1. 计算时间范围
    # GraphCast 需要历史数据作为输入 (通常是 T0 和 T-6)
    # 为了保险，我们下载这一整天的数据
    dt = datetime.datetime.strptime(target_date, "%Y%m%d%H")
    date_obj = dt.date()
    
    # 构造 date_range (start, end)
    # ERA5Mirror 需要 (date, date) 格式
    date_range = (date_obj, date_obj)
    
    # 需要的小时 (00, 06, 12, 18)
    hours = [0, 6, 12, 18] 

    # 2. 初始化 Mirror
    # 这里我们只下载，不计算 mean/std (直接用官方提供的)
    mirror = ERA5Mirror(base_path=str(ZARR_PATH))
    
    logger.info("⬇️ 开始下载数据 (通过 CDS API)...")
    try:
        zarr_paths = mirror.download(VARIABLES_34, date_range, hours)
    except Exception as e:
        logger.error(f"下载失败: {e}")
        return

    # 3. 转换为 HDF5 (推理用)
    logger.info("🔄 正在转换为 HDF5 格式...")
    
    # 打开所有 Zarr
    try:
        zarr_arrays = [xr.open_zarr(path) for path in zarr_paths]
        # 合并到一个 xarray Dataset
        era5_ds = xr.concat(
            [z[list(z.data_vars.keys())[0]] for z in zarr_arrays], dim="channel"
        )
        # 调整维度顺序 (Time, Channel, Lat, Lon)
        era5_ds = era5_ds.transpose("time", "channel", "latitude", "longitude")
        era5_ds.name = "fields"
        era5_ds = era5_ds.astype("float32")
        
        # 保存 HDF5
        HDF5_PATH.mkdir(parents=True, exist_ok=True)
        save_path = HDF5_PATH / f"{target_date}.h5"
        
        # 选择特定时间 (T0 和 T-6)
        # 这里的逻辑简化：直接把当天所有数据存进去，inference 时再切片
        logger.info(f"💾 保存到: {save_path}")
        era5_ds.to_netcdf(save_path, engine="h5netcdf")
        
        # 更新日期记录
        with open(DATE_FILE, "w") as f:
            f.write(target_date)
            
        logger.info("✅ 数据准备完成！")
        
    except Exception as e:
        logger.error(f"数据转换失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()
    
    date = args.date if args.date else get_target_date()
    prepare_data(date)
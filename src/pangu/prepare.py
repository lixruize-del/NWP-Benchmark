import os
import logging
import datetime
import numpy as np
import xarray as xr

# === 日志配置 ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Pangu.Prepare")

# === 路径配置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "assets", "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "assets", "data", "processed_pangu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Pangu 常量定义 ===
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

def read_grib_variable(filepath, short_name):
    """
    精确读取 GRIB 文件中的单个变量，避免 heightAboveGround 坐标冲突。
    """
    try:
        # 使用 filter_by_keys 只读取特定的 shortName
        # 这样 cfgrib 就不会因为合并不同高度层的变量而报错
        ds = xr.open_dataset(
            filepath, 
            engine='cfgrib', 
            backend_kwargs={'filter_by_keys': {'shortName': short_name}}
        )
        # ECMWF OpenData 的命名可能在 10u/u10 之间摇摆，做个兼容
        if short_name in ds:
            return ds[short_name].values
        
        # 尝试常见别名
        aliases = {
            '10u': 'u10', '10v': 'v10', 
            '2t': 't2m', 'msl': 'prmsl'
        }
        if short_name in aliases and aliases[short_name] in ds:
            return ds[aliases[short_name]].values
            
        raise ValueError(f"Variable {short_name} not found in {filepath}")
        
    except Exception as e:
        logger.error(f"读取变量 {short_name} 失败: {e}")
        raise e

def process_pangu_data(target_date_str):
    logger.info(f"开始为 Pangu 准备数据: {target_date_str}")
    
    date = datetime.datetime.fromisoformat(target_date_str)
    date_str_file = date.strftime('%Y%m%d%H')
    
    sfc_file = os.path.join(RAW_DATA_DIR, f"raw_{date_str_file}_sfc.grib")
    pl_file = os.path.join(RAW_DATA_DIR, f"raw_{date_str_file}_pl.grib")
    
    if not os.path.exists(sfc_file) or not os.path.exists(pl_file):
        raise FileNotFoundError("原始 GRIB 文件缺失，请先运行 src/common/downloader.py")

    # === 1. 处理高空数据 (Upper) ===
    # 目标: (5, 13, 721, 1440) -> Z, Q, T, U, V
    logger.info("处理高空数据 (Upper Air)...")
    
    try:
        # 高空数据通常层级结构一致，可以直接读取
        # 但为了保险，我们只筛选 isobaricInhPa 层级
        ds_pl = xr.open_dataset(pl_file, engine='cfgrib', 
                               backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
        
        # 提取并确保层级顺序正确
        gh = ds_pl['gh'].sel(isobaricInhPa=LEVELS).values
        q = ds_pl['q'].sel(isobaricInhPa=LEVELS).values
        t = ds_pl['t'].sel(isobaricInhPa=LEVELS).values
        u = ds_pl['u'].sel(isobaricInhPa=LEVELS).values
        v = ds_pl['v'].sel(isobaricInhPa=LEVELS).values
        
        # 转换位势: z = gh * 9.80665
        z = gh * 9.80665
        
        input_upper = np.stack([z, q, t, u, v], axis=0).astype(np.float32)
        logger.info(f"   Upper Shape: {input_upper.shape}")
        
    except Exception as e:
        logger.error(f"处理高空数据失败: {e}")
        exit(1)

    # === 2. 处理地表数据 (Surface) ===
    # 目标: (4, 721, 1440) -> MSLP, U10, V10, T2M
    logger.info("处理地表数据 (Surface)...")
    
    try:
        # 使用精确读取函数，避免坐标冲突
        mslp = read_grib_variable(sfc_file, 'msl')
        u10 = read_grib_variable(sfc_file, '10u')
        v10 = read_grib_variable(sfc_file, '10v')
        t2m = read_grib_variable(sfc_file, '2t')
        
        input_surface = np.stack([mslp, u10, v10, t2m], axis=0).astype(np.float32)
        logger.info(f"   Surface Shape: {input_surface.shape}")
        
    except Exception as e:
        logger.error(f"处理地表数据失败: {e}")
        exit(1)

    # === 3. 保存结果 ===
    upper_path = os.path.join(OUTPUT_DIR, "input_upper.npy")
    surface_path = os.path.join(OUTPUT_DIR, "input_surface.npy")
    
    np.save(upper_path, input_upper)
    np.save(surface_path, input_surface)
    
    logger.info("数据准备完成！")
    logger.info(f"   Upper: {upper_path}")
    logger.info(f"   Surface: {surface_path}")

if __name__ == "__main__":
    date_file = os.path.join(BASE_DIR, "assets", "target_date.txt")
    if not os.path.exists(date_file):
        logger.error("未找到日期文件")
        exit(1)
        
    with open(date_file, "r") as f:
        date_str = f.read().strip()
    
    try:
        process_pangu_data(date_str)
    except Exception as e:
        logger.error(f"程序异常终止: {e}")
        exit(1)
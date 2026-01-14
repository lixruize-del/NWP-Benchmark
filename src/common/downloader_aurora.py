import os
import argparse
import logging
import cdsapi
import pandas as pd

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Common.DownloaderAurora")

# 路径配置 (自动计算根目录) 
# 当前文件在 src/common/, 向上回溯三级找到 nwpbench 根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(BASE_DIR, "assets", "data", "era5_aurora")
# 日期记录文件
DATE_FILE_PATH = os.path.join(BASE_DIR, "assets", "target_date.txt")

# 确保目录存在
os.makedirs(SAVE_DIR, exist_ok=True)

# Aurora 需要的 13 个层级
PRESSURE_LEVELS = [
    '50', '100', '150', '200', '250', '300', '400', 
    '500', '600', '700', '850', '925', '1000'
]

def download_static(client):
    """下载静态变量 (地形、海陆掩码等) - 只需下载一次"""
    target = os.path.join(SAVE_DIR, "static.nc")
    if os.path.exists(target):
        logger.info("静态变量已存在，跳过。")
        return

    logger.info("正在下载静态变量...")
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'geopotential', 'land_sea_mask', 'soil_type',
                'angle_of_sub_gridscale_orography', 
                'slope_of_sub_gridscale_orography',
                'standard_deviation_of_filtered_subgrid_orography',
                'standard_deviation_of_orography'
            ],
            # 静态变量相对固定，这里硬编码一个日期即可
            'year': '2023', 'month': '01', 'day': '01', 'time': '00:00',
            'grid': [0.25, 0.25], # 强制 0.25 度
        },
        target
    )

def download_surface(client, date_str):
    """下载地面变量 (含 T0 和 T-6)"""
    target = os.path.join(SAVE_DIR, f"surface_{date_str}.nc")
    if os.path.exists(target):
        logger.info(f"地面变量 {date_str} 已存在。")
        return

    dt = pd.to_datetime(date_str, format="%Y%m%d%H")
    times_to_download = [dt, dt - pd.Timedelta(hours=6)]
    
    # 解析请求参数
    years = list(set([t.strftime("%Y") for t in times_to_download]))
    months = list(set([t.strftime("%m") for t in times_to_download]))
    days = list(set([t.strftime("%d") for t in times_to_download]))
    times = list(set([t.strftime("%H:%M") for t in times_to_download]))

    logger.info(f"正在下载地面变量: {date_str}...")
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '2m_temperature', '10m_u_component_of_wind', 
                '10m_v_component_of_wind', 'mean_sea_level_pressure'
            ],
            'year': years, 'month': months, 'day': days, 'time': times,
            'grid': [0.25, 0.25],
        },
        target
    )

def download_upper(client, date_str):
    """下载高空变量 (含 T0 和 T-6)"""
    target = os.path.join(SAVE_DIR, f"upper_{date_str}.nc")
    if os.path.exists(target):
        logger.info(f"高空变量 {date_str} 已存在。")
        return

    dt = pd.to_datetime(date_str, format="%Y%m%d%H")
    times_to_download = [dt, dt - pd.Timedelta(hours=6)]
    
    years = list(set([t.strftime("%Y") for t in times_to_download]))
    months = list(set([t.strftime("%m") for t in times_to_download]))
    days = list(set([t.strftime("%d") for t in times_to_download]))
    times = list(set([t.strftime("%H:%M") for t in times_to_download]))

    logger.info(f"正在下载高空变量: {date_str} (数据量较大，请耐心等待)...")
    client.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'geopotential', 'specific_humidity', 'temperature',
                'u_component_of_wind', 'v_component_of_wind'
            ],
            'pressure_level': PRESSURE_LEVELS,
            'year': years, 'month': months, 'day': days, 'time': times,
            'grid': [0.25, 0.25],
        },
        target
    )


def load_cds_config():
    """尝试从项目根目录读取 .cdsapirc"""
    # 假设 .cdsapirc 在 nwpbench 根目录下
    project_config = os.path.join(BASE_DIR, ".cdsapirc")
    
    url = "https://cds.climate.copernicus.eu/api"
    key = None

    # 1. 优先读取项目目录下的文件
    if os.path.exists(project_config):
        logger.info(f"读取项目配置文件: {project_config}")
        with open(project_config, 'r') as f:
            for line in f:
                if line.startswith("url:"):
                    url = line.split(":", 1)[1].strip()
                if line.startswith("key:"):
                    key = line.split(":", 1)[1].strip()
    
    # 2. 如果没找到，检查环境变量 (CDSAPI_KEY)
    if key is None:
        key = os.environ.get("CDSAPI_KEY")
        
    return url, key

def main(target_date):
    logger.info(f"开始处理 Aurora/ERA5 数据任务，目标日期: {target_date}")
    
    # 记录日期到文件，供推理脚本读取
    os.makedirs(os.path.dirname(DATE_FILE_PATH), exist_ok=True)
    with open(DATE_FILE_PATH, "w") as f:
        f.write(target_date)
        
    url, key = load_cds_config()
    
    if not key:
        logger.error("未找到 API Key！")
        logger.error(f"请在 {BASE_DIR} 下创建 .cdsapirc 文件，或设置 CDSAPI_KEY 环境变量")
        return

    # 显式初始化 Client
    c = cdsapi.Client(url=url, key=key)
    
    try:
        download_static(c)
        download_surface(c, target_date)
        download_upper(c, target_date)
        logger.info(f"所有数据下载完成！保存路径: {SAVE_DIR}")
    except Exception as e:
        logger.error(f"下载失败: {e}")
        logger.error("提示: 1. 检查 ~/.cdsapirc 是否配置")
        logger.error("      2. 确保请求的日期不是未来时间 (ERA5 有5天延迟)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aurora ERA5 Data Downloader")
    # 默认给一个 2023 年的历史日期，因为 ERA5 无法下载实时数据
    parser.add_argument("--date", type=str, default="2025120418", help="指定日期 (YYYYMMDDHH)")
    args = parser.parse_args()
    
    main(args.date)
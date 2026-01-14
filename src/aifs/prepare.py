import os
import datetime
import logging
import numpy as np
import earthkit.data as ekd
import earthkit.regrid as ekr
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AIFS.Prepare")

# 配置路径
# 回溯三级: src/aifs/ -> src/ -> nwpbench/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "assets", "data", "raw")
# 处理后的数据存放于 assets/data/processed_aifs
OUTPUT_DIR = os.path.join(BASE_DIR, "assets", "data", "processed_aifs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 常量定义
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["vsw", "sot"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1, 2]

def process_data(target_date_str):
    logger.info(f"开始处理数据: {target_date_str}")
    date = datetime.datetime.fromisoformat(target_date_str)
    fields = {}

    def fetch_and_process(file_tag, param, levelist=None):
        buffer = defaultdict(list)
        # AIFS 需要 T-6 和 T0 两个时间步
        for d in [date - datetime.timedelta(hours=6), date]:
            filename = f"raw_{d.strftime('%Y%m%d%H')}_{file_tag}.grib"
            filepath = os.path.join(RAW_DATA_DIR, filename)
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"本地文件缺失: {filepath}")
            
            logger.info(f"读取文件: {filename} ...")
            ds = ekd.from_source("file", filepath)
            
            # 筛选参数
            ds = ds.sel(param=param)
            if levelist:
                ds = ds.sel(levelist=levelist)
            
            for f in ds:
                # 1. 经度滚动 (-180/180 -> 0/360)
                val = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
                
                # 2. 网格插值 (0.25 -> N320)
                # 这是 CPU 密集型操作
                val = ekr.interpolate(val, {"grid": (0.25, 0.25)}, {"grid": "N320"})
                
                # 构造键名
                p = f.metadata('param')
                l = f.metadata('levelist')
                name = f"{p}_{l}" if l is not None else p
                
                buffer[name].append(val)
        
        # 堆叠时间维度 -> (2, N_grid)
        return {k: np.stack(v) for k, v in buffer.items()}

    logger.info("处理地表变量 (Surface)...")
    fields.update(fetch_and_process("sfc", PARAM_SFC))

    logger.info("处理土壤变量 (Soil)...")
    soil_data = fetch_and_process("soil", PARAM_SOIL, levelist=SOIL_LEVELS)
    
    # 变量重命名映射
    mapping = {
        'sot_1': 'stl1', 'sot_2': 'stl2',
        'vsw_1': 'swvl1', 'vsw_2': 'swvl2'
    }
    
    for k, v in soil_data.items():
        if k in mapping:
            fields[mapping[k]] = v
        else:
            logger.warning(f"忽略未知土壤变量: {k}")

    logger.info("处理高空变量 (Pressure Levels)...")
    fields.update(fetch_and_process("pl", PARAM_PL, levelist=LEVELS))

    # 转换位势高度 (Geopotential Height -> Geopotential)
    for level in LEVELS:
        key = f"gh_{level}"
        if key in fields:
            gh = fields.pop(key)
            fields[f"z_{level}"] = gh * 9.80665

    # 保存结果
    output_filename = f"init_{date.strftime('%Y%m%d_%H')}.npz"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    logger.info(f"保存处理结果至: {output_path}")
    np.savez(output_path, date=str(date), **fields)

if __name__ == "__main__":
    # 读取之前下载脚本保存的日期
    date_file = os.path.join(BASE_DIR, "assets", "target_date.txt")
    if not os.path.exists(date_file):
        logger.error("未找到日期文件，请先运行 src/common/downloader.py")
        exit(1)
        
    with open(date_file, "r") as f:
        date_str = f.read().strip()
    
    try:
        process_data(date_str)
        logger.info("数据预处理完成")
    except Exception as e:
        logger.error(f"处理失败: {e}")
        exit(1)
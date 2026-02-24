import os
import argparse
import logging
import datetime
from ecmwf.opendata import Client as OpendataClient

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Common.Downloader")

# 配置路径
# 当前文件在 src/common/, 需要向上回溯三级找到项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "assets", "data", "raw")

# 确保目录存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def download_raw_data(target_date_str=None):
    """
    下载 ECMWF 原始数据。
    如果未指定日期，默认下载最新可用数据。
    """
    logger.info("开始下载 ECMWF 原始数据...")
    client = OpendataClient()
    
    if target_date_str:
        try:
            date = datetime.datetime.strptime(target_date_str, "%Y%m%d%H")
            logger.info(f"使用指定日期: {date}")
        except ValueError:
            logger.error("日期格式错误，请使用 YYYYMMDDHH 格式")
            return
    else:
        date = client.latest()
        logger.info(f"自动获取最新日期: {date}")
    
    # 保存日期供后续脚本使用
    date_file_path = os.path.join(BASE_DIR, "assets", "target_date.txt")
    os.makedirs(os.path.dirname(date_file_path), exist_ok=True)
    with open(date_file_path, "w") as f:
        f.write(str(date))

    # 定义要下载的参数 (与官方 Notebook 一致)
    requests = [
        # Surface
        dict(tag="sfc", date=date, time=date.hour, step=0, type="fc", levtype="sfc", 
             param=["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]),
        # Soil
        # 注意：这里 levtype 为 sol，且指定了 levelist
        dict(tag="soil", date=date, time=date.hour, step=0, type="fc", levtype="sol", 
             levelist=[1, 2],
             param=["vsw", "sot"]),
        # Pressure Levels
        dict(tag="pl", date=date, time=date.hour, step=0, type="fc", levtype="pl", 
             levelist=[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
             param=["gh", "t", "u", "v", "w", "q"])
    ]
       
    prev_date = date - datetime.timedelta(hours=6)
    
    for req in requests:
        # 提取 tag 并从请求参数中移除 (retrieve 不接受 tag 参数)
        tag = req.pop('tag')
                
        # 下载当前时刻 T0
        filename_t0 = f"raw_{date.strftime('%Y%m%d%H')}_{tag}.grib"
        target_t0 = os.path.join(RAW_DATA_DIR, filename_t0)
        
        if not os.path.exists(target_t0):
            logger.info(f"下载 T0 ({tag}): {filename_t0} ...")
            try:
                client.retrieve(**req, target=target_t0)
            except Exception as e:
                logger.error(f"下载失败 T0 ({tag}): {e}")
        else:
            logger.info(f"文件已存在 T0 ({tag}): {filename_t0}")
            
        # 下载上一时刻 T-6
        req_prev = req.copy()
        req_prev['date'] = prev_date
        req_prev['time'] = prev_date.hour
        
        filename_t6 = f"raw_{prev_date.strftime('%Y%m%d%H')}_{tag}.grib"
        target_t6 = os.path.join(RAW_DATA_DIR, filename_t6)
        
        if not os.path.exists(target_t6):
            logger.info(f"下载 T-6 ({tag}): {filename_t6} ...")
            try:
                client.retrieve(**req_prev, target=target_t6)
            except Exception as e:
                logger.error(f"下载失败 T-6 ({tag}): {e}")
        else:
            logger.info(f"文件已存在 T-6 ({tag}): {filename_t6}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Common Data Downloader")
    parser.add_argument("--date", type=str, default=None, help="指定日期 (YYYYMMDDHH)，默认下载最新")
    args = parser.parse_args()
    
    download_raw_data(args.date)
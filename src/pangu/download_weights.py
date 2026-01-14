import os
import logging
from huggingface_hub import hf_hub_download

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Pangu.Weights")

# 配置路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(BASE_DIR, "assets", "weights", "pangu")

# 确保目录存在
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def download_pangu_weights():
    """
    下载 Pangu-Weather 24小时 ONNX 模型。
    """
    filename = "pangu_weather_24.onnx"
    target_path = os.path.join(WEIGHTS_DIR, filename)
    
    if os.path.exists(target_path):
        logger.info(f"权重已存在，跳过下载: {target_path}")
        return

    logger.info(f"开始下载 Pangu-Weather 权重 (约 1.1GB)...")
    logger.info(f"目标路径: {WEIGHTS_DIR}")

    # 设置国内镜像加速
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 尝试多个备选仓库
    repos = [
        "BitRail/Pangu-Weather",  # 备选 1
        "PowerL/Pangu-Weather",   # 备选 2
        "qq1990/Pangu"            # 备选 3 (国内源常用)
    ]

    success = False
    for repo in repos:
        try:
            logger.info(f"尝试从仓库 {repo} 下载...")
            downloaded_path = hf_hub_download(
                repo_id=repo, 
                filename=filename,
                local_dir=WEIGHTS_DIR,
                local_dir_use_symlinks=False
            )
            logger.info(f"下载完成: {downloaded_path}")
            success = True
            break
        except Exception as e:
            logger.warning(f"从 {repo} 下载失败: {e}")
            continue
    
    if not success:
        logger.error("所有仓库均下载失败！请检查网络或手动上传文件。")
        exit(1)

if __name__ == "__main__":
    download_pangu_weights()
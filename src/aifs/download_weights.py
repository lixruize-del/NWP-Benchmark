import os
import logging
from huggingface_hub import hf_hub_download

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AIFS.Weights")

# 配置路径
# 当前文件在 src/aifs/, 需要向上回溯三级找到项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# AIFS 权重独立存放于 assets/weights/aifs
WEIGHTS_DIR = os.path.join(BASE_DIR, "assets", "weights", "aifs")

# 确保目录存在
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def download_weights():
    filename = "aifs-single-mse-1.0.ckpt"
    target = os.path.join(WEIGHTS_DIR, filename)
    
    if os.path.exists(target):
        logger.info(f"权重已存在: {target}")
        return
    
    logger.info(f"开始下载 AIFS 模型权重至 {WEIGHTS_DIR} ...")
    
    # 设置国内镜像加速
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    try:
        hf_hub_download(
            repo_id="ecmwf/aifs-single-1.0", 
            filename=filename, 
            local_dir=WEIGHTS_DIR,
            local_dir_use_symlinks=False
        )
        logger.info("下载完成")
    except Exception as e:
        logger.error(f"下载失败: {e}")

if __name__ == "__main__":
    download_weights()
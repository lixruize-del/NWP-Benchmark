import os
import logging
from pathlib import Path

# ==============================================================================
# Environment Setup (Must be before importing huggingface_hub)
# ==============================================================================
# Set Hugging Face mirror endpoint for better connectivity in certain regions
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

# ==============================================================================
# Configuration & Setup
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("AIFS.DownloadWeights")

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "aifs"

def download_weights():
    """
    Download AIFS model weights from Hugging Face.
    Repo: ecmwf/aifs-single-1.0
    File: aifs-single-mse-1.0.ckpt
    """
    # Ensure directory exists
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    filename = "aifs-single-mse-1.1.ckpt"
    repo_id = "ecmwf/aifs-single-1.1"
    target_path = WEIGHTS_DIR / filename
    
    if target_path.exists():
        logger.info(f"Weights already exist at: {target_path}")
        return
    
    logger.info(f"Downloading AIFS weights from {repo_id}...")
    logger.info(f"Target path: {target_path}")
    
    try:
        hf_hub_download(
            repo_id=repo_id, 
            filename=filename, 
            local_dir=WEIGHTS_DIR,
            local_dir_use_symlinks=False
        )
        logger.info("Download completed successfully.")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Tip: If download fails, check your network connection or try manually downloading from:")
        logger.info(f"https://huggingface.co/{repo_id}/blob/main/{filename}")

if __name__ == "__main__":
    download_weights()
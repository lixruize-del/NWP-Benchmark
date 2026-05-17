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

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Stormer.DownloadWeights")

# Configure Paths
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "stormer"

def download_stormer_weights():
    logger.info(f"Weights directory: {WEIGHTS_DIR}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    repo_id = "tungnd/stormer"
    filename = "stormer_1.40625_patch_size_2.ckpt"
    target_file = WEIGHTS_DIR / filename
    
    if target_file.exists():
        logger.info(f"Weights file already exists: {target_file}")
        return

    logger.info(f"Downloading {filename} from HuggingFace...")
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

if __name__ == "__main__":
    download_stormer_weights()
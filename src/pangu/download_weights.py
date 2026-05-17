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
logger = logging.getLogger("PanguWeather.DownloadWeights")

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "pangu"

# Hugging Face repository information
REPO_ID = "qq1990/Pangu"
FILENAMES = [
    "pangu_weather_1.onnx",
    "pangu_weather_3.onnx",
    "pangu_weather_6.onnx",
    "pangu_weather_24.onnx"
]

def download_weights():
    """
    Download Pangu-Weather model weights from Hugging Face.
    Repo: qq1990/Pangu
    Files: pangu_weather_{1,3,6,24}.onnx
    """
    # Ensure directory exists
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Weights will be saved to: {WEIGHTS_DIR}")

    all_success = True
    for filename in FILENAMES:
        target_path = WEIGHTS_DIR / filename
        if target_path.exists():
            logger.info(f"File already exists, skipping: {filename}")
            continue

        logger.info(f"Downloading {filename} from {REPO_ID}...")
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=WEIGHTS_DIR,
                local_dir_use_symlinks=False
            )
            logger.info(f"Download completed: {filename}")
        except Exception as e:
            logger.error(f"Download failed for {filename}: {e}")
            all_success = False

    if all_success:
        logger.info("All Pangu-Weather weights downloaded successfully.")
    else:
        logger.warning("Some files failed to download. You can manually download from:")
        logger.warning(f"https://huggingface.co/{REPO_ID}/tree/main")
        logger.warning(f"and place them in: {WEIGHTS_DIR}")

if __name__ == "__main__":
    download_weights()
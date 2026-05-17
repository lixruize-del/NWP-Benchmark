import os
import logging
import gcsfs
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NeuralGCM.DownloadWeights")

# Path configuration
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "neuralgcm"

def download_weights():
    """
    Downloads NeuralGCM weights from the public Google Cloud bucket.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # NeuralGCM 1.4 degree stochastic model
    gcs_path = "gs://gs://neuralgcm/models/v1/stochastic_1_4_deg.pkl"
    local_filename = "models_v1_stochastic_1_4_deg.pkl"
    local_path = WEIGHTS_DIR / local_filename

    if local_path.exists():
        logger.info(f"Weights already exist at: {local_path}")
        return

    logger.info(f"Downloading weights from {gcs_path}...")
    
    try:
        # Use anonymous authentication for public buckets
        fs = gcsfs.GCSFileSystem(token='anon')
        fs.get(gcs_path, str(local_path))
        logger.info(f"Download complete. Saved to: {local_path}")
    except Exception as e:
        logger.error(f"Download failed: {e}")

if __name__ == "__main__":
    download_weights()
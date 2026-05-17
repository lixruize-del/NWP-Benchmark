import os
import logging
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm  # for progress bar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FuXi.DownloadWeights")

# Path configuration
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "fuxi"

# Zenodo record information – single ZIP containing all model variants
ZENODO_RECORD_ID = "10401602"
ZIP_FILENAME = "FuXi_EC.zip"
DOWNLOAD_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{ZIP_FILENAME}?download=1"

# After extraction, the ZIP creates folders: short/, medium/, long/
# We'll check for the presence of one key model file inside a subfolder
CHECK_PATH = WEIGHTS_DIR / "long" / "long.onnx"

def download_file(url: str, local_path: Path, chunk_size: int = 8192) -> None:
    """
    Download a file from a URL with streaming support and a tqdm progress bar.
    """
    logger.info(f"Downloading from: {url}")

    # Use a browser-like User-Agent to be polite
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()  # Raise an error for bad status codes

    total_size = int(response.headers.get("content-length", 0))
    logger.info(f"Downloading {local_path.name} ({total_size / (1024**3):.2f} GB)")

    # Use tqdm for a progress bar
    progress = tqdm(total=total_size, unit="B", unit_scale=True, desc=local_path.name)

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress.update(len(chunk))

    progress.close()
    logger.info(f"Download complete: {local_path}")

def download_weights():
    """
    Downloads FuXi model weights from Zenodo and extracts them.
    The ZIP contains folders short/, medium/, long/ each with .onnx files inside.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # If the key model file already exists, skip download
    if CHECK_PATH.exists():
        logger.info(f"Model weights already exist at {CHECK_PATH}. Skipping download.")
        return

    zip_path = WEIGHTS_DIR / ZIP_FILENAME

    # Download the ZIP file
    logger.info("Starting FuXi model download...")
    try:
        download_file(DOWNLOAD_URL, zip_path)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        # Clean up partial download if it exists
        if zip_path.exists():
            zip_path.unlink()
        return

    # Extract the ZIP file
    logger.info(f"Extracting {ZIP_FILENAME} to {WEIGHTS_DIR} ...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(WEIGHTS_DIR)
        logger.info("Extraction completed.")
    except zipfile.BadZipFile as e:
        logger.error(f"Extraction failed: {e} (the downloaded file may be corrupted)")
        zip_path.unlink()
        return
    except Exception as e:
        logger.error(f"Unexpected error during extraction: {e}")
        return

    # Remove the ZIP file to save space (optional)
    try:
        zip_path.unlink()
        logger.info(f"Removed temporary file {ZIP_FILENAME}.")
    except OSError as e:
        logger.warning(f"Could not remove {ZIP_FILENAME}: {e}")

    # Final verification
    if CHECK_PATH.exists():
        logger.info(f"FuXi model weights are ready at {WEIGHTS_DIR}")
    else:
        logger.error("Extraction completed but expected file not found. Something may be wrong.")

if __name__ == "__main__":
    download_weights()
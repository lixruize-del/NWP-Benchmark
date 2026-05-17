import os
import logging
import requests
from pathlib import Path
from tqdm import tqdm  # optional, for progress bar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FengWu.DownloadWeights")

# Path configuration
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "fengwu"

# OneDrive links for the two models
MODEL_URLS = {
    "fengwu_v1.onnx": "https://pjlab-my.sharepoint.cn/:u:/g/personal/chenkang_pjlab_org_cn/EVA6V_Qkp6JHgXwAKxXIzDsBPIddo5RgDtGCBQ-sQbMmwg",
    "fengwu_v2.onnx": "https://pjlab-my.sharepoint.cn/:u:/g/personal/chenkang_pjlab_org_cn/EZkFM7nQcEtBve6MsqlWaeIB_lmpa__hX0I8QYOPzf-X6A"
}

def download_file(url: str, local_path: Path, chunk_size: int = 8192) -> None:
    """
    Download a file from a URL with streaming support.
    Tries two common OneDrive download patterns: ?download=1 and /:u:/download.
    """
    # Try adding ?download=1 first
    download_url = url + "?download=1"
    logger.info(f"Attempting download from: {download_url}")

    # Use a browser-like User-Agent to avoid being blocked
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(download_url, headers=headers, stream=True)
    if response.status_code != 200:
        # Try alternative pattern: replace /:u:/ with /:u:/download
        alt_url = url.replace("/:u:/", "/:u:/download")
        logger.info(f"First attempt failed, trying: {alt_url}")
        response = requests.get(alt_url, headers=headers, stream=True)

    if response.status_code != 200:
        raise Exception(f"Failed to download from {url} (status code: {response.status_code})")

    total_size = int(response.headers.get("content-length", 0))
    logger.info(f"Downloading {local_path.name} ({total_size / (1024**3):.2f} GB)")

    # Use tqdm for a progress bar if available, otherwise simple write
    try:
        from tqdm import tqdm
        progress = tqdm(total=total_size, unit="B", unit_scale=True, desc=local_path.name)
    except ImportError:
        progress = None
        logger.info("Install tqdm for a progress bar: pip install tqdm")

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                if progress:
                    progress.update(len(chunk))

    if progress:
        progress.close()

    logger.info(f"Download complete: {local_path}")

def download_weights():
    """
    Downloads FengWu weights from OneDrive public share links.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in MODEL_URLS.items():
        local_path = WEIGHTS_DIR / filename

        if local_path.exists():
            logger.info(f"Weights already exist at: {local_path}")
            continue

        logger.info(f"Downloading {filename}...")
        try:
            download_file(url, local_path)
        except Exception as e:
            logger.error(f"Download failed for {filename}: {e}")

if __name__ == "__main__":
    download_weights()
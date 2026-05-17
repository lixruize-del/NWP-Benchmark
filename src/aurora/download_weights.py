import os
import argparse
from pathlib import Path

# Set Hugging Face mirror endpoint for better connectivity in certain regions
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


from huggingface_hub import hf_hub_download

# Path configuration
CURRENT_DIR = Path(__file__).resolve().parent
# Backtrack to the nwpbench root directory (src/aurora -> src -> nwpbench)
BASE_DIR = CURRENT_DIR.parent.parent
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "aurora"

def download_aurora_weights():
    print(f"Weights save directory: {WEIGHTS_DIR}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    repo_id = "microsoft/aurora"
    # filename = "aurora-0.25-pretrained.ckpt"
    filename = "aurora-0.1-finetuned.ckpt"
    target_file = WEIGHTS_DIR / filename
    
    if target_file.exists():
        print(f"Weights file already exists: {target_file}")
        return

    print(f"Downloading {filename} ...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=WEIGHTS_DIR,
        local_dir_use_symlinks=False
    )
    print("Download completed!")

if __name__ == "__main__":
    download_aurora_weights()
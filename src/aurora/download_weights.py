import os
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

# 路径配置
CURRENT_DIR = Path(__file__).resolve().parent
# 回溯到 nwpbench 根目录 (src/aurora -> src -> nwpbench)
BASE_DIR = CURRENT_DIR.parent.parent
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "aurora"

def download_aurora_weights():
    print(f"权重保存目录: {WEIGHTS_DIR}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    repo_id = "microsoft/aurora"
    filename = "aurora-0.25-pretrained.ckpt"
    
    target_file = WEIGHTS_DIR / filename
    
    if target_file.exists():
        print(f"权重文件已存在: {target_file}")
        return

    print(f"开始下载 {filename} ...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=WEIGHTS_DIR,
        local_dir_use_symlinks=False
    )
    print("下载完成！")

if __name__ == "__main__":
    download_aurora_weights()
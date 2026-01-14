import sys
import os
import argparse
from pathlib import Path

# === 路径配置 ===
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent

# 将 src 加入路径，以便导入 common
sys.path.append(str(BASE_DIR / "src"))

try:
    from common.downloader_aurora import main as download_era5
except ImportError:
    print("❌ 无法导入 src.common.downloader_aurora，请检查文件位置")
    sys.exit(1)

def prepare_data(target_date):
    """
    准备 Aurora 所需的 ERA5 数据。
    """
    print(f"🔄 [Aurora Prepare] 正在为日期 {target_date} 准备数据...")
    
    # 调用 common 中的下载逻辑
    # 注意：该脚本会自动下载 Static, Surface, Upper 变量到 assets/data/era5_aurora
    download_era5(target_date)
    
    # 更新 target_date.txt (保持与 aifs/pangu 逻辑一致)
    date_file = BASE_DIR / "assets" / "target_date.txt"
    with open(date_file, "w") as f:
        f.write(target_date)
    
    print(f"✅ 数据准备就绪，日期已更新为: {target_date}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认使用一个历史日期，因为 ERA5 无法下载实时数据
    parser.add_argument("--date", type=str, default="2023010112", help="YYYYMMDDHH")
    args = parser.parse_args()
    
    prepare_data(args.date)
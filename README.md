# NWPBench: AI-based Numerical Weather Prediction Benchmark

**NWPBench** is a unified framework for evaluating state-of-the-art AI weather forecasting models (e.g., Stormer, Aurora, GraphCast, Pangu-Weather) on standardized datasets.

## 🚀 Supported Models

| Model | Status | Resolution | Input Format |
| :--- | :--- | :--- | :--- |
| **Stormer** | ✅ Ready | 1.40625° (128x256) | GRIB / NetCDF |
| **Aurora** | ✅ Ready | 0.25° (721x1440) | NetCDF (ERA5) |
| **Pangu-Weather** | ✅ Ready | 0.25° | - |
| **Aifs** | ✅ Ready | 0.25° | - |
| **GraphCast** | 🚧 In Progress | 0.25° | - |

## 📂 Directory Structure

The project follows a standardized structure for easy extension:

```text
nwpbench/
├── assets/
│   ├── data/                 # Input data storage
│   │   ├── raw/              # Raw GRIB files for Stormer
│   │   └── era5_aurora/      # NetCDF files for Aurora
│   └── weights/              # Model checkpoints
├── configs/                  # Conda environment files
├── outputs/                  # Inference results (.nc files)
├── src/
│   ├── common/               # Shared utilities (Saver, Downloader)
│   ├── stormer/              # Stormer implementation
│   ├── aifs/                 # Aifs implementation
│   ├── pangu/                # Pangu-Weather implementation
│   └── aurora/               # Aurora implementation
└── README.md

🛠️ Installation
We recommend using Conda to manage environments for different models due to dependency conflicts.

1. Stormer Environment
code
Bash
conda env create -f configs/stormer_environment.yaml
conda activate stormer
# Note: Ensure compatible xformers/flash-attn are installed for your GPU (e.g. A100).

2. Aurora Environment
code
Bash
conda env create -f configs/aurora_environment.yaml
conda activate aurora_env

⚡ Quick Start (Inference)

1. Download Model Weights
Run the download script to fetch checkpoints from HuggingFace automatically:
code
Bash
# For Stormer
python src/stormer/download_weights.py

# For Aurora
python src/aurora/download_weights.py

2. Prepare Data
If you need to download ERA5 data for a specific date (e.g., 2023010112):
code
Bash
# For Stormer (Downloads GRIB)
# Note: Requires configured .cdsapirc or ECMWF credentials
python src/stormer/prepare.py --date 2023010112

# For Aurora (Downloads NetCDF)
python src/aurora/prepare.py --date 2023010112
Note: You can also manually place your data in assets/data/. Check src/<model>/inference.py for expected file paths.

3. Run Inference
The inference scripts will load data, run the model, and save standardized NetCDF outputs to outputs/<model>/.
Run Stormer:
code
Bash
conda activate stormer
python src/stormer/inference.py --date 2023010112
Run Aurora:
code
Bash
conda activate aurora_env
python src/aurora/inference.py --date 2023010112

📊 Output Format
All inference results are saved in NetCDF format under outputs/.
File naming convention: YYYY-MMDD-LL.nc (e.g., 2023-0101-06.nc for 6h lead time).
Output variables include both surface and pressure-level data, complying with CF conventions:
Surface: t2m, msl, u10, v10, tp, etc.
Upper Air: z, t, u, v, q (at levels 50, 100, ..., 1000 hPa).

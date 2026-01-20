This is the updated, comprehensive **README.md**.

I have incorporated **AIFS** and **Pangu-Weather**, updated the **Installation** and **Inference** sections, and clarified the data downloading logic based on your description (using `src/common/downloader.py` vs `downloader_aurora.py`).

You can copy this directly into your `README.md` file.

***

# NWPBench: AI-based Numerical Weather Prediction Benchmark

**NWPBench** is a unified framework for evaluating state-of-the-art AI weather forecasting models (e.g., Stormer, Aurora, GraphCast, Pangu-Weather, AIFS) on standardized datasets.

## 🚀 Supported Models

| Model | Status | Resolution | Input Format |
| :--- | :--- | :--- | :--- |
| **Stormer** | ✅ Ready | 1.40625° (128x256) | GRIB / NetCDF |
| **Aurora** | ✅ Ready | 0.25° (721x1440) | NetCDF (ERA5) |
| **AIFS** | ✅ Ready | 0.25° | GRIB / Zarr |
| **Pangu-Weather** | ✅ Ready | 0.25° | GRIB / NetCDF |
| **GraphCast** | 🚧 In Progress | 0.25° | - |
| **NeuralGCM** | 🚧 In Progress | 0.25° | - |

## 📂 Directory Structure

The project follows a standardized structure for easy extension:

```text
nwpbench/
├── assets/
│   ├── data/                 # Input data storage
│   │   ├── raw/              # Common GRIB files (Stormer/AIFS/Pangu)
│   │   └── era5_aurora/      # NetCDF files for Aurora
│   └── weights/              # Model checkpoints
├── configs/                  # Conda environment files
├── outputs/                  # Inference results (.nc files)
├── src/
│   ├── common/               # Shared utilities
│   │   ├── downloader.py     # Common GRIB downloader (ECMWF Open Data)
│   │   ├── downloader_aurora.py # Aurora-specific downloader (CDS API)
│   │   └── saver.py          # Unified NetCDF saver
│   ├── stormer/              # Stormer implementation
│   ├── aifs/                 # AIFS implementation
│   ├── pangu/                # Pangu-Weather implementation
│   └── aurora/               # Aurora implementation
└── README.md
```

## 🛠️ Installation

We recommend using **Conda** to manage separate environments for different models due to dependency conflicts (e.g., specific PyTorch/CUDA versions).

### 1. Stormer Environment
```bash
conda env create -f configs/stormer_environment.yaml
conda activate stormer
# Note: Ensure compatible xformers/flash-attn are installed for your GPU.
```

### 2. Aurora Environment
```bash
conda env create -f configs/aurora_environment.yaml
conda activate aurora_env
```

### 3. AIFS Environment
```bash
conda env create -f configs/aifs_environment.yaml
conda activate aifs
```

### 4. Pangu-Weather Environment
```bash
conda env create -f configs/pangu_environment.yaml
conda activate pangu
```

---

## ⚡ Quick Start (Inference)

### 1. Download Model Weights
Run the download scripts to fetch checkpoints automatically:

```bash
# For Stormer
python src/stormer/download_weights.py

# For Aurora
python src/aurora/download_weights.py

# For AIFS
python src/aifs/download_weights.py

# For Pangu-Weather
python src/pangu/download_weights.py
```

### 2. Prepare Data
Download and preprocess data for a specific date (e.g., `2023010112`).
*   **Stormer, AIFS, Pangu** use `src/common/downloader.py` (GRIB format).
*   **Aurora** uses `src/common/downloader_aurora.py` (NetCDF format).

You can run the model-specific prepare scripts, which wrap these downloaders:

```bash
# For Stormer (Uses common downloader)
python src/stormer/prepare.py --date 2023010112

# For AIFS
python src/aifs/prepare.py --date 2023010112

# For Pangu-Weather
python src/pangu/prepare.py --date 2023010112

# For Aurora (Uses Aurora-specific downloader)
python src/aurora/prepare.py --date 2023010112
```

### 3. Run Inference
The inference scripts will load data, run the model, and save standardized NetCDF outputs to `outputs/<model>/`.

**Run Stormer:**
```bash
conda activate stormer
python src/stormer/inference.py --date 2023010112
```

**Run Aurora:**
```bash
conda activate aurora_env
python src/aurora/inference.py --date 2023010112
```

**Run AIFS:**
```bash
conda activate aifs
python src/aifs/inference.py --date 2023010112
```

**Run Pangu-Weather:**
```bash
conda activate pangu
python src/pangu/inference.py --date 2023010112
```

---

## 📊 Output Format

All inference results are saved in **NetCDF** format under `outputs/`.

*   **File naming convention**: `YYYY-MMDD-LL.nc` (e.g., `2023-0101-06.nc` for a 6h lead time).
*   **Variables**: Output variables include both surface and pressure-level data, complying with CF conventions:
    *   **Surface**: `t2m`, `msl`, `u10`, `v10`, `tp`, etc.
    *   **Upper Air**: `z`, `t`, `u`, `v`, `q` (at levels 50, 100, ..., 1000 hPa).

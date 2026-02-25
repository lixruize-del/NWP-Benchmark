# NWPBench: AI-based Numerical Weather Prediction Benchmark

**NWPBench** is a unified framework for evaluating state-of-the-art AI weather forecasting models (e.g., Stormer, Aurora, GraphCast, Pangu-Weather, AIFS) on standardized datasets.

## 🚀 Supported Models

| Model | Status | Resolution | Input Format |
| :--- | :--- | :--- | :--- |
| **Stormer** | 🚧 In Progress | 1.40625° (128x256) | GRIB / NetCDF |
| **Aurora** | ✅ Ready | 0.25° (721x1440) | NetCDF (ERA5) |
| **AIFS** | ✅ Ready | 0.25° | GRIB / Zarr |
| **Pangu-Weather** | ✅ Ready | 0.25° | GRIB / NetCDF |
| **GraphCast** | 🚧 In Progress | 0.25° | - |
| **NeuralGCM** | ✅ Ready | 0.25° | - |

## 📂 Directory Structure

The project follows a standardized structure for easy extension:

```text
nwpbench/
├── assets/
│   ├── data/
│   │   └── raw/                 # Unified ERA5 NetCDF download output
│   ├── target_date.txt          # Latest target date written by downloader
│   └── weights/                 # Model checkpoints
├── configs/                     # Conda environment files
├── outputs/                     # Inference results (.nc files)
├── src/
│   ├── common/
│   │   ├── downloader_unified.py  # Unified CDS API ERA5 downloader (NetCDF)
│   │   └── saver.py               # Unified NetCDF saver
│   ├── stormer/
│   ├── aifs/
│   ├── neuralgcm/
│   ├── pangu/
│   └── aurora/
└── README.md
```

## 🛠️ Installation

We recommend using **Conda** to manage separate environments for different models due to dependency conflicts (e.g., specific PyTorch/CUDA versions).

### Environment
```bash
conda env create -f configs/nwp_unified.yaml
conda activate nwp_unified
```

### Install `cdsapi`
```bash
pip install cdsapi
```

## 🔐 Configure CDS API (Copernicus)

`src/common/downloader_unified.py` reads CDS credentials in this order:

1. `./.cdsapirc` (repo root)
2. environment variable `CDSAPI_KEY`

### Option A (recommended): `.cdsapirc` in repo root

Create `/workspace/NWP-Benchmark/.cdsapirc`:

```yaml
url: https://cds.climate.copernicus.eu/api
key: <uid>:<api-key>
```

> Replace `<uid>:<api-key>` with your Copernicus CDS API token.

### Option B: environment variable

```bash
export CDSAPI_KEY="<uid>:<api-key>"
```

## ⬇️ Unified ERA5 Download (NetCDF)

Use the unified downloader:

```bash
python src/common/downloader_unified.py --date 2023010112
```

### Output location

Downloaded files are saved to:

- `assets/data/raw/static.nc`
- `assets/data/raw/surface_YYYYMMDDHH.nc`
- `assets/data/raw/upper_YYYYMMDDHH.nc`

The script also writes target date to:

- `assets/target_date.txt`

> Notes
> - Download format is NetCDF.
> - Existing files are skipped automatically.

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
Use the unified downloader first (see section above), then run model-specific prepare scripts when needed:

```bash
# Unified ERA5 download (NetCDF)
python src/common/downloader_unified.py --date 2023010112

# Model-specific wrappers (as required)
python src/stormer/prepare.py --date 2023010112
python src/aifs/prepare.py --date 2023010112
python src/pangu/prepare.py --date 2023010112
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

- **File naming convention**: `YYYY-MMDD-LL.nc` (e.g., `2023-0101-06.nc` for a 6h lead time).
- **Variables**: Output variables include both surface and pressure-level data, complying with CF conventions:
  - **Surface**: `t2m`, `msl`, `u10`, `v10`, `tp`, etc.
  - **Upper Air**: `z`, `t`, `u`, `v`, `q` (at levels 50, 100, ..., 1000 hPa).

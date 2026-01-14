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

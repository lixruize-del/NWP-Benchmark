import os
import logging
import argparse
import xarray as xr
import pandas as pd
import pickle
import gcsfs
from pathlib import Path
import neuralgcm

# ==============================================================================
# Configuration & Setup
# ==============================================================================
# Configure Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("NeuralGCM.DownloadARCO")

# Path Configuration
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
# 保存路径保持不变，以便 prepare.py 能找到
SAVE_DIR = BASE_DIR / "assets" / "data" / "era5_neuralgcm"
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "neuralgcm"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

# ARCO-ERA5 Bucket URL (Google Public Data)
ARCO_ERA5_PATH = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

# Ensure output directory exists
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Helper Functions
# ==============================================================================
def load_model_for_metadata():
    """Load the model checkpoint to get the required variable names."""
    ckpt_name = "models_v1_deterministic_1_4_deg.pkl"
    ckpt_path = WEIGHTS_DIR / ckpt_name
    
    if not ckpt_path.exists():
        logger.error(f"Weights not found at {ckpt_path}. Please run src/neuralgcm/download_weights.py first.")
        raise FileNotFoundError("Model weights missing")
        
    logger.info(f"Loading model metadata from {ckpt_name}...")
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    
    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
    return model

def download_from_arco(target_date_str):
    """
    Download data from Google ARCO-ERA5 Zarr and save as NetCDF.
    """
    logger.info(f"Connecting to Google Cloud Storage: {ARCO_ERA5_PATH}")
    
    # 1. Connect to Zarr
    try:
        # storage_options={'token': 'anon'} means anonymous access (public bucket)
        ds = xr.open_zarr(
            ARCO_ERA5_PATH, 
            chunks=None, 
            storage_options=dict(token='anon')
        )
    except Exception as e:
        logger.error(f"Failed to connect to GCS. Network issue? Error: {e}")
        return

    # 2. Determine Time Range (T0 and T-6h)
    dt = pd.to_datetime(target_date_str, format="%Y%m%d%H")
    times = [dt - pd.Timedelta(hours=6), dt]
    
    logger.info(f"Selecting time slices: {times}")

    # 3. Get Required Variables from Model
    try:
        model = load_model_for_metadata()
        req_vars = model.input_variables + model.forcing_variables
        # Deduplicate
        req_vars = list(set(req_vars))
        logger.info(f"Required variables: {req_vars}")
    except Exception as e:
        logger.error(f"Failed to load model metadata: {e}")
        return

    # 4. Slice Data (Lazy Loading)
    try:
        # Slice variables
        # Note: ARCO variable names usually match NeuralGCM expectations, but we need to be careful.
        # ARCO uses 'level' for pressure levels, NeuralGCM might expect specific names.
        # Luckily NeuralGCM is built by Google on top of ARCO, so names should align.
        
        subset = ds[req_vars].sel(time=times)
        
        # 5. Separate Surface and Upper Air for compatibility with prepare.py
        # We need to save them as surface_XXX.nc and upper_XXX.nc
        # Check dimensions to separate them
        
        # Identify variables with 'level' dimension (Upper Air) vs only lat/lon (Surface)
        upper_vars = [v for v in req_vars if 'level' in subset[v].dims]
        surf_vars = [v for v in req_vars if 'level' not in subset[v].dims]
        
        ds_upper = subset[upper_vars]
        ds_surf = subset[surf_vars]
        
        # 6. Compute (Download) and Save
        # This is where the actual download happens
        
        # Save Surface
        surf_out = SAVE_DIR / f"surface_{target_date_str}.nc"
        logger.info(f"Downloading & Saving Surface data to {surf_out}...")
        ds_surf.to_netcdf(surf_out)
        
        # Save Upper
        upper_out = SAVE_DIR / f"upper_{target_date_str}.nc"
        logger.info(f"Downloading & Saving Upper Air data to {upper_out}...")
        ds_upper.to_netcdf(upper_out)
        
        # Save Static (optional, ARCO might have geopotential_at_surface in prognostic)
        # We can extract static from the first time step if needed, or rely on surface file
        
        logger.info("✅ Download from ARCO complete.")
        
    except KeyError as e:
        logger.error(f"Variable missing in ARCO dataset: {e}")
    except Exception as e:
        logger.error(f"Data processing failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="2020082212", help="Target date YYYYMMDDHH")
    args = parser.parse_args()

    # Update date file
    with open(DATE_FILE, "w") as f:
        f.write(args.date)

    download_from_arco(args.date)

if __name__ == "__main__":
    main()
import os
import sys
import logging
import argparse
import pickle
import numpy as np
import xarray as xr
from pathlib import Path

# NeuralGCM specific imports
import neuralgcm
from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("NeuralGCM.Prepare")

# Path configuration
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
RAW_DATA_DIR = BASE_DIR / "assets" / "data" / "era5_neuralgcm"
PROCESSED_DIR = BASE_DIR / "assets" / "data" / "processed_neuralgcm"
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "neuralgcm"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def get_target_date():
    if DATE_FILE.exists():
        with open(DATE_FILE) as f:
            return f.read().strip()
    return "2023010112"

def load_model_structure():
    """
    Load the model checkpoint to retrieve the target Gaussian grid definition.
    """
    ckpt_name = "models_v1_deterministic_1_4_deg.pkl"
    ckpt_path = WEIGHTS_DIR / ckpt_name
    
    if not ckpt_path.exists():
        logger.error(f"Weights not found at {ckpt_path}. Run download_weights.py first.")
        sys.exit(1)
        
    logger.info(f"Loading checkpoint structure from {ckpt_name}...")
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    
    # Initialize model wrapper to get coordinates
    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
    return model

def prepare_data(target_date):
    logger.info(f"Starting data preparation for {target_date}...")
    
    # 1. Check Input Files
    surf_path = RAW_DATA_DIR / f"surface_{target_date}.nc"
    upper_path = RAW_DATA_DIR / f"upper_{target_date}.nc"
    
    if not surf_path.exists() or not upper_path.exists():
        logger.error(f"Input files missing in {RAW_DATA_DIR}. Run download_data.py first.")
        sys.exit(1)

    # 2. Load and Merge ERA5 Data
    logger.info("Loading ERA5 NetCDF files...")
    ds_surf = xr.open_dataset(surf_path)
    ds_upper = xr.open_dataset(upper_path)
    
    # Merge surface and upper air data
    ds_full = xr.merge([ds_surf, ds_upper])

    # 3. Define Source Grid (ERA5 Lat/Lon)
    logger.info("Defining source grid from ERA5 data...")
    # Ensure coordinates are sorted for the regridder
    ds_full = ds_full.sortby("latitude", ascending=True)
    ds_full = ds_full.sortby("longitude", ascending=True)
    
    source_grid = spherical_harmonic.Grid(
        latitude_nodes=ds_full.sizes['latitude'],
        longitude_nodes=ds_full.sizes['longitude'],
        latitude_spacing=xarray_utils.infer_latitude_spacing(ds_full.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(ds_full.longitude),
    )

    # 4. Define Target Grid (NeuralGCM Gaussian)
    model = load_model_structure()
    target_coords = model.data_coords.horizontal

    # 5. Perform Regridding
    logger.info("Initializing Conservative Regridder...")
    regridder = horizontal_interpolation.ConservativeRegridder(
        source_grid, target_coords, skipna=True
    )
    
    logger.info("Regridding data to Gaussian grid...")
    # Regrid the dataset
    ds_regridded = xarray_utils.regrid(ds_full, regridder)
    
    # Fill NaNs (common near coastlines)
    ds_regridded = xarray_utils.fill_nan_with_nearest(ds_regridded)
    
    # 6. Save Output
    output_filename = f"input_{target_date}_gaussian.nc"
    save_path = PROCESSED_DIR / output_filename
    
    logger.info(f"Saving processed data to {save_path}...")
    ds_regridded.to_netcdf(save_path)
    logger.info("Data preparation completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()
    
    date = args.date if args.date else get_target_date()
    prepare_data(date)
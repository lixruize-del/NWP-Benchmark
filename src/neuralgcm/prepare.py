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
RAW_DATA_DIR = BASE_DIR / "assets" / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "assets" / "data" / "processed_neuralgcm"
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "neuralgcm"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

os.makedirs(PROCESSED_DIR, exist_ok=True)

VAR_NAME_MAP = {
    'z': 'geopotential',
    't': 'temperature',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    'q': 'specific_humidity',
    'ciwc': 'specific_cloud_ice_water_content',
    'clwc': 'specific_cloud_liquid_water_content',
    'sst': 'sea_surface_temperature',
    'siconc': 'sea_ice_cover',
}

def standardize_latlon_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize latitude/longitude coordinates for downstream regridding.

    - Ensures latitude is monotonically increasing (-90 -> 90).
    - Ensures longitude is within [0, 360) and monotonically increasing.
    """
    if "longitude" in ds.coords:
        lon = ds["longitude"]
        if float(lon.min()) < 0.0:
            ds = ds.assign_coords(longitude=((lon + 360.0) % 360.0))
        ds = ds.sortby("longitude")

    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")

    return ds

def get_target_date():
    if DATE_FILE.exists():
        with open(DATE_FILE) as f:
            return f.read().strip()
    return "2023010112"

def load_model_structure():
    """
    Load the model checkpoint to retrieve the target Gaussian grid definition.
    """
    ckpt_name = "models_v1_deterministic_0_7_deg.pkl"
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

    rename_dict = {}
    if 'valid_time' in ds_surf.dims:
        rename_dict['valid_time'] = 'time'
    if rename_dict:
        ds_surf = ds_surf.rename(rename_dict)
    if 'valid_time' in ds_upper.dims:
        ds_upper = ds_upper.rename({'valid_time': 'time'})
    if 'pressure_level' in ds_upper.dims:
        ds_upper = ds_upper.rename({'pressure_level': 'level'})
    
    # Merge surface and upper air data
    ds_full = xr.merge([ds_surf, ds_upper])

    if ds_full['level'].values[0] > ds_full['level'].values[-1]:
        ds_full = ds_full.isel(level=slice(None, None, -1))

    # Critical: conservative regridding assumes monotonic coordinates.
    ds_full = standardize_latlon_coords(ds_full)

    rename_vars = {k: v for k, v in VAR_NAME_MAP.items() if k in ds_full}
    if rename_vars:
        logger.info(f"Renaming variables: {rename_vars}")
        ds_full = ds_full.rename(rename_vars)

    model = load_model_structure()
    required_vars = list(set(model.input_variables + model.forcing_variables))
    logger.info(f"Required variables from model: {required_vars}")

    available_vars = [v for v in required_vars if v in ds_full]
    missing_vars = set(required_vars) - set(available_vars)
    if missing_vars:
        logger.error(f"Missing required variables in input files: {missing_vars}")
        sys.exit(1)

    ds_full = ds_full[available_vars]

    # Ensure canonical dimension order for Dinosaur/NeuralGCM utilities.
    # Many regridding/interpolation paths assume (..., longitude, latitude).
    if "longitude" in ds_full.dims and "latitude" in ds_full.dims:
        ds_full = ds_full.transpose(..., "longitude", "latitude")

    # 3. Define Source Grid (ERA5 Lat/Lon)
    logger.info("Defining source grid from ERA5 data...")
    # Ensure coordinates are sorted for the regridder
    # ds_full = ds_full.sortby("latitude", ascending=True)
    # ds_full = ds_full.sortby("longitude", ascending=True)
    ds_full = standardize_latlon_coords(ds_full)
    
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
    
    """
    # 6. Save Output
    output_filename = f"input_{target_date}_gaussian.nc"
    save_path = PROCESSED_DIR / output_filename
    
    logger.info(f"Saving processed data to {save_path}...")
    ds_regridded.to_netcdf(save_path)
    logger.info("Data preparation completed successfully.")
    """

    tmp_output = Path("/tmp") / f"input_{target_date}_gaussian.nc"
    ds_regridded.to_netcdf(tmp_output)
    logger.info(f"✅ Saved to temporary file: {tmp_output}")

    output_path = PROCESSED_DIR / f"input_{target_date}_gaussian.nc"
    import shutil
    shutil.move(str(tmp_output), str(output_path))
    logger.info(f"✅ Moved to final destination: {output_path}")

    with open(PROCESSED_DIR / "latest_date.txt", "w") as f:
        f.write(target_date)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()
    
    date = args.date if args.date else get_target_date()
    prepare_data(date)
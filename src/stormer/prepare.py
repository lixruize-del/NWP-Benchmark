import os
import sys
import logging
import argparse
import numpy as np
import xarray as xr
import cfgrib
from pathlib import Path

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Stormer.Prepare")

# Configure Paths
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
RAW_DATA_DIR = BASE_DIR / "assets" / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "assets" / "data" / "processed_stormer"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def get_target_date():
    if DATE_FILE.exists():
        with open(DATE_FILE) as f:
            return f.read().strip()
    return "2025120412"

def read_variable_safe(file_path, possible_names, level=None, target_grid=None):
    """
    Robustly read a single variable from a GRIB file using open_datasets to avoid index conflicts.
    """
    lat_target, lon_target = target_grid
    
    # Suppress cfgrib internal warnings
    logging.getLogger('cfgrib').setLevel(logging.ERROR)

    try:
        datasets = cfgrib.open_datasets(file_path)
    except Exception as e:
        logger.warning(f"Failed to open GRIB file {file_path.name}: {e}")
        return np.zeros((128, 256), dtype=np.float32)

    found_da = None
    for ds in datasets:
        for name in possible_names:
            if name in ds:
                if level is not None:
                    if 'isobaricInhPa' in ds.coords:
                        try:
                            found_da = ds[name].sel(isobaricInhPa=level)
                            break
                        except: continue 
                else:
                    found_da = ds[name]
                    break
        if found_da is not None: break
            
    if found_da is not None:
        # Interpolate to Stormer grid (128x256)
        val = found_da.interp(latitude=lat_target, longitude=lon_target, kwargs={"fill_value": "extrapolate"}).values
        return val.astype(np.float32)
    else:
        return np.zeros((128, 256), dtype=np.float32)

def prepare_data(target_date):
    logger.info(f"Starting data preparation for date: {target_date}")
    
    file_sfc = RAW_DATA_DIR / f"raw_{target_date}_sfc.grib"
    file_pl = RAW_DATA_DIR / f"raw_{target_date}_pl.grib"
    
    if not file_sfc.exists() or not file_pl.exists():
        logger.error(f"Raw data missing. Please check: {file_sfc}")
        sys.exit(1)

    # Define Target Grid for Stormer (1.40625 deg -> 128x256)
    lat_target = np.linspace(90, -90, 128)
    lon_target = np.linspace(0, 360, 256, endpoint=False)
    grid = (lat_target, lon_target)

    arrays = []
    
    logger.info("Processing Surface variables...")
    arrays.append(read_variable_safe(file_sfc, ['2t', 't2m'], target_grid=grid))
    arrays.append(read_variable_safe(file_sfc, ['10u', 'u10'], target_grid=grid))
    arrays.append(read_variable_safe(file_sfc, ['10v', 'v10'], target_grid=grid))
    arrays.append(read_variable_safe(file_sfc, ['msl', 'prmsl'], target_grid=grid))
    
    logger.info("Processing Upper Air variables...")
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    for l in levels: arrays.append(read_variable_safe(file_pl, ['z', 'gh'], level=l, target_grid=grid))
    for l in levels: arrays.append(read_variable_safe(file_pl, ['u'], level=l, target_grid=grid))
    for l in levels: arrays.append(read_variable_safe(file_pl, ['v'], level=l, target_grid=grid))
    for l in levels: arrays.append(read_variable_safe(file_pl, ['t'], level=l, target_grid=grid))
    for l in levels: arrays.append(read_variable_safe(file_pl, ['q'], level=l, target_grid=grid))
    
    # Stack and Save
    input_tensor = np.stack(arrays).astype(np.float32)
    output_path = PROCESSED_DIR / f"input_{target_date}.npy"
    
    np.save(output_path, input_tensor)
    logger.info(f"Data preparation complete. Output saved to: {output_path}")
    logger.info(f"Tensor shape: {input_tensor.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()
    
    date = args.date if args.date else get_target_date()
    prepare_data(date)
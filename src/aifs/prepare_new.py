import os
import sys
import logging
import argparse
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# --- Dependency Check ---
try:
    import earthkit.regrid as ekr
except ImportError:
    print("Error: earthkit-regrid is missing. Please install it: pip install earthkit-regrid")
    sys.exit(1)

try:
    import netCDF4
except ImportError:
    print("Error: netCDF4 library is missing. Please install it: pip install netCDF4")
    sys.exit(1)

# ==============================================================================
# Configuration & Setup
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("AIFS.Prepare")

# Paths
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent

# Input: Use the standardized ERA5 NetCDF folder
RAW_DATA_DIR = BASE_DIR / "assets" / "data" / "raw"
# Output: AIFS specific processed data
OUTPUT_DIR = BASE_DIR / "assets" / "data" / "processed_aifs"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# AIFS specific target grid (Reduced Gaussian Grid N320)
TARGET_GRID = {"grid": "N320"}
# Source grid definition (ERA5 0.25 degree)
SOURCE_GRID = {"grid": [0.25, 0.25]} 

# ==============================================================================
# Variable Mappings
# ==============================================================================
# Map CDS NetCDF variable names to AIFS internal names
SURFACE_MAPPING = {
    # Dynamic
    "2m_temperature": "2t", "t2m": "2t",
    "10m_u_component_of_wind": "10u", "u10": "10u",
    "10m_v_component_of_wind": "10v", "v10": "10v",
    "mean_sea_level_pressure": "msl", "msl": "msl",
    "surface_pressure": "sp", "sp": "sp",
    "total_precipitation": "tp", "tp": "tp",
    "total_column_water_vapour": "tcw", "tcwv": "tcw",
    "skin_temperature": "skt", "skt": "skt",
    
    # Static / Forcing
    "land_sea_mask": "lsm", "lsm": "lsm",
    "geopotential": "z", "z": "z", # Orography
    "standard_deviation_of_orography": "sdor", "sdor": "sdor",
    "slope_of_sub_gridscale_orography": "slor", "slor": "slor",
    "angle_of_sub_gridscale_orography": "anor", "anor": "anor",
    "standard_deviation_of_filtered_subgrid_orography": "isor", "isor": "isor"
}

PRESSURE_VARS_MAP = {
    "geopotential": "z", "z": "z",
    "temperature": "t", "t": "t",
    "u_component_of_wind": "u", "u": "u",
    "v_component_of_wind": "v", "v": "v",
    "specific_humidity": "q", "q": "q",
    "vertical_velocity": "w", "w": "w"
}

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# ==============================================================================
# Helper Functions
# ==============================================================================
def get_target_date():
    if DATE_FILE.exists():
        with open(DATE_FILE) as f:
            return f.read().strip()
    return "2023010112"

def regrid_field(data_array):
    """Interpolate to N320. Input must be 2D (Lat, Lon)."""
    # Ensure input is 2D
    if data_array.ndim > 2:
        data_array = data_array.squeeze()
    
    if data_array.ndim != 2:
        # If squeeze didn't work (e.g. time dim > 1), we have a problem
        raise ValueError(f"Input for regridding must be 2D, got {data_array.shape}")
        
    return ekr.interpolate(data_array, SOURCE_GRID, TARGET_GRID)

def process_data(target_date):
    logger.info(f"🚀 Preparing AIFS data for: {target_date}")
    
    # 1. Load NetCDF Files
    surf_path = RAW_DATA_DIR / f"surface_{target_date}.nc"
    upper_path = RAW_DATA_DIR / f"upper_{target_date}.nc"
    static_path = RAW_DATA_DIR / "static.nc"

    if not surf_path.exists() or not upper_path.exists():
        logger.error(f"Missing input files in {RAW_DATA_DIR}")
        sys.exit(1)

    try:
        ds_surf = xr.open_dataset(surf_path, engine='netcdf4')
        ds_upper = xr.open_dataset(upper_path, engine='netcdf4')
        ds_static = xr.open_dataset(static_path, engine='netcdf4') if static_path.exists() else None
    except Exception as e:
        logger.error(f"NetCDF Open Error: {e}")
        sys.exit(1)

    # 2. Time Setup
    dt = pd.to_datetime(target_date, format="%Y%m%d%H")
    times = [dt - pd.Timedelta(hours=6), dt]
    
    fields = {}
    processed_keys = set()

    # Determine time dimension name
    time_key_surf = 'valid_time' if 'valid_time' in ds_surf.dims else 'time'
    time_key_upper = 'valid_time' if 'valid_time' in ds_upper.dims else 'time'

    # --- 1. Merge Static into Surface ---
    if ds_static:
        ds_surf = xr.merge([ds_surf, ds_static], compat='override')

    logger.info("Processing Surface & Static variables...")
    
    for cds_name, aifs_name in SURFACE_MAPPING.items():
        if aifs_name in processed_keys: continue
        
        if cds_name in ds_surf:
            try:
                da = ds_surf[cds_name]
                
                # Check if variable is static (no time dim or time dim size 1)
                is_static = time_key_surf not in da.dims or \
                           (da.sizes[time_key_surf] == 1 and aifs_name in ['z', 'lsm', 'sdor', 'slor', 'anor', 'isor'])
                
                if is_static:
                    val = da.values
                    # Handle potential single-item dims
                    if val.ndim > 2: val = val.squeeze()
                    
                    val_regridded = regrid_field(val)
                    # Broadcast to (2, N_grid) for T-6 and T0
                    fields[aifs_name] = np.stack([val_regridded, val_regridded])
                    logger.info(f"  + Static: {aifs_name}")
                
                else:
                    # Dynamic variable
                    da_sel = da.sel({time_key_surf: times})
                    
                    regridded_steps = []
                    for t_step in da_sel:
                        val = np.nan_to_num(t_step.values, nan=0.0)
                        val_regridded = regrid_field(val)
                        regridded_steps.append(val_regridded)
                    
                    fields[aifs_name] = np.stack(regridded_steps)
                    # logger.info(f"  + Dynamic: {aifs_name}")
                
                processed_keys.add(aifs_name)

            except Exception as e:
                logger.warning(f"  Failed {cds_name}->{aifs_name}: {e}")

    # --- 2. Upper Air ---
    logger.info("Processing Upper Air variables...")
    processed_pl = set()

    for cds_name, aifs_base in PRESSURE_VARS_MAP.items():
        if aifs_base in processed_pl: continue

        if cds_name in ds_upper:
            processed_pl.add(aifs_base)
            da_full = ds_upper[cds_name].sel({time_key_upper: times})
            
            for level in LEVELS:
                try:
                    level_key = 'level' if 'level' in ds_upper.dims else 'pressure_level'
                    da_level = da_full.sel({level_key: level})
                    
                    regridded_steps = []
                    for t_step in da_level:
                        val = t_step.values
                        val_regridded = regrid_field(val)
                        regridded_steps.append(val_regridded)
                    
                    key_name = f"{aifs_base}_{level}"
                    fields[key_name] = np.stack(regridded_steps)
                except KeyError:
                    pass

    # Save
    output_filename = f"init_{dt.strftime('%Y%m%d_%H')}.npz"
    output_path = OUTPUT_DIR / output_filename
    
    logger.info(f"Saving {len(fields)} fields to: {output_path}")
    np.savez(output_path, date=str(dt), **fields)
    logger.info("✅ Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()
    
    target = args.date if args.date else get_target_date()
    process_data(target)
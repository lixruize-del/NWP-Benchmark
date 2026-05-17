import os
import sys
import logging
import argparse
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# ==============================================================================
# Configuration & Setup
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("FuXi.Prepare")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "assets" / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "assets" / "data" / "processed_fuxi"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# FuXi Required Constants
# ==============================================================================
PL_NAMES = ['z', 't', 'u', 'v', 'r']
SFC_NAMES = ['t2m', 'u10', 'v10', 'msl', 'tp']
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

VAR_MAP = {
    # surface
    't2m': 't2m',
    'u10': 'u10',
    'v10': 'v10',
    'msl': 'msl',
    'tp': 'tp',
    # pressure levels
    'z': 'z',
    't': 't',
    'u': 'u',
    'v': 'v',
    'r': 'r',
}

# ==============================================================================
# Helper Functions
# ==============================================================================
def get_target_date():
    if DATE_FILE.exists():
        return DATE_FILE.read_text().strip()
    logger.warning("DATE_FILE not found, using default 2023010112")
    return "2023010112"

def ensure_lat_order(ds):
    """Ensure latitude is decreasing (north to south)."""
    if ds.lat[0] < ds.lat[-1]:
        logger.debug("Reversing latitude order to north-to-south.")
        ds = ds.isel(lat=slice(None, None, -1))
    return ds

def ensure_lon_0_360(ds):
    """Ensure longitude is in [0, 360) and monotonic."""
    lon = ds.lon.values
    if (lon < 0).any():
        logger.debug("Converting longitude from [-180,180) to [0,360).")
        lon_positive = lon % 360
        sort_idx = np.argsort(lon_positive)
        ds = ds.isel(lon=sort_idx)
        ds = ds.assign_coords(lon=lon_positive[sort_idx])
    return ds

def load_and_prepare(target_date):
    dt = pd.to_datetime(target_date, format="%Y%m%d%H")
    times = [dt - pd.Timedelta(hours=6), dt]

    surf_path = RAW_DATA_DIR / f"surface_{target_date}.nc"
    upper_path = RAW_DATA_DIR / f"upper_{target_date}.nc"

    for p in [surf_path, upper_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    logger.info(f"Loading surface data from {surf_path}")
    ds_surf = xr.open_dataset(surf_path)
    logger.info(f"Loading upper air data from {upper_path}")
    ds_upper = xr.open_dataset(upper_path)

    # --- Rename dimensions/coordinates to standard names ---
    # Surface dataset
    if 'valid_time' in ds_surf.dims:
        ds_surf = ds_surf.rename({'valid_time': 'time'})
    if 'latitude' in ds_surf.coords:
        ds_surf = ds_surf.rename({'latitude': 'lat'})
    if 'longitude' in ds_surf.coords:
        ds_surf = ds_surf.rename({'longitude': 'lon'})

    # Upper dataset
    if 'valid_time' in ds_upper.dims:
        ds_upper = ds_upper.rename({'valid_time': 'time'})
    if 'latitude' in ds_upper.coords:
        ds_upper = ds_upper.rename({'latitude': 'lat'})
    if 'longitude' in ds_upper.coords:
        ds_upper = ds_upper.rename({'longitude': 'lon'})

    # --- Ensure coordinate order ---
    ds_surf = ensure_lon_0_360(ds_surf)
    ds_upper = ensure_lon_0_360(ds_upper)
    ds_surf = ensure_lat_order(ds_surf)
    ds_upper = ensure_lat_order(ds_upper)

    # --- Select the two required time steps ---
    ds_surf = ds_surf.sel(time=times)
    ds_upper = ds_upper.sel(time=times)

    # --- Collect variables ---
    data_arrays = []

    # Pressure level variables
    for pl_name in PL_NAMES:
        netcdf_var = None
        for nc_name, internal_name in VAR_MAP.items():
            if internal_name == pl_name and nc_name in ds_upper:
                netcdf_var = nc_name
                break
        if netcdf_var is None:
            raise ValueError(f"Pressure level variable '{pl_name}' not found in upper dataset.")

        da = ds_upper[netcdf_var]
        if 'pressure_level' not in da.dims:
            raise ValueError(f"Variable '{netcdf_var}' has no pressure_level dimension.")
        da = da.sel(pressure_level=LEVELS)
        da = da.transpose('time', 'pressure_level', 'lat', 'lon')
        da = da.rename({'pressure_level': 'level'})   # level dimension now has coordinates (the pressure values)
        da = da.astype(np.float32)
        if da.isnull().any():
            logger.warning(f"NaNs found in {pl_name}, filling with 0.")
            da = da.fillna(0)
        data_arrays.append(da)
        logger.info(f"  + Added {pl_name.upper()} with shape {da.shape}")

    # Surface variables (including TP)
    for sfc_name in SFC_NAMES:
        if sfc_name == 'tp':
            if 'tp' not in ds_surf:
                raise ValueError("'tp' variable not found in surface dataset.")
            da = ds_surf['tp']  
            da = da.expand_dims(dim={'level': 1}, axis=1)
            da = da.assign_coords(level=[0]) 
        else:
            netcdf_var = None
            for nc_name, internal_name in VAR_MAP.items():
                if internal_name == sfc_name and nc_name in ds_surf:
                    netcdf_var = nc_name
                    break
            if netcdf_var is None:
                raise ValueError(f"Surface variable '{sfc_name}' not found in surface dataset.")
            da = ds_surf[netcdf_var]
            # Add dummy level dimension and a placeholder coordinate
            da = da.expand_dims(dim={'level': 1}, axis=1)
            da = da.assign_coords(level=[0])   # placeholder coordinate

        da = da.transpose('time', 'level', 'lat', 'lon').astype(np.float32)
        if da.isnull().any():
            logger.warning(f"NaNs found in {sfc_name}, filling with 0.")
            da = da.fillna(0)
        data_arrays.append(da)
        logger.info(f"  + Added {sfc_name.upper()} with shape {da.shape}")

    return data_arrays, times

def make_fuxi_input(target_date):
    data_arrays, times = load_and_prepare(target_date)

    ds_concat = xr.concat(data_arrays, dim='level')

    # Create channel names as required by FuXi
    channel_names = [f'{n.upper()}{l}' for n in PL_NAMES for l in LEVELS] + [n.upper() for n in SFC_NAMES]
    ds_concat = ds_concat.assign_coords(level=channel_names)

    # Final checks
    assert ds_concat.lat[0] > ds_concat.lat[-1], "Latitude must be decreasing (north to south)."
    assert ds_concat.lon[0] >= 0 and ds_concat.lon[-1] <= 360, "Longitude must be in [0,360)."
    assert ds_concat.dims == ('time', 'level', 'lat', 'lon'), f"Unexpected dimensions: {ds_concat.dims}"
    assert ds_concat.shape == (2, 70, 721, 1440), f"Shape mismatch: {ds_concat.shape}"

    logger.info(f"Final dataset shape: {ds_concat.shape}")
    logger.info(f"Levels: {ds_concat.level.values.tolist()}")
    return ds_concat

def main(target_date):
    logger.info(f"🚀 Preparing FuXi input for date: {target_date}")
    ds = make_fuxi_input(target_date)

    """
    output_filename = f"input_{target_date}.nc"
    output_path = OUTPUT_DIR / output_filename
    ds.to_netcdf(output_path)
    logger.info(f"✅ Saved FuXi input to {output_path}")
    """

    tmp_output = Path("/tmp") / f"input_{target_date}.nc"
    ds.to_netcdf(tmp_output)
    logger.info(f"✅ Saved to temporary file: {tmp_output}")

    output_path = OUTPUT_DIR / f"input_{target_date}.nc"
    import shutil
    shutil.move(str(tmp_output), str(output_path))
    logger.info(f"✅ Moved to final destination: {output_path}")

    with open(OUTPUT_DIR / "latest_date.txt", "w") as f:
        f.write(target_date) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="Target date in format YYYYMMDDHH. If not provided, reads from assets/target_date.txt")
    args = parser.parse_args()

    target = args.date if args.date else get_target_date()
    main(target)
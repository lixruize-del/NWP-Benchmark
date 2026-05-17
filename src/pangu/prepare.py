import os
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
logger = logging.getLogger("Pangu.Prepare")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "assets" / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "assets" / "data" / "processed_pangu"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Pangu Required Constants
# ==============================================================================
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]  # high to low pressure
UPPER_VARS = ['z', 'q', 't', 'u', 'v']   # upper-air variable order
SURFACE_VARS = ['msl', 'u10', 'v10', 't2m']  # surface variable order

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
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    if ds[lat_name][0] < ds[lat_name][-1]:
        logger.debug("Reversing latitude order to north-to-south.")
        ds = ds.isel({lat_name: slice(None, None, -1)})
    return ds

def ensure_lon_0_360(ds):
    """Ensure longitude is in [0, 360) and monotonic."""
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
    lon = ds[lon_name].values
    if (lon < 0).any():
        logger.debug("Converting longitude from [-180,180) to [0,360).")
        lon_positive = lon % 360
        sort_idx = np.argsort(lon_positive)
        ds = ds.isel({lon_name: sort_idx})
        ds = ds.assign_coords({lon_name: lon_positive[sort_idx]})
    else:
        ds = ds.sortby(lon_name)
    return ds

def prepare_pangu_data(target_date):
    logger.info(f"🚀 Preparing Pangu input for date: {target_date}")

    dt = pd.to_datetime(target_date, format="%Y%m%d%H")

    surf_path = RAW_DATA_DIR / f"surface_{target_date}.nc"
    upper_path = RAW_DATA_DIR / f"upper_{target_date}.nc"

    for p in [surf_path, upper_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    logger.info(f"Loading surface data from {surf_path}")
    ds_surf = xr.open_dataset(surf_path)
    logger.info(f"Loading upper air data from {upper_path}")
    ds_upper = xr.open_dataset(upper_path)

    # --- Rename coordinates to standard names ---
    # surface
    rename_surf = {}
    if 'valid_time' in ds_surf.dims:
        rename_surf['valid_time'] = 'time'
    if 'latitude' in ds_surf.coords:
        rename_surf['latitude'] = 'lat'
    if 'longitude' in ds_surf.coords:
        rename_surf['longitude'] = 'lon'
    if rename_surf:
        ds_surf = ds_surf.rename(rename_surf)

    # upper air
    rename_upper = {}
    if 'valid_time' in ds_upper.dims:
        rename_upper['valid_time'] = 'time'
    if 'latitude' in ds_upper.coords:
        rename_upper['latitude'] = 'lat'
    if 'longitude' in ds_upper.coords:
        rename_upper['longitude'] = 'lon'
    if 'pressure_level' in ds_upper.dims:
        rename_upper['pressure_level'] = 'level'
    if rename_upper:
        ds_upper = ds_upper.rename(rename_upper)

    # --- Ensure coordinate ordering ---
    ds_surf = ensure_lon_0_360(ds_surf)
    ds_upper = ensure_lon_0_360(ds_upper)
    ds_surf = ensure_lat_order(ds_surf)
    ds_upper = ensure_lat_order(ds_upper)

    # --- Select analysis time (T0)---
    ds_surf = ds_surf.sel(time=dt, method='nearest')
    ds_upper = ds_upper.sel(time=dt, method='nearest')

    # --- Extract and stack surface variables ---
    logger.info("Processing surface variables...")
    surface_arrays = []
    for var in SURFACE_VARS:
        if var not in ds_surf:
            raise ValueError(f"Surface variable '{var}' not found in dataset.")
        da = ds_surf[var].transpose('lat', 'lon').astype(np.float32)
        surface_arrays.append(da.values)
    input_surface = np.stack(surface_arrays, axis=0)  # (4, 721, 1440)

    # --- Extract and stack upper-air variables ---
    logger.info("Processing upper-air variables...")
    upper_arrays = []
    for var in UPPER_VARS:
        if var not in ds_upper:
            raise ValueError(f"Upper variable '{var}' not found in dataset.")
        da = ds_upper[var]
        # Select pressure levels (LEVELS order)
        da = da.sel(level=LEVELS)
        da = da.transpose('level', 'lat', 'lon').astype(np.float32)
        upper_arrays.append(da.values)
    input_upper = np.stack(upper_arrays, axis=0)  # (5, 13, 721, 1440)

    # --- Final checks ---
    # latitude decreasing (N to S)
    lat_vals = ds_surf.lat.values
    assert lat_vals[0] > lat_vals[-1], "Latitude must be decreasing (north to south)."
    # longitude in [0, 360)
    lon_vals = ds_surf.lon.values
    assert lon_vals[0] >= 0 and lon_vals[-1] <= 360, "Longitude must be in [0,360)."

    # --- Save fixed-name .npy inputs---
    upper_out = OUTPUT_DIR / "input_upper.npy"
    surface_out = OUTPUT_DIR / "input_surface.npy"

    np.save(upper_out, input_upper)
    np.save(surface_out, input_surface)

    logger.info(f"✅ Saved upper data to {upper_out}")
    logger.info(f"✅ Saved surface data to {surface_out}")

    # Optional: write latest date stamp
    with open(OUTPUT_DIR / "latest_date.txt", "w") as f:
        f.write(target_date)

def main():
    parser = argparse.ArgumentParser(description="Prepare Pangu input data from ERA5 NetCDF files.")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date in format YYYYMMDDHH. If not provided, reads from assets/target_date.txt")
    args = parser.parse_args()

    target = args.date if args.date else get_target_date()
    prepare_pangu_data(target)

if __name__ == "__main__":
    main()
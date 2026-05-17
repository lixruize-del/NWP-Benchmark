import logging
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

from graphcast import data_utils, solar_radiation

# ==============================================================================
# Logging setup
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GraphCast.Prepare")

# ==============================================================================
# Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "assets" / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "assets" / "data" / "processed_graphcast"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Constants
# ==============================================================================
PRESSURE_LEVELS =[1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
                   100, 125, 150, 175, 200, 225, 250, 300,
                   350, 400, 450, 500, 550, 600, 650, 700,
                   750, 775, 800, 825, 850, 875, 900, 925,
                   950, 975, 1000]   # 37 levels

# Mapping from short names (as downloaded) to long names (as expected by model)
SURFACE_MAP = {
    't2m': '2m_temperature',
    'u10': '10m_u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    'msl': 'mean_sea_level_pressure',
    'tp':  'total_precipitation_6hr',   # processed
}
UPPER_MAP = {
    'z': 'geopotential',
    'q': 'specific_humidity',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    't': 'temperature',
    'w': 'vertical_velocity',
}
STATIC_MAP = {
    'z':   'geopotential_at_surface',
    'lsm': 'land_sea_mask',
}

# ==============================================================================
# Helper functions
# ==============================================================================
def get_target_date():
    if DATE_FILE.exists():
        return DATE_FILE.read_text().strip()
    logger.warning("DATE_FILE not found, using default 2023010112")
    return "2023010112"

def ensure_lat_lon_order(ds):
    """Ensure latitude decreasing (north→south) and longitude in[0,360)."""
    # Latitude
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    if ds[lat_name][0] < ds[lat_name][-1]:
        ds = ds.isel({lat_name: slice(None, None, -1)})

    # Longitude
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
    lon = ds[lon_name].values
    if (lon < 0).any():
        lon_positive = lon % 360
        sort_idx = np.argsort(lon_positive)
        ds = ds.isel({lon_name: sort_idx})
        ds = ds.assign_coords({lon_name: lon_positive[sort_idx]})
    return ds

# ==============================================================================
# Main processing function
# ==============================================================================
def prepare_graphcast_input(target_date):
    dt = pd.to_datetime(target_date, format="%Y%m%d%H")
    times =[dt - pd.Timedelta(hours=6), dt]

    surf_file = RAW_DATA_DIR / f"surface_{target_date}.nc"
    upper_file = RAW_DATA_DIR / f"upper_{target_date}.nc"
    static_file = RAW_DATA_DIR / "static.nc"

    for f in[surf_file, upper_file, static_file]:
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}")

    logger.info(f"Loading surface data: {surf_file}")
    ds_surf = xr.open_dataset(surf_file)
    logger.info(f"Loading upper air data: {upper_file}")
    ds_upper = xr.open_dataset(upper_file)
    logger.info(f"Loading static data: {static_file}")
    ds_static = xr.open_dataset(static_file)

    def drop_junk_coords(ds):
        for coord in ['expver', 'number']:
            if coord in ds.coords:
                ds = ds.drop_vars(coord)
        return ds

    ds_surf = drop_junk_coords(ds_surf)
    ds_upper = drop_junk_coords(ds_upper)
    ds_static = drop_junk_coords(ds_static)

    rename_map = {'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon', 'pressure_level': 'level'}
    ds_surf = ds_surf.rename({k: v for k, v in rename_map.items() if k in ds_surf.dims or k in ds_surf.coords})
    ds_upper = ds_upper.rename({k: v for k, v in rename_map.items() if k in ds_upper.dims or k in ds_upper.coords})
    ds_static = ds_static.rename({k: v for k, v in rename_map.items() if k in ds_static.dims or k in ds_static.coords})

    # --- Ensure correct lat/lon order ---
    ds_surf = ensure_lat_lon_order(ds_surf)
    ds_upper = ensure_lat_lon_order(ds_upper)
    ds_static = ensure_lat_lon_order(ds_static)

    # --- Select the two time steps ---
    ds_surf = ds_surf.sel(time=times)
    ds_upper = ds_upper.sel(time=times)

    # --- Collect all variables (long names) ---
    data_vars = {}

    # Surface variables (5)
    for short, long in SURFACE_MAP.items():
        data_vars[long] = ds_surf[short].transpose('time', 'lat', 'lon').astype(np.float32)

    # Upper-air variables (6 * 37 levels)
    for short, long in UPPER_MAP.items():
        da = ds_upper[short].sel(level=PRESSURE_LEVELS)
        data_vars[long] = da.transpose('time', 'level', 'lat', 'lon').astype(np.float32)

    # Static variables (2)
    for src, tgt in STATIC_MAP.items():
        da = ds_static[src].isel(time=0) if 'time' in ds_static[src].dims else ds_static[src]
        data_vars[tgt] = da.transpose('lat', 'lon').astype(np.float32)

    # --- Forcing variables for the two initial time steps ---
    lat = ds_surf.lat.values
    lon = ds_surf.lon.values
    time_coords = np.array([np.datetime64(t) for t in times])
    seconds = time_coords.astype('datetime64[s]').astype(np.int64)

    # --- Create final Dataset ---
    ds_out = xr.Dataset(data_vars, attrs={'description': 'GraphCast input data'})
    
    # Attach datetime coordinate (required by add_derived_vars)
    time_coords = np.array([np.datetime64(t) for t in times])
    ds_out = ds_out.assign_coords(datetime=('time', time_coords))
    
    # Add static features and forcings via add_derived_vars
    data_utils.add_derived_vars(ds_out)
    
    logger.info(f"Created Dataset with variables: {list(ds_out.data_vars)}")
    logger.info(f"Dimensions: {ds_out.dims}")
    logger.info(f"Latitude range: {ds_out.lat.values[0]:.2f} to {ds_out.lat.values[-1]:.2f}")
    logger.info(f"Longitude range: {ds_out.lon.values[0]:.2f} to {ds_out.lon.values[-1]:.2f}")
 
    """
    out_file = OUTPUT_DIR / f"input_{target_date}.nc"
    ds_out.to_netcdf(out_file)
    logger.info(f"Saved GraphCast input to {out_file}")
    """

    tmp_output = Path("/home") / f"input_{target_date}.nc"
    ds_out.to_netcdf(tmp_output)
    logger.info(f"✅ Saved to temporary file: {tmp_output}")

    output_path = OUTPUT_DIR / f"input_{target_date}.nc"
    import shutil
    shutil.move(str(tmp_output), str(output_path))
    logger.info(f"✅ Moved to final destination: {output_path}")

    with open(OUTPUT_DIR / "latest_date.txt", "w") as f:
        f.write(target_date)


# ==============================================================================
# Command-line entry
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="Target date in YYYYMMDDHH format (default: read from target_date.txt)")
    args = parser.parse_args()

    target = args.date if args.date else get_target_date()
    prepare_graphcast_input(target)





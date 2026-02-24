import os
import logging
import argparse
import cdsapi
import pandas as pd
import xarray as xr
import zipfile
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

# ==============================================================================
# Configuration
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Downloader")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SAVE_DIR = BASE_DIR / "assets" / "data" / "raw" 
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

PRESSURE_LEVELS = [
    '50', '100', '150', '200', '250', '300', '400', 
    '500', '600', '700', '850', '925', '1000'
]

# ==============================================================================
# Robust Download Function (The Solution)
# ==============================================================================
def robust_retrieve(client, dataset, request, target_path):
    """
    Downloads data from CDS. If a ZIP is returned (due to mixed stepTypes),
    it automatically extracts, merges, and saves a single NetCDF file.
    """
    target_path = Path(target_path)
    
    # 1. Download to a temporary filename
    temp_download_path = target_path.with_suffix(".download_temp")
    
    try:
        client.retrieve(dataset, request, str(temp_download_path))
    except Exception as e:
        logger.error(f"CDS API Error: {e}")
        if temp_download_path.exists(): temp_download_path.unlink()
        return

    # 2. Check if it is a ZIP file (CDS often zips mixed data)
    if zipfile.is_zipfile(temp_download_path):
        logger.info("Detected ZIP archive (mixed step types). Merging...")
        
        with TemporaryDirectory() as temp_dir:
            # Extract
            with zipfile.ZipFile(temp_download_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find all NC files
            nc_files = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.nc')])
            
            if not nc_files:
                logger.error("No NetCDF files found inside the ZIP archive.")
                return

            try:
                # Open and Merge
                # compat='override' handles slight precision diffs in coordinates
                ds_list = [xr.open_dataset(f) for f in nc_files]
                ds_merged = xr.merge(ds_list, compat='override')
                
                # Save merged file to final target
                logger.info(f"Saving merged NetCDF to: {target_path}")
                ds_merged.to_netcdf(target_path)
                
                # Close handlers
                for ds in ds_list: ds.close()
                
            except Exception as e:
                logger.error(f"Merge failed: {e}")
            
    else:
        # It is a standard NetCDF, just rename it to target
        logger.info("Standard NetCDF detected.")
        shutil.move(str(temp_download_path), str(target_path))

    # Cleanup
    if temp_download_path.exists():
        temp_download_path.unlink()

# ==============================================================================
# Specific Download Tasks
# ==============================================================================
def download_static(client):
    target = SAVE_DIR / "static.nc"
    if target.exists():
        logger.info("Static data already exists.")
        return

    logger.info("Downloading static variables...")
    request = {
        'product_type': 'reanalysis',
        'data_format': 'netcdf',
        'variable': [
            'geopotential', 'land_sea_mask', 'soil_type',
            'angle_of_sub_gridscale_orography', 
            'slope_of_sub_gridscale_orography',
            'standard_deviation_of_filtered_subgrid_orography',
            'standard_deviation_of_orography'
        ],
        'year': '2023', 'month': '01', 'day': '01', 'time': '00:00',
        'grid': [0.25, 0.25],
    }
    robust_retrieve(client, 'reanalysis-era5-single-levels', request, target)

def download_surface(client, date_str):
    target = SAVE_DIR / f"surface_{date_str}.nc"
    if target.exists():
        logger.info(f"Surface data {date_str} already exists.")
        return

    dt = pd.to_datetime(date_str, format="%Y%m%d%H")
    times = [dt, dt - pd.Timedelta(hours=6)]
    
    # Generate unique lists for request
    years = sorted(list(set([t.strftime("%Y") for t in times])))
    months = sorted(list(set([t.strftime("%m") for t in times])))
    days = sorted(list(set([t.strftime("%d") for t in times])))
    hours = sorted(list(set([t.strftime("%H:%M") for t in times])))

    logger.info(f"Downloading surface variables for {date_str}...")
    
    # MIXED VARIABLES (Instant + Accum) -> This usually triggers ZIP
    request = {
        'product_type': 'reanalysis',
        'data_format': 'netcdf',
        'variable': [
            '2m_temperature', 'mean_sea_level_pressure', 
            '10m_u_component_of_wind', '10m_v_component_of_wind',
            'surface_pressure',
            '2m_dewpoint_temperature', 'skin_temperature',
            # Accumulated variables:
            'total_precipitation', 'toa_incident_solar_radiation',
            # Others
            'total_cloud_cover', 'total_column_water_vapour',
            'sea_surface_temperature', 'sea_ice_cover'
        ],
        'year': years, 'month': months, 'day': days, 'time': hours,
        'grid': [0.25, 0.25],
    }
    robust_retrieve(client, 'reanalysis-era5-single-levels', request, target)

def download_upper(client, date_str):
    target = SAVE_DIR / f"upper_{date_str}.nc"
    if target.exists():
        logger.info(f"Upper data {date_str} already exists.")
        return

    dt = pd.to_datetime(date_str, format="%Y%m%d%H")
    times = [dt, dt - pd.Timedelta(hours=6)]
    
    years = sorted(list(set([t.strftime("%Y") for t in times])))
    months = sorted(list(set([t.strftime("%m") for t in times])))
    days = sorted(list(set([t.strftime("%d") for t in times])))
    hours = sorted(list(set([t.strftime("%H:%M") for t in times])))

    logger.info(f"Downloading upper air variables for {date_str}...")
    request = {
        'product_type': 'reanalysis',
        'data_format': 'netcdf',
        'variable': [
            'geopotential', 'specific_humidity', 'temperature',
            'u_component_of_wind', 'v_component_of_wind',
            'vertical_velocity', 'relative_humidity'
        ],
        'pressure_level': PRESSURE_LEVELS,
        'year': years, 'month': months, 'day': days, 'time': hours,
        'grid': [0.25, 0.25],
    }
    robust_retrieve(client, 'reanalysis-era5-pressure-levels', request, target)

def load_cds_config():
    config_path = BASE_DIR / ".cdsapirc"
    url = "https://cds.climate.copernicus.eu/api"
    key = os.environ.get("CDSAPI_KEY")

    if config_path.exists():
        with open(config_path, 'r') as f:
            for line in f:
                if line.startswith("url:"): url = line.split(":", 1)[1].strip()
                if line.startswith("key:"): key = line.split(":", 1)[1].strip()
    return url, key

def main(target_date):
    logger.info(f"Task: Download ERA5 for {target_date}")
    
    # Save target date
    with open(DATE_FILE, "w") as f:
        f.write(target_date)

    url, key = load_cds_config()
    if not key:
        logger.error("API Key not found.")
        return

    client = cdsapi.Client(url=url, key=key)
    
    download_static(client)
    download_surface(client, target_date)
    download_upper(client, target_date)
    
    logger.info("All downloads completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="2023010112")
    args = parser.parse_args()
    main(args.date)
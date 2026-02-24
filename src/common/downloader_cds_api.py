import os
import logging
import argparse
import cdsapi
import pandas as pd
from pathlib import Path

# ==============================================================================
# Configuration & Setup
# ==============================================================================
# Configure Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("DownloadData")

# Path Configuration
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
SAVE_DIR = BASE_DIR / "assets" / "data" / "era5_neuralgcm"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"
CONFIG_FILE = BASE_DIR / ".cdsapirc"

# Ensure output directory exists
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Standard Pressure Levels (13 levels)
PRESSURE_LEVELS = [
    '50', '100', '150', '200', '250', '300', '400', 
    '500', '600', '700', '850', '925', '1000'
]

# ==============================================================================
# Helper Functions
# ==============================================================================
def load_cds_config():
    """Load CDS API credentials from project root or environment variables."""
    url = "https://cds.climate.copernicus.eu/api"
    key = os.environ.get("CDSAPI_KEY")

    if CONFIG_FILE.exists():
        logger.info(f"Loading config from: {CONFIG_FILE}")
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                if line.startswith("url:"): url = line.split(":", 1)[1].strip()
                if line.startswith("key:"): key = line.split(":", 1)[1].strip()
    
    return url, key

def download_static(client):
    """Download static variables (Geopotential, Land-sea mask)."""
    target = SAVE_DIR / "static.nc"
    if target.exists():
        logger.info("Static data already exists. Skipping.")
        return

    logger.info("Downloading static variables...")
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['geopotential', 'land_sea_mask'],
            'year': '2023', 'month': '01', 'day': '01', 'time': '00:00',
            'grid': [0.25, 0.25],
        },
        str(target)
    )

def download_surface(client, date_str):
    """Download surface variables including SST and Sea Ice (required by NeuralGCM)."""
    target = SAVE_DIR / f"surface_{date_str}.nc"
    if target.exists():
        logger.info(f"Surface data for {date_str} already exists. Skipping.")
        return

    dt = pd.to_datetime(date_str, format="%Y%m%d%H")
    # NeuralGCM requires history (T0 and T-6)
    times = [dt, dt - pd.Timedelta(hours=6)]
    
    # Extract unique components for CDS request
    years = list(set([t.strftime("%Y") for t in times]))
    months = list(set([t.strftime("%m") for t in times]))
    days = list(set([t.strftime("%d") for t in times]))
    hours = list(set([t.strftime("%H:%M") for t in times]))

    logger.info(f"Downloading surface variables for {date_str}...")
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '2m_temperature', 'mean_sea_level_pressure', 
                '10m_u_component_of_wind', '10m_v_component_of_wind',
                'sea_surface_temperature', 'sea_ice_cover' # Critical for NeuralGCM
            ],
            'year': years, 'month': months, 'day': days, 'time': hours,
            'grid': [0.25, 0.25],
        },
        str(target)
    )

def download_upper(client, date_str):
    """Download atmospheric variables on pressure levels."""
    target = SAVE_DIR / f"upper_{date_str}.nc"
    if target.exists():
        logger.info(f"Upper air data for {date_str} already exists. Skipping.")
        return

    dt = pd.to_datetime(date_str, format="%Y%m%d%H")
    times = [dt, dt - pd.Timedelta(hours=6)]
    
    years = list(set([t.strftime("%Y") for t in times]))
    months = list(set([t.strftime("%m") for t in times]))
    days = list(set([t.strftime("%d") for t in times]))
    hours = list(set([t.strftime("%H:%M") for t in times]))

    logger.info(f"Downloading upper air variables for {date_str}...")
    client.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'geopotential', 'specific_humidity', 'temperature',
                'u_component_of_wind', 'v_component_of_wind'
            ],
            'pressure_level': PRESSURE_LEVELS,
            'year': years, 'month': months, 'day': days, 'time': hours,
            'grid': [0.25, 0.25],
        },
        str(target)
    )

def main():
    parser = argparse.ArgumentParser(description="NeuralGCM Data Downloader")
    parser.add_argument("--date", type=str, default="2020082218", help="Target date YYYYMMDDHH")
    args = parser.parse_args()

    # Update target date record
    with open(DATE_FILE, "w") as f:
        f.write(args.date)

    # Initialize CDS Client
    url, key = load_cds_config()
    if not key:
        logger.error("CDS API Key not found. Please check .cdsapirc or env vars.")
        return

    client = cdsapi.Client(url=url, key=key)
    
    try:
        download_static(client)
        download_surface(client, args.date)
        download_upper(client, args.date)
        logger.info(f"All downloads complete. Data saved to: {SAVE_DIR}")
    except Exception as e:
        logger.error(f"Download failed: {e}")

if __name__ == "__main__":
    main()
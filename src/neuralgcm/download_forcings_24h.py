import os
import argparse
import logging
from pathlib import Path

import cdsapi
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("NeuralGCM.ForcingDownloader")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SAVE_DIR = BASE_DIR / "assets" / "data" / "raw"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def load_cds_config():
    config_path = BASE_DIR / ".cdsapirc"
    url = "https://cds.climate.copernicus.eu/api"
    key = os.environ.get("CDSAPI_KEY")

    if config_path.exists():
        with open(config_path, "r") as f:
            for line in f:
                if line.startswith("url:"):
                    url = line.split(":", 1)[1].strip()
                if line.startswith("key:"):
                    key = line.split(":", 1)[1].strip()
    return url, key


def main(date_str: str):
    dt = pd.to_datetime(date_str, format="%Y%m%d%H")
    times = pd.date_range(end=dt, periods=5, freq="6h")  # t-24..t
    dates = sorted({t.strftime("%Y-%m-%d") for t in times})
    hours = sorted({t.strftime("%H:%M") for t in times})

    out_path = SAVE_DIR / f"neuralgcm_forcings_24h_{date_str}.nc"
    if out_path.exists():
        logger.info(f"Exists, skip: {out_path}")
        return

    url, key = load_cds_config()
    if not key:
        raise RuntimeError("CDS API key not found. Set CDSAPI_KEY or .cdsapirc")

    logger.info(f"Downloading SST/SIC forcing window for {date_str}: {times[0]} .. {times[-1]}")
    client = cdsapi.Client(url=url, key=key)
    req = {
        "product_type": "reanalysis",
        "data_format": "netcdf",
        "variable": ["sea_surface_temperature", "sea_ice_cover"],
        "date": dates,
        "time": hours,
        "grid": [0.25, 0.25],
    }
    client.retrieve("reanalysis-era5-single-levels", req, str(out_path))
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDDHH, e.g. 2023010112")
    args = ap.parse_args()
    main(args.date)
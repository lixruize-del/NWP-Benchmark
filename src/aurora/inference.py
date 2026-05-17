import os
import sys
import logging
import argparse
import traceback
import torch
import xarray as xr
import numpy as np
from pathlib import Path
from unittest.mock import patch
from typing import List, Tuple, Dict

# --- Path Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent

# Append project root to path for src.common imports
sys.path.append(str(BASE_DIR))

from src.common.saver import Saver
from aurora import Aurora, Batch, Metadata

# --- Configuration ---
DATA_DIR = BASE_DIR / "assets" / "data" / "raw"
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "aurora"
OUTPUT_DIR = BASE_DIR / "outputs" / "aurora"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s', 
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Aurora.Inference")

# --- Variable Mapping ---
# Map Aurora internal names to NWPBench standard names
SURFACE_NAME_MAP = {
    "2t": "t2m",
    "10u": "u10",
    "10v": "v10",
    "msl": "msl"
}

def get_target_date():
    """Retrieve the target initialization date."""
    if DATE_FILE.exists():
        with open(DATE_FILE) as f:
            return f.read().strip()
    return "2023010112"

def setup_device():
    """Configure computation device."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Accelerator detected: {device_name}")
        return "cuda"
    else:
        logger.warning("No GPU detected. Running on CPU (Significant performance degradation expected).")
        return "cpu"

def load_data(target_date: str, device: str) -> Batch:
    """
    Load NetCDF files and construct the Aurora Batch object.
    
    Args:
        target_date (str): Date string YYYYMMDDHH.
        device (str): Device to load tensors onto.
        
    Returns:
        Batch: Aurora input batch containing history (T-6, T0).
    """
    static_file = DATA_DIR / "static.nc"
    surf_file = DATA_DIR / f"surface_{target_date}.nc"
    upper_file = DATA_DIR / f"upper_{target_date}.nc"

    if not surf_file.exists() or not upper_file.exists():
        raise FileNotFoundError(f"Missing data for {target_date}. Expected: {surf_file}")

    logger.info("Loading NetCDF input data...")
    static_ds = xr.open_dataset(static_file, engine="netcdf4")
    surf_ds = xr.open_dataset(surf_file, engine="netcdf4")
    upper_ds = xr.open_dataset(upper_file, engine="netcdf4")

    logger.info("Constructing Aurora Batch tensor...")

    AURORA_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    level_indices = [list(upper_ds.pressure_level.values).index(level) for level in AURORA_PRESSURE_LEVELS]
    
    # Aurora expects [Batch, Time, Lat, Lon]. 
    # Slicing [:2] assumes the file contains [T-6, T0] as the first two steps.
    batch = Batch(
        surf_vars={
            "2t": torch.from_numpy(surf_ds["t2m"].values[:2][None]).float().to(device),
            "10u": torch.from_numpy(surf_ds["u10"].values[:2][None]).float().to(device),
            "10v": torch.from_numpy(surf_ds["v10"].values[:2][None]).float().to(device),
            "msl": torch.from_numpy(surf_ds["msl"].values[:2][None]).float().to(device),
        },
        static_vars={
            "z": torch.from_numpy(static_ds["z"].values[0]).float().to(device),
            "slt": torch.from_numpy(static_ds["slt"].values[0]).float().to(device),
            "lsm": torch.from_numpy(static_ds["lsm"].values[0]).float().to(device),
        },
        atmos_vars={
            "t": torch.from_numpy(upper_ds["t"].isel(pressure_level=level_indices).values[:2][None]).float().to(device),
            "u": torch.from_numpy(upper_ds["u"].isel(pressure_level=level_indices).values[:2][None]).float().to(device),
            "v": torch.from_numpy(upper_ds["v"].isel(pressure_level=level_indices).values[:2][None]).float().to(device),
            "q": torch.from_numpy(upper_ds["q"].isel(pressure_level=level_indices).values[:2][None]).float().to(device),
            "z": torch.from_numpy(upper_ds["z"].isel(pressure_level=level_indices).values[:2][None]).float().to(device),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_ds.latitude.values).to(device),
            lon=torch.from_numpy(surf_ds.longitude.values).to(device),
            time=(surf_ds.valid_time.values.astype("datetime64[s]").tolist()[1],),
            atmos_levels=tuple(AURORA_PRESSURE_LEVELS),
        ),
    )
    return batch

def load_model(device: str) -> Aurora:
    """
    Load the Aurora model using local weights, bypassing HuggingFace hub connection.
    """
    weights_path = WEIGHTS_DIR / "aurora-0.25-pretrained.ckpt"
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {weights_path}")

    logger.info(f"Loading model checkpoint: {weights_path.name}")
    model = Aurora(use_lora=False)

    # Mock function to bypass HF download
    def fake_download(*args, **kwargs):
        return str(weights_path)

    try:
        # Monkey patch hf_hub_download to load local file
        with patch('aurora.model.aurora.hf_hub_download', side_effect=fake_download):
            model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
            logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load checkpoint via patch: {e}")
        raise e

    return model.to(device).eval()

def process_and_save_output(pred: Batch, 
                            init_time: str, 
                            lead_time_hours: int, 
                            saver: Saver):
    """
    Extract data from Aurora Batch, flatten it, and save using standard Saver.
    
    Args:
        pred (Batch): Output batch from model.
        init_time (str): Initialization time string.
        lead_time_hours (int): Forecast lead time.
        saver (Saver): Instance of common Saver.
    """
    logger.info("Processing output tensors for saving...")
    
    data_list = []
    channel_mapping = []
    
    # 1. Process Surface Variables
    # Shape: [Batch, Time, Lat, Lon]. Take last time step (-1) and remove Batch (0).
    for aurora_name, tensor in pred.surf_vars.items():
        if aurora_name in SURFACE_NAME_MAP:
            std_name = SURFACE_NAME_MAP[aurora_name]
            # [B, T, H, W] -> [H, W]
            val = tensor[0, -1, ...].detach().cpu().numpy()
            data_list.append(val)
            channel_mapping.append(std_name)

    # 2. Process Atmospheric Variables
    # Shape: [Batch, Time, Level, Lat, Lon]
    levels = pred.metadata.atmos_levels
    
    for aurora_name, tensor in pred.atmos_vars.items():
        # [B, T, Lev, H, W] -> [Lev, H, W]
        val_levels = tensor[0, -1, ...].detach().cpu().numpy()
        
        for i, level in enumerate(levels):
            # Construct standard name: e.g., "t_850", "z_500"
            std_name = f"{aurora_name}_{level}"
            data_list.append(val_levels[i])
            channel_mapping.append(std_name)
            
    # 3. Aggregate and Save
    combined_data = np.stack(data_list, axis=0) # [C, H, W]
    
    # Extract coordinates from metadata
    lats = pred.metadata.lat.cpu().numpy()
    lons = pred.metadata.lon.cpu().numpy()
    
    saver.save(
        data=combined_data,
        channel_mapping=channel_mapping,
        init_time_str=init_time,
        lead_time_hours=lead_time_hours,
        lat_values=lats,
        lon_values=lons
    )
    logger.info(f"Saved formatted output via Saver to {OUTPUT_DIR}")

def main():
    parser = argparse.ArgumentParser(description="Aurora Inference Runner")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYYMMDDHH")
    args = parser.parse_args()

    try:
        device = setup_device()
        target_date = args.date if args.date else get_target_date()
        logger.info(f"Target Initial Time (T0): {target_date}")

        # 1. Load Resources
        batch = load_data(target_date, device)
        model = load_model(device)

        # 2. Run Inference
        logger.info("Running inference (Forecast +6h)...")
        with torch.inference_mode():
            pred = model.forward(batch)

        # 3. Save Results
        saver = Saver(save_root=str(OUTPUT_DIR))
        process_and_save_output(pred, target_date, 6, saver)

    except Exception as e:
        logger.error(f"Inference process failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
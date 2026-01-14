import os
import sys
import glob
import logging
import argparse
import datetime
import torch
import numpy as np
from pathlib import Path
from anemoi.inference.runners.simple import SimpleRunner

# --- Path Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent

# Append project root to path for src.common imports
sys.path.append(str(BASE_DIR))

from src.common.saver import Saver

# --- Configuration ---
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "aifs"
DATA_DIR = BASE_DIR / "assets" / "data" / "processed_aifs"
OUTPUT_DIR = BASE_DIR / "outputs" / "aifs"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s', 
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("AIFS.Inference")

# --- Variable Mapping ---
# Map AIFS/ECMWF short names to NWPBench standard names
AIFS_VAR_MAP = {
    "2t": "t2m",
    "10u": "u10",
    "10v": "v10",
    "msl": "msl",
    "sp": "sp",
    "tp": "tp"
}

# Grid Resolution (Standard ECMWF 0.25 degree)
LAT_RES = 721
LON_RES = 1440

def setup_environment(deterministic: bool = True):
    """
    Configure the PyTorch and CUDA environment for inference.
    """
    # Memory optimization (PyTorch 2.4+)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['ANEMOI_INFERENCE_NUM_CHUNKS'] = '16'
    
    if deterministic:
        logger.info("Enabling deterministic mode...")
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    if torch.cuda.is_available():
        logger.info(f"Accelerator detected: {torch.cuda.get_device_name(0)}")
        try:
            import flash_attn
            logger.info(f"FlashAttention available: v{flash_attn.__version__}")
        except ImportError:
            logger.warning("FlashAttention not found. Falling back to SDPA.")
    else:
        logger.warning("No GPU detected.")

def load_input_state() -> dict:
    """
    Load the most recent initialization state from processed .npz files.
    """
    pattern = os.path.join(DATA_DIR, "init_*.npz")
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No input data found in {DATA_DIR}")
    
    latest_file = files[-1]
    logger.info(f"Loading input file: {os.path.basename(latest_file)}")
    
    data = np.load(latest_file, allow_pickle=True)
    
    # Reconstruct dictionary structure
    fields = {k: data[k] for k in data.files if k != 'date'}
    
    # Reconstruct datetime object
    date_val = data['date']
    date_str = str(date_val.item()) if isinstance(date_val, np.ndarray) else str(date_val)
    date_obj = datetime.datetime.fromisoformat(date_str)
    
    return dict(date=date_obj, fields=fields)

def get_state_attr(state, key):
    """Helper to access attributes from either object or dictionary."""
    if hasattr(state, key):
        return getattr(state, key)
    elif isinstance(state, dict) and key in state:
        return state[key]
    return None

def process_and_save(step: int, state: dict, start_date: datetime.datetime, saver: Saver):
    """
    Process the AIFS output state and save it using the standardized Saver.

    Args:
        step (int): Current forecast step index.
        state (dict): Output state from Anemoi runner containing fields.
        start_date (datetime): Initialization time.
        saver (Saver): Instance of the common Saver class.
    """
    valid_time = get_state_attr(state, 'date')
    fields = get_state_attr(state, 'fields')
    
    if valid_time is None:
        valid_time = start_date + datetime.timedelta(hours=step * 6)
    
    # Calculate forecast lead time
    lead_time_hours = int((valid_time - start_date).total_seconds() / 3600)
    init_time_str = start_date.strftime("%Y%m%d%H")

    # Ensure fields is a dictionary
    if not isinstance(fields, dict):
        try:
            fields = dict(fields)
        except Exception as e:
            logger.error(f"Failed to convert fields to dict: {e}")
            return

    # --- Flatten Dictionary to Tensor ---
    data_list = []
    channel_mapping = []
    
    # Sort keys to ensure deterministic channel order
    # keys are expected to be like 'z_500', '2t', etc.
    sorted_keys = sorted(fields.keys())
    
    for key in sorted_keys:
        val = fields[key]
        
        # Determine standard name
        if key in AIFS_VAR_MAP:
            std_name = AIFS_VAR_MAP[key]
        else:
            std_name = key  # Fallback (e.g., z_500 usually matches standard schema)

        # Handle tensor vs numpy
        if hasattr(val, 'cpu'):
            val = val.detach().cpu().numpy()
        
        # Ensure 2D shape [H, W]
        if val.ndim == 3:
            val = val.squeeze(0)
            
        data_list.append(val)
        channel_mapping.append(std_name)
    
    # Stack into [C, H, W]
    combined_data = np.stack(data_list, axis=0)

    # Generate Coordinates (AIFS Standard 0.25 deg)
    lats = np.linspace(90, -90, LAT_RES)
    lons = np.linspace(0, 360, LON_RES, endpoint=False)

    # Save via Saver
    saver.save(
        data=combined_data,
        channel_mapping=channel_mapping,
        init_time_str=init_time_str,
        lead_time_hours=lead_time_hours,
        lat_values=lats,
        lon_values=lons
    )
    logger.info(f"Forecast step saved: +{lead_time_hours}h")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead-time", type=int, default=24, help="Forecast lead time in hours")
    parser.add_argument("--checkpoint", type=str, default="aifs-single-mse-1.0.ckpt", help="Model checkpoint filename")
    args = parser.parse_args()

    setup_environment()

    # 1. Load Data
    try:
        input_state = load_input_state()
        start_date = input_state['date']
        logger.info(f"Initialization Time (T0): {start_date}")
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        sys.exit(1)

    # 2. Load Model
    checkpoint_path = os.path.join(WEIGHTS_DIR, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint missing: {checkpoint_path}")
        sys.exit(1)

    logger.info(f"Loading model: {args.checkpoint}")
    
    try:
        # Initialize Runner
        runner = SimpleRunner(checkpoint=checkpoint_path, device="cuda")
        logger.info("Model loaded successfully.")

        # Initialize Saver
        saver = Saver(save_root=str(OUTPUT_DIR))

        # 3. Run Inference Loop
        logger.info(f"Starting inference ({args.lead_time} hours)...")
        forecast = runner.run(input_state=input_state, lead_time=args.lead_time)
        
        for step, state in enumerate(forecast, start=1):
            current_time = get_state_attr(state, 'date')
            if current_time is None:
                 current_time = start_date + datetime.timedelta(hours=step * 6)
                 
            logger.info(f"Step {step}: Valid Time {current_time}")
            
            # Save step output
            process_and_save(step, state, start_date, saver)
            
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("AIFS inference task completed.")

    except Exception as e:
        logger.error(f"Inference process failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
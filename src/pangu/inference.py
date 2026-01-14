import os
import sys
import argparse
import logging
import datetime
import numpy as np
import onnxruntime as ort
from pathlib import Path

# --- Path Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent

# Append project root to path for src.common imports
sys.path.append(str(BASE_DIR))

from src.common.saver import Saver

# --- Configuration ---
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "pangu"
DATA_DIR = BASE_DIR / "assets" / "data" / "processed_pangu"
OUTPUT_DIR = BASE_DIR / "outputs" / "pangu"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s', 
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Pangu.Inference")

# --- Pangu Definitions ---
# Pangu-Weather output structure specifics
UPPER_VARS = ['z', 'q', 't', 'u', 'v']
SURFACE_VARS = ['msl', 'u10', 'v10', 't2m']
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# Grid Resolution (0.25 degree)
LAT_RES = 721
LON_RES = 1440


def get_start_date():
    """
    Retrieve the initialization date from the target file or fallback to current time.
    """
    if os.path.exists(DATE_FILE):
        with open(DATE_FILE, 'r') as f:
            date_str = f.read().strip()
        
        # Try parsing different formats
        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
        try:
            return datetime.datetime.strptime(date_str, "%Y%m%d%H")
        except ValueError:
            pass
            
    logger.warning("Target date file not found or invalid. Using current system time.")
    return datetime.datetime.now()

def generate_channel_names():
    """
    Generate the flat list of variable names corresponding to the merged Pangu output tensor.
    
    Order matches Pangu's (5, 13, H, W) flattened to (65, H, W) + Surface (4, H, W).
    Sequence:
      1. For each upper variable (z, q, t, u, v):
           For each level (1000...50)
      2. Surface variables (msl, u10, v10, t2m)
    """
    channel_names = []
    
    # Upper air variables
    for var in UPPER_VARS:
        for level in PRESSURE_LEVELS:
            channel_names.append(f"{var}_{level}")
            
    # Surface variables
    channel_names.extend(SURFACE_VARS)
    
    return channel_names

def run_inference(lead_time_hours: int):
    """
    Execute the Pangu-Weather inference workflow.
    
    Args:
        lead_time_hours (int): Total forecast duration in hours (must be multiple of 24).
    """
    # 1. Validation
    model_path = os.path.join(WEIGHTS_DIR, "pangu_weather_24.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing: {model_path}")

    input_upper_path = os.path.join(DATA_DIR, "input_upper.npy")
    input_surface_path = os.path.join(DATA_DIR, "input_surface.npy")
    
    if not os.path.exists(input_upper_path) or not os.path.exists(input_surface_path):
        raise FileNotFoundError("Input data missing. Please run src/pangu/prepare.py first.")

    # 2. Data Loading
    logger.info("Loading input data...")
    # Shape: (5, 13, 721, 1440) -> (Vars, Levels, Lat, Lon)
    input_upper = np.load(input_upper_path).astype(np.float32)
    # Shape: (4, 721, 1440) -> (Vars, Lat, Lon)
    input_surface = np.load(input_surface_path).astype(np.float32)

    # 3. ONNX Session Initialization
    logger.info("Initializing ONNX Session (CUDA)...")
    
    sess_options = ort.SessionOptions()
    sess_options.enable_cpu_mem_arena = False
    sess_options.enable_mem_pattern = False
    sess_options.enable_mem_reuse = False
    sess_options.intra_op_num_threads = 4
    
    cuda_provider_options = {'arena_extend_strategy': 'kSameAsRequested'}
    
    try:
        session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options, 
            providers=[('CUDAExecutionProvider', cuda_provider_options)]
        )
    except Exception as e:
        logger.error(f"Failed to initialize ONNX Session: {e}")
        logger.error("Ensure CUDA is available and onnxruntime-gpu is installed.")
        sys.exit(1)

    # 4. Inference Loop
    start_date = get_start_date()
    init_time_str = start_date.strftime("%Y%m%d%H")
    logger.info(f"Initialization Time: {start_date}")
    
    steps = lead_time_hours // 24
    if steps < 1: 
        logger.warning("Lead time < 24h, defaulting to 1 step (24h).")
        steps = 1
    
    curr_upper = input_upper
    curr_surface = input_surface
    
    # Prepare standard saver
    saver = Saver(save_root=str(OUTPUT_DIR))
    channel_mapping = generate_channel_names()
    
    # Generate coordinates for Pangu (0.25 deg)
    lats = np.linspace(90, -90, LAT_RES)
    lons = np.linspace(0, 360, LON_RES, endpoint=False)
    
    logger.info(f"Starting inference ({steps * 24} hours total, {steps} steps)...")

    for i in range(steps):
        step_num = i + 1
        forecast_hour = step_num * 24
        
        logger.info(f"Step {step_num}: Forecasting +{forecast_hour}h")
        
        try:
            # Inputs must match ONNX node names: 'input', 'input_surface'
            outputs = session.run(
                None, 
                {'input': curr_upper, 'input_surface': curr_surface}
            )
            pred_upper, pred_surface = outputs
            
        except Exception as e:
            logger.error(f"Inference computation failed: {e}")
            sys.exit(1)

        # 5. Data Unification for Saver
        # Flatten upper air data: (5, 13, H, W) -> (65, H, W)
        # Note: Reshape preserves the (Var, Level) order consistent with generation logic
        pred_upper_flat = pred_upper.reshape(-1, LAT_RES, LON_RES)
        
        # Concatenate with surface data: (65, H, W) + (4, H, W) -> (69, H, W)
        combined_pred = np.concatenate([pred_upper_flat, pred_surface], axis=0)

        # 6. Save using Saver
        saver.save(
            data=combined_pred,
            channel_mapping=channel_mapping,
            init_time_str=init_time_str,
            lead_time_hours=forecast_hour,
            lat_values=lats,
            lon_values=lons
        )

        # Update inputs for next autoregressive step
        curr_upper = pred_upper
        curr_surface = pred_surface

    logger.info("Pangu inference completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pangu-Weather Inference Script")
    parser.add_argument("--lead-time", type=int, default=24, help="Total forecast hours (multiple of 24)")
    args = parser.parse_args()
    
    run_inference(args.lead_time)
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
UPPER_VARS = ['z', 'q', 't', 'u', 'v']
SURFACE_VARS = ['msl', 'u10', 'v10', 't2m']
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# Grid Resolution (0.25 degree)
LAT_RES = 721
LON_RES = 1440

# Available model steps and corresponding filenames
MODEL_STEPS = {
    1:  "pangu_weather_1.onnx",
    3:  "pangu_weather_3.onnx",
    6:  "pangu_weather_6.onnx",
    24: "pangu_weather_24.onnx"
}
AVAILABLE_STEPS = sorted(MODEL_STEPS.keys(), reverse=True)  # [24, 6, 3, 1]


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

def decompose_lead_time(lead_time_hours):
    """
    Decompose total lead time into available step sizes (greedy largest first).
    Returns a list of step sizes (e.g., [24, 6, 1] for 31 hours).
    """
    remaining = lead_time_hours
    steps = []
    for step in AVAILABLE_STEPS:
        while remaining >= step:
            steps.append(step)
            remaining -= step
    if remaining != 0:
        raise ValueError(f"Lead time {lead_time_hours}h cannot be composed from available steps {AVAILABLE_STEPS}. "
                         f"Remaining {remaining}h after decomposition.")
    return steps

def run_inference(lead_time_hours: int):
    """
    Execute the Pangu-Weather inference workflow for any lead time (hours).
    Automatically selects appropriate model steps (1h, 3h, 6h, 24h) to achieve the total.
    """
    # 1. Decompose lead time into model steps
    try:
        step_sequence = decompose_lead_time(lead_time_hours)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Lead time {lead_time_hours}h will be produced using steps: {step_sequence}")

    # 2. Validate all required model files exist
    model_paths = {}
    for step in set(step_sequence):
        fname = MODEL_STEPS[step]
        path = os.path.join(WEIGHTS_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required model file missing: {path}")
        model_paths[step] = path

    # 3. Load input data
    input_upper_path = os.path.join(DATA_DIR, "input_upper.npy")
    input_surface_path = os.path.join(DATA_DIR, "input_surface.npy")
    
    if not os.path.exists(input_upper_path) or not os.path.exists(input_surface_path):
        raise FileNotFoundError("Input data missing. Please run src/pangu/prepare.py first.")

    logger.info("Loading input data...")
    # Shape: (5, 13, 721, 1440)
    input_upper = np.load(input_upper_path).astype(np.float32)
    # Shape: (4, 721, 1440)
    input_surface = np.load(input_surface_path).astype(np.float32)

    # 4. Initialize saver and metadata
    start_date = get_start_date()
    init_time_str = start_date.strftime("%Y%m%d%H")
    logger.info(f"Initialization Time: {start_date}")

    saver = Saver(save_root=str(OUTPUT_DIR))
    channel_mapping = generate_channel_names()
    lats = np.linspace(90, -90, LAT_RES)
    lons = np.linspace(0, 360, LON_RES, endpoint=False)

    # 5. Iterate through step sequence
    curr_upper = input_upper
    curr_surface = input_surface
    accumulated_hours = 0

    for i, step_hours in enumerate(step_sequence):
        accumulated_hours += step_hours
        logger.info(f"Step {i+1}/{len(step_sequence)}: +{step_hours}h (total +{accumulated_hours}h)")

        # Load the appropriate model session (cache by step size)
        model_path = model_paths[step_hours]
        if step_hours not in model_paths or not hasattr(run_inference, "sessions"):
            # Simple caching: store sessions in function attribute
            if not hasattr(run_inference, "sessions"):
                run_inference.sessions = {}
            if step_hours not in run_inference.sessions:
                logger.info(f"Loading {step_hours}h model...")
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
                    logger.error(f"Failed to initialize ONNX Session for {step_hours}h model: {e}")
                    sys.exit(1)
                run_inference.sessions[step_hours] = session
            session = run_inference.sessions[step_hours]
        else:
            session = run_inference.sessions[step_hours]

        # Run inference
        try:
            outputs = session.run(
                None,
                {'input': curr_upper, 'input_surface': curr_surface}
            )
            pred_upper, pred_surface = outputs
        except Exception as e:
            logger.error(f"Inference computation failed at step {i+1}: {e}")
            sys.exit(1)

        # Save intermediate result
        pred_upper_flat = pred_upper.reshape(-1, LAT_RES, LON_RES)
        combined_pred = np.concatenate([pred_upper_flat, pred_surface], axis=0)

        saver.save(
            data=combined_pred,
            channel_mapping=channel_mapping,
            init_time_str=init_time_str,
            lead_time_hours=accumulated_hours,
            lat_values=lats,
            lon_values=lons
        )

        # Update for next step
        curr_upper = pred_upper
        curr_surface = pred_surface

    logger.info("Pangu inference completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pangu-Weather Inference Script (auto‑selects model steps)")
    parser.add_argument("--lead-time", type=int, default=6,
                        help="Total forecast hours (will be composed from 1,3,6,24h steps)")
    args = parser.parse_args()
    
    run_inference(args.lead_time)
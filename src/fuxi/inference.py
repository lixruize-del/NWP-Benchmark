import os
import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import onnxruntime as ort

# --- Path Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from src.common.saver import Saver

# --- Configuration ---
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "fuxi"
DATA_DIR = BASE_DIR / "assets" / "data" / "processed_fuxi"
OUTPUT_DIR = BASE_DIR / "outputs" / "fuxi"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FuXi.Inference")

# --- Constants (must match prepare.py) ---
PL_NAMES = ['z', 't', 'u', 'v', 'r']
SFC_NAMES = ['t2m', 'u10', 'v10', 'msl', 'tp']
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
STEP_HOURS = 6


def get_target_date():
    """Read target date from DATE_FILE."""
    if not DATE_FILE.exists():
        raise FileNotFoundError(f"Target date file not found: {DATE_FILE}")
    date_str = DATE_FILE.read_text().strip()
    # Accept both YYYYMMDDHH and YYYY-MM-DD HH:MM:SS formats
    for fmt in ("%Y%m%d%H", "%Y-%m-%d %H:%M:%S"):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date string: {date_str}")


def parse_fuxi_channel(channel_name: str):
    """
    Convert FuXi channel name (e.g., 'Z50', 'T2M') to (short_name, level_or_None).
    - Pressure levels: 'Z50' -> ('z', 50)
    - Surface: 'T2M' -> ('t2m', None), 'TP' -> ('tp6h', None)
    """
    if channel_name in ['T2M', 'U10', 'V10', 'MSL']:
        return channel_name.lower(), None
    if channel_name == 'TP':
        return 'tp6h', None
    # Pressure level: first char is variable, rest is level number
    var_upper = channel_name[0]
    level_str = channel_name[1:]
    var_map = {'Z': 'z', 'T': 't', 'U': 'u', 'V': 'v', 'R': 'r'}
    if var_upper in var_map and level_str.isdigit():
        return var_map[var_upper], int(level_str)
    raise ValueError(f"Unrecognized channel name: {channel_name}")


def generate_channel_mapping(fuxi_level_names):
    """Convert FuXi level names to Saver-compatible channel mapping."""
    mapping = []
    for name in fuxi_level_names:
        short, level = parse_fuxi_channel(name)
        if level is None:
            mapping.append(short)          # surface
        else:
            mapping.append(f"{short}_{level}")  # pressure
    return mapping


def time_encoding(init_time, total_step, freq=6):
    """Generate time embeddings for FuXi (from original script)."""
    init_time = np.array([init_time])
    tembs = []
    for i in range(total_step):
        hours = np.array([pd.Timedelta(hours=t*freq) for t in [i-1, i, i+1]])
        times = init_time[:, None] + hours[None]
        times = [pd.Period(t, 'H') for t in times.reshape(-1)]
        times = [(p.day_of_year/366, p.hour/24) for p in times]
        temb = np.array(times, dtype=np.float32)
        temb = np.concatenate([np.sin(temb), np.cos(temb)], axis=-1)
        temb = temb.reshape(1, -1)               # shape (1, 6)
        tembs.append(temb)
    return np.stack(tembs)                       # shape (total_step, 1, 6)


def load_model(model_path: str):
    """Load ONNX model with optimal settings."""
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 1

    cuda_options = {'arena_extend_strategy': 'kSameAsRequested'}
    providers = [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
    except Exception:
        logger.warning("CUDA unavailable, falling back to CPU.")
        session = ort.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
    return session


def load_input_data(target_date):
    """Load preprocessed input NetCDF for the given date."""
    input_path = DATA_DIR / f"input_{target_date.strftime('%Y%m%d%H')}.nc"
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    logger.info(f"Loading input from {input_path}")
    data = xr.open_dataarray(input_path)
    # Validate shape and orientation
    if data.dims != ('time', 'level', 'lat', 'lon'):
        raise ValueError(f"Unexpected dimensions: {data.dims}")
    if data.shape != (2, 70, 721, 1440):
        raise ValueError(f"Expected shape (2,70,721,1440), got {data.shape}")
    # Ensure latitude decreasing (north to south)
    if data.lat[0] < data.lat[-1]:
        data = data.isel(lat=slice(None, None, -1))
        logger.info("Reversed latitude order to north‑to‑south.")
    return data


def run_inference(num_steps):
    """Main inference routine."""
    # --- Determine initialization time ---
    init_time = get_target_date()
    init_time_str = init_time.strftime("%Y%m%d%H")
    logger.info(f"Initialization time: {init_time_str}")

    # --- Load input data ---
    data = load_input_data(init_time)
    lats = data.lat.values
    lons = data.lon.values
    channel_names = data.level.values.tolist()  # e.g., ['Z50', 'T50', ...]
    channel_mapping = generate_channel_mapping(channel_names)

    # --- Prepare for inference ---
    total_step = sum(num_steps)
    tembs = time_encoding(init_time, total_step)   # (total_step, 1, 6)

    # Convert to numpy and add batch dimension
    input_np = data.values[None]  # (1, 2, 70, 721, 1440)
    logger.info(f"Input shape: {input_np.shape}")

    # --- Load models (cascade stages) ---
    stages = ['short', 'medium', 'long']
    models = {}
    for stage in stages:
        model_path = WEIGHTS_DIR / f"{stage}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing: {model_path}")
        logger.info(f"Loading {stage} model...")
        models[stage] = load_model(str(model_path))

    # --- Saver ---
    saver = Saver(save_root=str(OUTPUT_DIR))

    step = 0
    for stage_idx, num_step in enumerate(num_steps):
        stage = stages[stage_idx]
        session = models[stage]

        logger.info(f"Running {stage} stage ({num_step} steps)...")
        start_time = time.perf_counter()

        for _ in range(num_step):
            temb = tembs[step]          # shape (1, 6) – rank 2
            # FuXi expects input (1,2,70,721,1440) and temb (1,6)
            new_input, = session.run(None, {'input': input_np, 'temb': temb})
            # new_input shape: (1,2,70,721,1440)
            output = new_input[0, -1]   # (70,721,1440)

            lead_hours = (step + 1) * STEP_HOURS
            logger.info(f"Step {step+1:02d} (lead +{lead_hours:03d}h): "
                        f"min={output.min():.2f}, max={output.max():.2f}")

            # Save using Saver
            saver.save(
                data=output,
                channel_mapping=channel_mapping,
                init_time_str=init_time_str,
                lead_time_hours=lead_hours,
                lat_values=lats,
                lon_values=lons,
            )

            # Prepare input for next step
            input_np = new_input
            step += 1

        elapsed = time.perf_counter() - start_time
        logger.info(f"{stage} stage completed in {elapsed:.2f} sec")

    logger.info("All inference steps finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FuXi Inference (simplified version)")
    parser.add_argument(
        "--num_steps", type=int, nargs=3, default=[1, 0, 0],
        help="Number of steps for [short, medium, long] stages. Each step = 6h. Default: 20 20 20"
    )
    parser.add_argument(
        "--lead-time", type=int, default=None,
        help="Optional total forecast hours for validation (must equal sum(num_steps)*6)."
    )
    args = parser.parse_args()

    total_hours = sum(args.num_steps) * STEP_HOURS
    if args.lead_time is not None and args.lead_time != total_hours:
        raise ValueError(
            f"Total forecast hours from --num_steps ({total_hours}h) "
            f"does not match --lead-time ({args.lead_time}h)."
        )

    try:
        run_inference(args.num_steps)
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)
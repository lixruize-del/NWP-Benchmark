import os
import sys
import argparse
import logging
import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort

# --- Path Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from src.common.saver import Saver

# --- Configuration ---
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "fengwu"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Fengwu.Inference")

# Fengwu uses 6-hour step autoregression in the official demo
STEP_HOURS = 6

# Same pressure level list you used during prepare (must match channel order!)
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

LAT_RES = 721
LON_RES = 1440


def resolve_fengwu_variant(model_name: str, explicit: Optional[str]) -> str:
    """v1/v2 for assets/data/fengwu_* and outputs/fengwu_*."""
    if explicit is not None:
        return explicit
    name = model_name.lower()
    if "v2" in name:
        return "v2"
    return "v1"


def paths_for_variant(variant: str) -> Tuple[Path, Path]:
    data_dir = BASE_DIR / "assets" / "data" / f"fengwu_{variant}"
    output_dir = BASE_DIR / "outputs" / f"fengwu_{variant}"
    return data_dir, output_dir


def get_start_date():
    if os.path.exists(DATE_FILE):
        date_str = Path(DATE_FILE).read_text().strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d%H"):
            try:
                return datetime.datetime.strptime(date_str, fmt)
            except ValueError:
                pass
        try:
            return datetime.datetime.fromisoformat(date_str)
        except ValueError:
            pass

    logger.warning("Target date file not found or invalid. Using current system time.")
    return datetime.datetime.now()


def generate_channel_names():
    """
    Must match your prepared data channel order.
    surface: u10, v10, t2m, msl
    upper: z(levels), q(levels), u(levels), v(levels), t(levels)
    """
    names = ["u10", "v10", "t2m", "msl"]
    for var in ["z", "q", "u", "v", "t"]:
        for lev in PRESSURE_LEVELS:
            names.append(f"{var}_{lev}")
    if len(names) != 69:
        raise RuntimeError(f"Channel mapping should be 69, got {len(names)}")
    return names


def run_inference(
    lead_time_hours: int,
    model_name: str = "fengwu_v1.onnx",
    variant: Optional[str] = None,
):
    if lead_time_hours < STEP_HOURS:
        logger.warning(f"Lead time < {STEP_HOURS}h, defaulting to 1 step ({STEP_HOURS}h).")
        lead_time_hours = STEP_HOURS
    if lead_time_hours % STEP_HOURS != 0:
        raise ValueError(f"--lead-time must be multiple of {STEP_HOURS} hours for Fengwu.")

    steps = lead_time_hours // STEP_HOURS

    var = resolve_fengwu_variant(model_name, variant)
    data_dir, output_dir = paths_for_variant(var)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Variant {var}: data_dir={data_dir}, output_dir={output_dir}")

    model_path = WEIGHTS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file missing: {model_path}")

    mean_path = CURRENT_DIR / "normalization_constants" / "data_mean.npy"
    std_path = CURRENT_DIR / "normalization_constants" / "data_std.npy"
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(f"Missing mean/std: {mean_path}, {std_path}")

    input1_path = data_dir / "input1.npy"
    input2_path = data_dir / "input2.npy"
    if not input1_path.exists() or not input2_path.exists():
        raise FileNotFoundError(
            f"Missing inputs under {data_dir}. Run prepare with the same variant "
            f"(e.g. --source era5 -> fengwu_v1, --source hres -> fengwu_v2, or --variant)."
        )

    logger.info("Loading inputs & normalization stats...")
    data_mean = np.load(mean_path).astype(np.float32)[:, None, None]
    data_std = np.load(std_path).astype(np.float32)[:, None, None]

    input1 = np.load(input1_path).astype(np.float32)
    input2 = np.load(input2_path).astype(np.float32)

    if input1.shape != (69, LAT_RES, LON_RES) or input2.shape != (69, LAT_RES, LON_RES):
        raise ValueError(f"Expected inputs (69,{LAT_RES},{LON_RES}), got {input1.shape}, {input2.shape}")

    input1n = (input1 - data_mean) / data_std
    input2n = (input2 - data_mean) / data_std
    curr = np.concatenate([input1n, input2n], axis=0)[None, ...].astype(np.float32)

    logger.info("Initializing ONNX Session (CUDA if available)...")
    sess_options = ort.SessionOptions()
    sess_options.enable_cpu_mem_arena = False
    sess_options.enable_mem_pattern = False
    sess_options.enable_mem_reuse = False
    sess_options.intra_op_num_threads = 1

    cuda_provider_options = {"arena_extend_strategy": "kSameAsRequested"}

    try:
        providers = [("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"]
        session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
    except Exception:
        logger.warning("CUDAExecutionProvider unavailable; falling back to CPUExecutionProvider.")
        session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"])

    saver = Saver(save_root=str(output_dir))
    channel_mapping = generate_channel_names()
    lats = np.linspace(90, -90, LAT_RES)
    lons = np.linspace(0, 360, LON_RES, endpoint=False)

    start_date = get_start_date()
    init_time_str = start_date.strftime("%Y%m%d%H")
    logger.info(f"Initialization Time: {start_date} | steps={steps} ({lead_time_hours}h total)")

    for i in range(steps):
        forecast_hour = (i + 1) * STEP_HOURS
        logger.info(f"Step {i+1}/{steps}: Forecasting +{forecast_hour}h")

        out = session.run(None, {"input": curr})[0]

        curr = np.concatenate((curr[:, 69:], out[:, :69]), axis=1).astype(np.float32)

        pred_norm = out[0, :69]
        pred = (pred_norm * data_std) + data_mean

        saver.save(
            data=pred.astype(np.float32),
            channel_mapping=channel_mapping,
            init_time_str=init_time_str,
            lead_time_hours=forecast_hour,
            lat_values=lats,
            lon_values=lons,
        )

    logger.info("Fengwu inference completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fengwu Inference Script (Saver-compatible)")
    parser.add_argument(
        "--lead-time",
        type=int,
        default=6,
        help="Total forecast hours (multiple of 6). e.g., 12=2 steps",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fengwu_v1.onnx",
        help="Model filename under assets/weights/fengwu/",
    )
    parser.add_argument(
        "--variant",
        choices=["v1", "v2"],
        default=None,
        help="Use assets/data/fengwu_{variant} and outputs/fengwu_{variant}. Default: infer from --model.",
    )
    args = parser.parse_args()

    try:
        run_inference(args.lead_time, args.model, variant=args.variant)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise SystemExit(1)

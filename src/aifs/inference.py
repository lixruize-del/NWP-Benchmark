"""AIFS inference: ``prepare.py`` NPZ → ``SimpleRunner`` → N320→0.25° NetCDF."""
import argparse
import datetime
import glob
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import earthkit.regrid as ekr
from anemoi.inference.runners.simple import SimpleRunner

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.common.saver import Saver

WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "aifs"
DATA_DIR = BASE_DIR / "assets" / "data" / "processed_aifs"
OUTPUT_DIR = BASE_DIR / "outputs" / "aifs"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

LAT_RES, LON_RES = 721, 1440
N320_POINTS = 542080
SOURCE_GRID = {"grid": "N320"}
TARGET_GRID = {"grid": (0.25, 0.25)}

DEFAULT_CHECKPOINT = "aifs-single-mse-1.1.ckpt"
DEFAULT_LEAD_HOURS = 6

AIFS_VAR_MAP = {
    "2t": "t2m",
    "10u": "u10",
    "10v": "v10",
    "msl": "msl",
    "sp": "sp",
    "tcw": "tcwv",
    "skt": "skt",
    "2d": "d2m",
    "tp": "tp",
    "cp": "cp",
    "tcc": "tcc",
    "lsm": "lsm",
    "z": "z",
    "sdor": "sdor",
    "slor": "slor",
    "stl1": "stl1",
    "stl2": "stl2",
    "swvl1": "swvl1",
    "swvl2": "swvl2",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AIFS.Inference")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_environment(deterministic: bool = False) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS", "16")
    if deterministic:
        logger.info("Deterministic CUDA mode (slower)")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        try:
            import flash_attn

            logger.info("FlashAttention: v%s", flash_attn.__version__)
        except ImportError:
            logger.warning("FlashAttention not found; using SDPA.")
    else:
        logger.warning("No CUDA — inference may be impractical for AIFS.")


def load_input_state() -> dict:
    """
    Prefer ``assets/target_date.txt`` → ``input_YYYYMMDDHH.npz``.
    ``prepare.py`` does not store ``date`` in the npz; T0 is taken from the filename.
    """
    if DATE_FILE.exists():
        target = DATE_FILE.read_text().splitlines()[0].strip()
        path = DATA_DIR / f"input_{target}.npz"
        if path.exists():
            latest_file = path
            logger.info("Loading %s (from target_date.txt)", latest_file.name)
        else:
            logger.warning("%s missing; using latest input_*.npz", path.name)
            files = sorted(glob.glob(str(DATA_DIR / "input_*.npz")))
            if not files:
                raise FileNotFoundError(f"No input_*.npz in {DATA_DIR}")
            latest_file = Path(files[-1])
            logger.info("Loading %s", latest_file.name)
    else:
        files = sorted(glob.glob(str(DATA_DIR / "input_*.npz")))
        if not files:
            raise FileNotFoundError(f"No input_*.npz in {DATA_DIR}")
        latest_file = Path(files[-1])
        logger.info("Loading %s", latest_file.name)

    data = np.load(latest_file, allow_pickle=True)
    fields = {k: data[k] for k in data.files if k != "date"}

    stem = latest_file.stem
    if not stem.startswith("input_"):
        raise ValueError(f"Expected input_YYYYMMDDHH.npz, got {latest_file.name}")
    date_str = stem[6:]
    date_obj = datetime.datetime.strptime(date_str, "%Y%m%d%H")

    return {"date": date_obj, "fields": fields}


def get_state_attr(state, key: str):
    if hasattr(state, key):
        return getattr(state, key)
    if isinstance(state, dict) and key in state:
        return state[key]
    return None


def create_regridder(device: torch.device):
    weights_csr, target_shape = ekr.db.find(SOURCE_GRID, TARGET_GRID, "linear")
    w = torch.sparse_csr_tensor(
        torch.from_numpy(weights_csr.indptr),
        torch.from_numpy(weights_csr.indices),
        torch.from_numpy(weights_csr.data),
        size=weights_csr.shape,
        device=device,
    )

    def regrid(arr_1d: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.asarray(arr_1d, dtype=np.float64)).to(device)
        out = w.matmul(t)
        return out.detach().cpu().numpy().astype(np.float32).reshape(target_shape)

    return regrid


def process_and_save(
    step: int,
    state,
    start_date: datetime.datetime,
    saver: Saver,
    regrid_func,
) -> None:
    valid_time = get_state_attr(state, "date")
    fields = get_state_attr(state, "fields")
    if valid_time is None:
        valid_time = start_date + datetime.timedelta(hours=step * 6)

    lead_time_hours = int((valid_time - start_date).total_seconds() / 3600)
    init_time_str = start_date.strftime("%Y%m%d%H")

    if not isinstance(fields, dict):
        try:
            fields = dict(fields)
        except Exception as exc:
            logger.error("Cannot read fields: %s", exc)
            return

    data_list = []
    channel_mapping = []

    for key in sorted(fields.keys()):
        val = fields[key]
        if hasattr(val, "cpu"):
            val = val.detach().cpu().numpy()
        val = np.asarray(val)
        if val.ndim > 1:
            val = val.squeeze()

        std_name = AIFS_VAR_MAP.get(key, key)

        if val.ndim == 1 and val.size == N320_POINTS:
            val_2d = regrid_func(val)
        elif val.ndim == 2 and val.shape == (LAT_RES, LON_RES):
            val_2d = val.astype(np.float32)
        else:
            logger.warning("Skip %s: unexpected shape %s", key, val.shape)
            continue

        data_list.append(val_2d)
        channel_mapping.append(std_name)

    if not data_list:
        logger.warning("Step %s: nothing to save", step)
        return

    combined_data = np.stack(data_list, axis=0)
    lats = np.linspace(90.0, -90.0, LAT_RES)
    lons = np.linspace(0.0, 360.0, LON_RES, endpoint=False)

    saver.save(
        data=combined_data,
        channel_mapping=channel_mapping,
        init_time_str=init_time_str,
        lead_time_hours=lead_time_hours,
        lat_values=lats,
        lon_values=lons,
    )
    logger.info("Saved step %s: +%sh (%s channels)", step, lead_time_hours, len(channel_mapping))


def main() -> None:
    parser = argparse.ArgumentParser(description="AIFS SimpleRunner + N320→0.25° NetCDF.")
    parser.add_argument(
        "--lead-time",
        type=int,
        default=DEFAULT_LEAD_HOURS,
        help="Forecast length in hours. Default: %(default)s",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint under assets/weights/aifs/. Default: %(default)s",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device for model and sparse regrid. Default: cuda",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic CUDA (slower).",
    )
    args = parser.parse_args()

    setup_environment(deterministic=args.deterministic)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA requested but not available.")
        sys.exit(1)

    torch_device = torch.device("cuda" if args.device == "cuda" else "cpu")

    try:
        input_state = load_input_state()
    except Exception as exc:
        logger.error("Failed to load input: %s", exc)
        sys.exit(1)

    start_date = input_state["date"]
    input_state["fields"] = {k: v.astype(np.float32) for k, v in input_state["fields"].items()}
    logger.info("Initialization time (T0): %s", start_date)

    checkpoint_path = WEIGHTS_DIR / args.checkpoint
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    logger.info("Loading model: %s", checkpoint_path.name)
    try:
        runner = SimpleRunner(checkpoint=str(checkpoint_path), device=args.device)
    except Exception as exc:
        logger.error("SimpleRunner failed: %s", exc)
        sys.exit(1)

    regrid_func = create_regridder(torch_device)
    saver = Saver(save_root=str(OUTPUT_DIR))

    logger.info("Running inference (lead_time=%s h)...", args.lead_time)
    try:
        forecast = runner.run(input_state=input_state, lead_time=args.lead_time)
        for step, state in enumerate(forecast, start=1):
            vt = get_state_attr(state, "date")
            logger.info("Step %s: valid %s", step, vt)
            process_and_save(step, state, start_date, saver, regrid_func)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info("AIFS inference completed.")
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

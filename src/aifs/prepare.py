import datetime
import logging
import shutil
import numpy as np
import earthkit.data as ekd
import earthkit.regrid as ekr
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("AIFS.Prepare")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "assets" / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "assets" / "data" / "processed_aifs"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# AIFS variable definitions
# ──────────────────────────────────────────────────────────────

# Dynamic surface variables (extracted for both T-6 and T0)
DYNAMIC_SFC_PARAMS = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]

# Time-invariant surface fields (extracted once from T0, then duplicated)
STATIC_SFC_PARAMS = ["lsm", "z", "slor", "sdor"]

# Soil variables
SOIL_PARAMS = ["vsw", "sot"]
SOIL_LEVELS = [1, 2]
SOIL_RENAME = {
    "vsw_1": "swvl1",
    "vsw_2": "swvl2",
    "sot_1": "stl1",
    "sot_2": "stl2",
}

# Upper-air variables (gh will be converted to z)
PRESSURE_PARAMS = ["gh", "t", "u", "v", "q", "w"]
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# N320 reduced Gaussian grid point count
N320_POINTS = 542080
SOURCE_GRID = {"grid": (0.25, 0.25)}
TARGET_GRID = {"grid": "N320"}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def regrid_to_n320(arr_2d: np.ndarray) -> np.ndarray:
    """Regrid a 2-D (lat × lon) 0.25° field to the 1-D N320 Gaussian grid."""
    result = ekr.interpolate(arr_2d, SOURCE_GRID, TARGET_GRID)
    return result.astype(np.float32)


def roll_longitude_before_n320(arr: np.ndarray) -> np.ndarray:
    """Longitude roll on the 0.25° lat–lon slice before ``regrid_to_n320`` (axis=1, half width)."""
    if arr.ndim != 2:
        return arr
    return np.roll(arr, -arr.shape[1] // 2, axis=1)


def read_grib_fields(filepath: Path, params: list, levelist: list = None) -> dict:
    """
    Open a GRIB file and return a dict {name: 2d_numpy_array}.

    Keys for surface fields:  '<param>'  (e.g. '10u', 'lsm')
    Keys for levelled fields: '<param>_<level>'  (e.g. 'gh_500', 'vsw_1')
    """
    logger.info(f"  Reading {filepath.name} ...")
    ds = ekd.from_source("file", str(filepath))
    ds = ds.sel(param=params)
    if levelist is not None:
        ds = ds.sel(levelist=levelist)

    result = {}
    for f in ds:
        p = f.metadata("param")
        lev = f.metadata("levelist", default=None)
        key = f"{p}_{lev}" if lev is not None else p
        result[key] = f.to_numpy()

    return result


# ──────────────────────────────────────────────────────────────
# Main processing
# ──────────────────────────────────────────────────────────────

def process_data(target_date_str: str):
    logger.info(f"Starting AIFS data preparation for: {target_date_str}")

    dt_t0 = datetime.datetime.strptime(target_date_str, "%Y%m%d%H")
    dt_t6 = dt_t0 - datetime.timedelta(hours=6)
    t0_str = dt_t0.strftime("%Y%m%d%H")
    t6_str = dt_t6.strftime("%Y%m%d%H")

    logger.info(f"  T-6 : {t6_str}")
    logger.info(f"  T0  : {t0_str}")

    # Verify all required files exist before processing
    required_files = [
        RAW_DATA_DIR / f"surface_{t6_str}_hres.grib",
        RAW_DATA_DIR / f"surface_{t0_str}_hres.grib",
        RAW_DATA_DIR / f"land_{t6_str}_hres.grib",
        RAW_DATA_DIR / f"land_{t0_str}_hres.grib",
        RAW_DATA_DIR / f"upper_{t6_str}_hres.grib",
        RAW_DATA_DIR / f"upper_{t0_str}_hres.grib",
    ]
    for p in required_files:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # Accumulate fields as lists: fields[name] = [arr_t6, arr_t0]
    fields_raw: dict[str, list] = {}

    # ── 1. Dynamic surface variables (T-6 and T0) ──────────────
    logger.info("Processing dynamic surface variables ...")
    for i, dt_str in enumerate([t6_str, t0_str]):
        sfc_file = RAW_DATA_DIR / f"surface_{dt_str}_hres.grib"
        data = read_grib_fields(sfc_file, DYNAMIC_SFC_PARAMS)
        for name, arr in data.items():
            fields_raw.setdefault(name, [None, None])
            fields_raw[name][i] = regrid_to_n320(roll_longitude_before_n320(arr))
            logger.info(f"    {name}  [{dt_str}]  → N320")

    # ── 2. Static surface variables (from T0, duplicated) ──────
    logger.info("Processing static surface variables ...")
    sfc_t0_file = RAW_DATA_DIR / f"surface_{t0_str}_hres.grib"
    static_data = read_grib_fields(sfc_t0_file, STATIC_SFC_PARAMS)
    for name, arr in static_data.items():
        arr_n320 = regrid_to_n320(roll_longitude_before_n320(arr))
        # Duplicate: identical value at T-6 and T0
        fields_raw[name] = [arr_n320, arr_n320]
        logger.info(f"    {name}  [static]  → N320 (duplicated)")

    # ── 3. Soil variables (T-6 and T0) ─────────────────────────
    logger.info("Processing soil variables ...")
    for i, dt_str in enumerate([t6_str, t0_str]):
        soil_file = RAW_DATA_DIR / f"land_{dt_str}_hres.grib"
        data = read_grib_fields(soil_file, SOIL_PARAMS, levelist=SOIL_LEVELS)
        for raw_name, arr in data.items():
            aifs_name = SOIL_RENAME.get(raw_name, raw_name)
            fields_raw.setdefault(aifs_name, [None, None])
            fields_raw[aifs_name][i] = regrid_to_n320(roll_longitude_before_n320(arr))
            logger.info(f"    {raw_name} → {aifs_name}  [{dt_str}]  → N320")

    # ── 4. Upper-air variables (T-6 and T0) ────────────────────
    logger.info("Processing upper-air variables ...")
    for i, dt_str in enumerate([t6_str, t0_str]):
        pl_file = RAW_DATA_DIR / f"upper_{dt_str}_hres.grib"
        data = read_grib_fields(pl_file, PRESSURE_PARAMS, levelist=PRESSURE_LEVELS)
        for raw_name, arr in data.items():
            # gh (geopotential height, m) → z (geopotential, m²/s²)
            if raw_name.startswith("gh_"):
                level = raw_name.split("_", 1)[1]
                aifs_name = f"z_{level}"
                arr = arr * 9.80665
            else:
                aifs_name = raw_name
            fields_raw.setdefault(aifs_name, [None, None])
            fields_raw[aifs_name][i] = regrid_to_n320(roll_longitude_before_n320(arr))
            logger.info(f"    {raw_name} → {aifs_name}  [{dt_str}]  → N320")

    # ── 5. Stack into (2, N320_POINTS) and validate ────────────
    logger.info("Stacking time steps and validating shapes ...")
    fields: dict[str, np.ndarray] = {}
    for name, val_list in fields_raw.items():
        if any(v is None for v in val_list):
            logger.warning(f"  Skipping '{name}': incomplete data (missing a time step)")
            continue
        arr = np.stack(val_list, axis=0)  # (2, N320_POINTS)
        if arr.shape != (2, N320_POINTS):
            logger.warning(
                f"  '{name}' has unexpected shape {arr.shape}; expected (2, {N320_POINTS})"
            )
        fields[name] = arr

    logger.info(f"Total fields prepared: {len(fields)}")
    for name, arr in sorted(fields.items()):
        logger.info(f"  {name:<12}  shape={arr.shape}  dtype={arr.dtype}")

    # ── 6. Save ─────────────────────────────────────────────────
    output_filename = f"input_{target_date_str}.npz"
    output_path = OUTPUT_DIR / output_filename
    tmp_path = Path("/tmp") / output_filename

    logger.info(f"Saving to {output_path} ...")
    np.savez_compressed(str(tmp_path), **fields)
    shutil.move(str(tmp_path), str(output_path))

    # Update bookkeeping file
    (OUTPUT_DIR / "latest_date.txt").write_text(target_date_str)

    logger.info(f"Done. Saved {len(fields)} fields to {output_path}")


if __name__ == "__main__":
    if not DATE_FILE.exists():
        logger.error(f"Date file not found: {DATE_FILE}")
        raise SystemExit(1)

    date_str = DATE_FILE.read_text().strip()
    process_data(date_str)

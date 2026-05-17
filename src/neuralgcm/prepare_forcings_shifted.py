import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import neuralgcm
from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("NeuralGCM.PrepareForcingsShifted")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "assets" / "data" / "raw"
OUT_DIR = BASE_DIR / "assets" / "data" / "processed_neuralgcm"
WEIGHTS_DIR = BASE_DIR / "assets" / "weights" / "neuralgcm"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(model_res: str) -> neuralgcm.PressureLevelModel:
    name_map = {
        "0.7": "models_v1_deterministic_0_7_deg.pkl",
        "1.4": "models_v1_deterministic_1_4_deg.pkl",
        "2.8": "models_v1_deterministic_2_8_deg.pkl",
    }
    p = WEIGHTS_DIR / name_map[model_res]
    with open(p, "rb") as f:
        ckpt = pickle.load(f)
    return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)


def standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if float(ds.longitude.min()) < 0:
        ds = ds.assign_coords(longitude=((ds.longitude + 360.0) % 360.0))
    ds = ds.sortby("longitude")
    ds = ds.sortby("latitude")
    return ds


def main(date_str: str, model_res: str):
    init_time = np.datetime64(pd.to_datetime(date_str, format="%Y%m%d%H"))

    in_path = RAW_DIR / f"neuralgcm_forcings_24h_{date_str}.nc"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing: {in_path} (run download_forcings_24h.py first)")

    ds = xr.open_dataset(in_path)
    ds = standardize_latlon(ds)

    # rename to NeuralGCM long names if needed (ERA5 single-levels uses sst/siconc sometimes)
    rename = {}
    if "sst" in ds:
        rename["sst"] = "sea_surface_temperature"
    if "siconc" in ds:
        rename["siconc"] = "sea_ice_cover"
    if rename:
        ds = ds.rename(rename)

    forcing_vars = ["sea_surface_temperature", "sea_ice_cover"]
    for v in forcing_vars:
        if v not in ds:
            raise ValueError(f"Missing forcing var {v} in {in_path.name}")

    # Official behavior: shift forcing vars by +24h (so t uses original t-24h)
    logger.info("Applying selective_temporal_shift(time_shift='24 hours') to SST/SIC")
    ds_shifted = xarray_utils.selective_temporal_shift(
        ds[forcing_vars],
        variables=forcing_vars,
        time_shift="24 hours",
        time_name="time",
    )

    # Take the init time slice (time=1)
    ds_t = ds_shifted.sel(time=[init_time], method="nearest")

    model = load_model(model_res)
    # Regrid to Gaussian
    src_grid = spherical_harmonic.Grid(
        latitude_nodes=ds_t.sizes["latitude"],
        longitude_nodes=ds_t.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(ds_t.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(ds_t.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        src_grid, model.data_coords.horizontal, skipna=True
    )
    ds_g = xarray_utils.regrid(ds_t, regridder)
    ds_g = xarray_utils.fill_nan_with_nearest(ds_g)
    
    """
    out_path = OUT_DIR / f"forcings_shifted_{date_str}_gaussian.nc"
    ds_g.to_netcdf(out_path)
    logger.info(f"Saved: {out_path}")
    """
    tmp_output = Path("/tmp") / f"forcings_shifted_{date_str}_gaussian.nc"
    ds_g.to_netcdf(tmp_output)
    logger.info(f"✅ Saved to temporary file: {tmp_output}")

    output_path = OUT_DIR / f"forcings_shifted_{date_str}_gaussian.nc"
    import shutil
    shutil.move(str(tmp_output), str(output_path))
    logger.info(f"✅ Moved to final destination: {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDDHH, e.g. 2023010112")
    ap.add_argument("--model-res", default="0.7", choices=["0.7", "1.4", "2.8"])
    args = ap.parse_args()
    main(args.date, args.model_res)
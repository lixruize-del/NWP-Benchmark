import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Stormer.EvalZ500")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "stormer"
RAW_DATA_DIR = BASE_DIR / "assets" / "data" / "raw"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"


def _normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    if "lat" in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords:
        rename_map["lon"] = "longitude"
    if "isobaricInhPa" in ds.coords:
        rename_map["isobaricInhPa"] = "level"
    if "pressure_level" in ds.coords:
        rename_map["pressure_level"] = "level"
    if "valid_time" in ds.coords:
        rename_map["valid_time"] = "time"
    if rename_map:
        ds = ds.rename(rename_map)

    if "longitude" in ds.coords:
        ds = ds.assign_coords(longitude=((ds.longitude + 360) % 360)).sortby("longitude")
    if "latitude" in ds.coords and ds.latitude[0] < ds.latitude[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))
    return ds


def latitude_weights(lat: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()


def wrmse(pred: xr.DataArray, gt: xr.DataArray) -> float:
    w = latitude_weights(gt.latitude)
    w, diff2 = xr.broadcast(w, (pred - gt) ** 2)
    return float(np.sqrt((diff2 * w).mean()))


def get_target_date() -> str:
    if DATE_FILE.exists():
        return DATE_FILE.read_text().strip()
    return "2023010112"


def main(date: str, lead: int):
    init_time = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:8]}T{date[8:10]}:00")
    valid_time = init_time + np.timedelta64(lead, "h")
    valid_date = str(valid_time).replace("-", "").replace("T", "")[:10]

    pred_path = OUTPUT_DIR / date / f"{date[:4]}-{date[4:8]}-{lead:02d}.nc"
    gt_path = RAW_DATA_DIR / f"upper_{valid_date}.nc"

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    ds_pred = _normalize_coords(xr.open_dataset(pred_path))
    ds_gt = _normalize_coords(xr.open_dataset(gt_path))

    if "z" not in ds_pred:
        raise KeyError("Prediction file has no variable 'z'")
    gt_var = "z" if "z" in ds_gt else "geopotential"
    if gt_var not in ds_gt:
        raise KeyError("GT file has no 'z'/'geopotential'")

    da_pred = ds_pred["z"].sel(level=500)
    if "time" in da_pred.dims:
        da_pred = da_pred.isel(time=0)

    da_gt = ds_gt[gt_var]
    if "time" in da_gt.dims:
        # no regrid of prediction; interpolate GT to prediction grid only
        try:
            da_gt = da_gt.sel(time=valid_time)
        except Exception:
            da_gt = da_gt.isel(time=0)
    da_gt = da_gt.sel(level=500)
    da_gt = da_gt.interp(latitude=da_pred.latitude, longitude=da_pred.longitude, method="linear")

    score = wrmse(da_pred, da_gt)
    logger.info("Z500 WRMSE at native Stormer grid (no pred regrid): %.3f", score)
    logger.info("Pred mean=%.3f | GT mean=%.3f", float(da_pred.mean()), float(da_gt.mean()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stormer Z500 WRMSE on native grid")
    parser.add_argument("--date", type=str, default=None, help="init time YYYYMMDDHH")
    parser.add_argument("--lead", type=int, default=6, help="lead hours")
    args = parser.parse_args()

    main(args.date or get_target_date(), args.lead)


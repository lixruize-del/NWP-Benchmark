import argparse
import logging
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr

import regridding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stormer.prepare")

DDEG_OUT = 1.40625
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "assets" / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "assets" / "data" / "processed_stormer"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    *[f"geopotential_{l}" for l in PRESSURE_LEVELS],
    *[f"u_component_of_wind_{l}" for l in PRESSURE_LEVELS],
    *[f"v_component_of_wind_{l}" for l in PRESSURE_LEVELS],
    *[f"temperature_{l}" for l in PRESSURE_LEVELS],
    *[f"specific_humidity_{l}" for l in PRESSURE_LEVELS],
]

SFC_MAP = {
    "2m_temperature": ["2m_temperature", "t2m"],
    "10m_u_component_of_wind": ["10m_u_component_of_wind", "u10"],
    "10m_v_component_of_wind": ["10m_v_component_of_wind", "v10"],
    "mean_sea_level_pressure": ["mean_sea_level_pressure", "msl"],
}
PL_MAP = {
    "geopotential": ["geopotential", "z"],
    "u_component_of_wind": ["u_component_of_wind", "u"],
    "v_component_of_wind": ["v_component_of_wind", "v"],
    "temperature": ["temperature", "t"],
    "specific_humidity": ["specific_humidity", "q"],
}


def get_target_grid(ddeg_out: float = DDEG_OUT):
    lat_start = -90 + ddeg_out / 2
    lat_stop = 90 - ddeg_out / 2
    n_lat = int(180 / ddeg_out)
    n_lon = int(360 / ddeg_out)
    new_lat = np.linspace(lat_start, lat_stop, num=n_lat, endpoint=True)
    new_lon = np.linspace(0, 360, num=n_lon, endpoint=False)
    return new_lat, new_lon


def create_regridder(ds_source: xr.Dataset, ddeg_out: float = DDEG_OUT):
    old_lon = ds_source.coords["lon"].values
    old_lat = ds_source.coords["lat"].values
    new_lat, new_lon = get_target_grid(ddeg_out)
    source_grid = regridding.Grid.from_degrees(lon=old_lon, lat=np.sort(old_lat))
    target_grid = regridding.Grid.from_degrees(lon=new_lon, lat=new_lat)
    return regridding.ConservativeRegridder(source_grid, target_grid), (new_lat, new_lon)


def regrid_dataset(ds: xr.Dataset, regridder) -> xr.Dataset:
    ds_out = regridder.regrid_dataset(ds)
    return ds_out.transpose(..., "lat", "lon")


def get_target_date():
    if DATE_FILE.exists():
        return DATE_FILE.read_text().strip()
    return "2023010112"


def rename_time_dim(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds


def ensure_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "lat" in ds.coords and ds["lat"][0] < ds["lat"][-1]:
        ds = ds.isel(lat=slice(None, None, -1))
    if "lon" in ds.coords:
        lon = ds["lon"].values
        if (lon < 0).any():
            lon2 = lon % 360
            idx = np.argsort(lon2)
            ds = ds.isel(lon=idx).assign_coords(lon=lon2[idx])
    return ds


def pick_existing_var(ds: xr.Dataset, candidates, where: str):
    for name in candidates:
        if name in ds.data_vars or name in ds.variables:
            return name
    raise KeyError(f"None of {candidates} found in {where}. Available: {list(ds.data_vars)}")


def coerce_2d(da: xr.DataArray) -> np.ndarray:
    da = da.astype(np.float32)
    extra_dims = [d for d in da.dims if d not in ("lat", "lon")]
    for d in extra_dims:
        if da.sizes[d] == 1:
            da = da.isel({d: 0})
        else:
            raise ValueError(f"Unexpected non-singleton extra dim {d} in {da.dims}")
    if not set(("lat", "lon")).issubset(da.dims):
        raise ValueError(f"lat/lon not found in dims={da.dims}")
    return da.values


def open_pair(date_str: str):
    surf = RAW_DATA_DIR / f"surface_{date_str}.nc"
    upper = RAW_DATA_DIR / f"upper_{date_str}.nc"
    if not surf.exists() or not upper.exists():
        raise FileNotFoundError(f"Missing raw pair for {date_str}: {surf}, {upper}")

    ds_s = ensure_lat_lon(rename_time_dim(xr.open_dataset(surf)))
    ds_u = ensure_lat_lon(rename_time_dim(xr.open_dataset(upper)))

    dt = pd.to_datetime(date_str, format="%Y%m%d%H")
    if "time" in ds_s.coords:
        times = pd.to_datetime(ds_s["time"].values)
        ds_s = ds_s.sel(time=[dt]) if (times == dt).any() else ds_s.isel(time=[-1])
    if "time" in ds_u.coords:
        times = pd.to_datetime(ds_u["time"].values)
        ds_u = ds_u.sel(time=[dt]) if (times == dt).any() else ds_u.isel(time=[-1])
    return ds_s, ds_u


def extract_fields(ds_s, ds_u, regridder_s, regridder_u):
    out = {}
    ds_s = regrid_dataset(ds_s, regridder_s)
    ds_u = regrid_dataset(ds_u, regridder_u)

    for v in ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"]:
        nc = pick_existing_var(ds_s, SFC_MAP[v], "surface nc")
        da = ds_s[nc]
        if "time" in da.dims:
            da = da.isel(time=0)
        out[v] = coerce_2d(da)

    level_dim = next((c for c in ["pressure_level", "level", "isobaricInhPa"] if c in ds_u.dims or c in ds_u.coords), None)
    if level_dim is None:
        raise ValueError(f"Cannot find pressure level dim in upper: dims={ds_u.dims}")

    for base in ["geopotential", "u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity"]:
        nc = pick_existing_var(ds_u, PL_MAP[base], "upper nc")
        da = ds_u[nc]
        if "time" in da.dims:
            da = da.isel(time=0)
        da = da.sel({level_dim: PRESSURE_LEVELS}).transpose(level_dim, "lat", "lon").astype(np.float32)
        for lvl in PRESSURE_LEVELS:
            out[f"{base}_{lvl}"] = coerce_2d(da.sel({level_dim: lvl}))

    missing = [v for v in VARIABLES if v not in out]
    if missing:
        raise RuntimeError(f"Missing variables after extraction: {missing}")
    return out


def write_stormer_h5(init_date: str, out_path: Path, lead_times=(6,)):
    init_dt = pd.to_datetime(init_date, format="%Y%m%d%H")
    tmp_path = Path("/tmp") / out_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds_s0, ds_u0 = open_pair(init_date)
    regridder_s, target_grid = create_regridder(ds_s0)
    regridder_u, _ = create_regridder(ds_u0)
    input_fields = extract_fields(ds_s0, ds_u0, regridder_s, regridder_u)
    ds_s0.close()
    ds_u0.close()

    with h5py.File(tmp_path, "w", libver="latest") as f:
        g_in = f.create_group("input")
        g_in.create_dataset("time", data=np.string_(init_date))
        new_lat, new_lon = target_grid
        g_in.create_dataset("lat", data=new_lat.astype(np.float32))
        g_in.create_dataset("lon", data=new_lon.astype(np.float32))
        for v in VARIABLES:
            g_in.create_dataset(v, data=np.asarray(input_fields[v], dtype=np.float32), compression=None)

        g_out = f.create_group("output")
        for lt in lead_times:
            tgt_dt = init_dt + pd.Timedelta(hours=int(lt))
            tgt_date = tgt_dt.strftime("%Y%m%d%H")
            surf_path = RAW_DATA_DIR / f"surface_{tgt_date}.nc"
            upper_path = RAW_DATA_DIR / f"upper_{tgt_date}.nc"
            if not (surf_path.exists() and upper_path.exists()):
                logger.warning(f"Skip output lead_time={lt}: missing raw files for {tgt_date}")
                continue
            ds_s, ds_u = open_pair(tgt_date)
            out_fields = extract_fields(ds_s, ds_u, regridder_s, regridder_u)
            ds_s.close()
            ds_u.close()

            g_lt = g_out.create_group(str(int(lt)))
            g_lt.create_dataset("time", data=np.string_(tgt_date))
            for v in VARIABLES:
                g_lt.create_dataset(v, data=np.asarray(out_fields[v], dtype=np.float32), compression=None)

    shutil.move(str(tmp_path), str(out_path))
    logger.info(f"Saved: {out_path}")


def main(date_str: str, lead_times):
    init_dt = pd.to_datetime(date_str, format="%Y%m%d%H")
    year = init_dt.year
    
    # Calculate the file sequence number: Assume one file every 6 hours starting from 00Z on January 1st of the current year.
    start_of_year = pd.Timestamp(f"{year}-01-01 00:00:00")
    hours_since_start = (init_dt - start_of_year).total_seconds() / 3600
    file_idx = int(hours_since_start // 6)  # 6h gap
    
    out_path = OUTPUT_DIR / f"{year}_{file_idx:04d}.h5"
    write_stormer_h5(date_str, out_path, lead_times=lead_times)
    (OUTPUT_DIR / "latest_date.txt").write_text(date_str)
    logger.info(f"Generated {out_path} for init_time={date_str}, idx={file_idx}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--lead_times", type=int, nargs="+", default=[6])
    args = p.parse_args()

    d = args.date or get_target_date()
    main(d, args.lead_times)

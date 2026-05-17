"""
Build GraphCast operational input (13 WeatherBench levels, HRES GRIB, no precip input).
Aligns with google-deepmind/graphcast TASK_13_PRECIP_OUT.
"""
import logging
import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

from graphcast import data_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GraphCast.PrepareOperational")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "assets" / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "assets" / "data" / "processed_graphcast_operational"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Official WeatherBench 13 (graphcast.graphcast.PRESSURE_LEVELS_WEATHERBENCH_13)
PRESSURE_LEVELS_13 = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

# IFS standard gravity per ECMWF documentation (geopotential <-> geopotential height)
G0 = 9.80665

STATIC_MAP = {"z": "geopotential_at_surface", "lsm": "land_sea_mask"}


def get_target_date():
    if DATE_FILE.exists():
        return DATE_FILE.read_text().strip()
    logger.warning("DATE_FILE not found, using default 2023010112")
    return "2023010112"


def ensure_lat_lon_order(ds):
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    if ds[lat_name][0] < ds[lat_name][-1]:
        ds = ds.isel({lat_name: slice(None, None, -1)})

    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lon = ds[lon_name].values
    if (lon < 0).any():
        lon_positive = lon % 360
        sort_idx = np.argsort(lon_positive)
        ds = ds.isel({lon_name: sort_idx})
        ds = ds.assign_coords({lon_name: lon_positive[sort_idx]})
    return ds


def _rename_coords(ds):
    m = {"latitude": "lat", "longitude": "lon", "isobaricInhPa": "level"}
    return ds.rename({k: v for k, v in m.items() if k in ds.dims or k in ds.coords})


def load_surface_grib(path: Path) -> xr.Dataset:
    """Load 2t, 10u, 10v, msl from one HRES surface GRIB (separate filters per shortName)."""
    arrays = {}
    for short_name in ("2t", "10u", "10v", "msl"):
        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"shortName": short_name}},
        )
        ds = _rename_coords(ds)
        v = list(ds.data_vars)[0]
        da = ds[v].reset_coords(drop=True)
        arrays[v] = da
    out = xr.Dataset(arrays)
    out = ensure_lat_lon_order(out)
    return out.astype(np.float32)


def load_upper_grib(path: Path) -> xr.Dataset:
    """Pressure-level HRES GRIB: gh (gpm), q, t, u, v, w on 13 levels."""
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
    )
    ds = _rename_coords(ds)
    ds = ensure_lat_lon_order(ds)
    if "level" not in ds.dims and "isobaricInhPa" in ds.dims:
        ds = ds.rename({"isobaricInhPa": "level"})

    # Geopotential (m^2 s^-2) from geopotential height (gpm) when cfgrib exposes gh as height
    gh = ds["gh"]
    units = str(gh.attrs.get("units", "")).lower()
    cf = str(gh.attrs.get("GRIB_cfName", "")).lower()
    if "m**-2" in units or "m2" in units or "m s" in units:
        z_m2s2 = gh.astype(np.float32)
    elif "gpm" in units or "geopotential_height" in cf or gh.name == "gh":
        z_m2s2 = (G0 * gh).astype(np.float32)
    else:
        logger.warning("Ambiguous gh units=%r cfName=%r; assuming gpm → × G0", units, cf)
        z_m2s2 = (G0 * gh).astype(np.float32)

    out = xr.Dataset(
        {
            "geopotential": z_m2s2,
            "specific_humidity": ds["q"].astype(np.float32),
            "temperature": ds["t"].astype(np.float32),
            "u_component_of_wind": ds["u"].astype(np.float32),
            "v_component_of_wind": ds["v"].astype(np.float32),
            "vertical_velocity": ds["w"].astype(np.float32),
        }
    )
    out = out.assign_coords(level=ds["level"].astype(np.float32))
    return out


def prepare_operational_input(target_date: str):
    dt = pd.to_datetime(target_date, format="%Y%m%d%H")
    times = [dt - pd.Timedelta(hours=6), dt]
    time_labels = [t.strftime("%Y%m%d%H") for t in times]

    static_file = RAW_DATA_DIR / "static.nc"
    if not static_file.exists():
        raise FileNotFoundError(f"Missing static file: {static_file}")

    for lab in time_labels:
        for suffix in ("surface", "upper"):
            f = RAW_DATA_DIR / f"{suffix}_{lab}_hres.grib"
            if not f.exists():
                raise FileNotFoundError(f"Missing HRES GRIB: {f}")

    surf_slices = []
    upper_slices = []
    for lab in time_labels:
        sf = RAW_DATA_DIR / f"surface_{lab}_hres.grib"
        uf = RAW_DATA_DIR / f"upper_{lab}_hres.grib"
        logger.info("Surface %s", sf)
        surf_slices.append(load_surface_grib(sf))
        logger.info("Upper %s", uf)
        upper_slices.append(load_upper_grib(uf))

    surf_t = xr.concat(surf_slices, dim="time")
    surf_t = surf_t.assign_coords(time=np.array([np.datetime64(t) for t in times]))

    upper_t = xr.concat(upper_slices, dim="time")
    upper_t = upper_t.assign_coords(time=np.array([np.datetime64(t) for t in times]))

    # WeatherBench level order
    upper_t = upper_t.sel(level=list(PRESSURE_LEVELS_13))
    upper_t = upper_t.transpose("time", "level", "lat", "lon")

    # Long names (TASK_13_PRECIP_OUT surface inputs — no precipitation)
    data_vars = {
        "2m_temperature": surf_t["t2m"].transpose("time", "lat", "lon").astype(np.float32),
        "mean_sea_level_pressure": surf_t["msl"].transpose("time", "lat", "lon").astype(np.float32),
        "10m_v_component_of_wind": surf_t["v10"].transpose("time", "lat", "lon").astype(np.float32),
        "10m_u_component_of_wind": surf_t["u10"].transpose("time", "lat", "lon").astype(np.float32),
    }
    for name in (
        "geopotential",
        "specific_humidity",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "vertical_velocity",
    ):
        data_vars[name] = upper_t[name].astype(np.float32)

    ds_static = xr.open_dataset(static_file)
    ds_static = ds_static.rename(
        {k: v for k, v in {"valid_time": "time", "latitude": "lat", "longitude": "lon"}.items() if k in ds_static.dims or k in ds_static.coords}
    )
    ds_static = ensure_lat_lon_order(ds_static)
    ds_static = ds_static.drop_vars(["expver", "number"], errors="ignore")

    for src, tgt in STATIC_MAP.items():
        da = ds_static[src].isel(time=0) if "time" in ds_static[src].dims else ds_static[src]
        arr = da.transpose("lat", "lon").astype(np.float32)
        if tgt == "land_sea_mask":
            arr = arr.fillna(0.0)
        data_vars[tgt] = arr

    ds_out = xr.Dataset(data_vars, attrs={"description": "GraphCast operational input (HRES)"})
    ds_out = ds_out.assign_coords(datetime=("time", np.array([np.datetime64(t) for t in times])))
    data_utils.add_derived_vars(ds_out)

    logger.info("Variables: %s", list(ds_out.data_vars))
    logger.info("Dims: %s", dict(ds_out.sizes))

    out_path = OUTPUT_DIR / f"input_{target_date}.nc"
    tmp = Path("/tmp") / f"input_operational_{target_date}.nc"
    ds_out.to_netcdf(tmp)
    shutil.move(str(tmp), str(out_path))
    logger.info("Saved %s", out_path)

    with open(OUTPUT_DIR / "latest_date.txt", "w", encoding="utf-8") as f:
        f.write(target_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="Target init YYYYMMDDHH")
    args = parser.parse_args()
    target = args.date if args.date else get_target_date()
    prepare_operational_input(target)

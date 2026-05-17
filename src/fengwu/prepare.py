import argparse
import logging
import datetime as dt
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.graphcast.prepare_operational import (
    ensure_lat_lon_order,
    load_surface_grib,
    load_upper_grib,
)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("Fengwu.Prepare")

# Paths (repo root layout)
RAW_DATA_DIR = BASE_DIR / "assets" / "data" / "raw"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"


def assets_data_dir_for_variant(variant: str) -> Path:
    """Prepared stacks: assets/data/fengwu_v1 or fengwu_v2."""
    d = BASE_DIR / "assets" / "data" / f"fengwu_{variant}"
    d.mkdir(parents=True, exist_ok=True)
    return d

# 13 pressure levels for FengWu, low to high (hPa)
PLEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def _find_coord_name(ds: xr.Dataset, candidates):
    for c in candidates:
        if c in ds.coords:
            return c
        if c in ds.dims:
            return c
    return None


def _standardize_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    """Normalize grid: latitude 90N→90S, longitude in [0, 360) (wrap/sort if needed)."""
    lat_name = _find_coord_name(ds, ["latitude", "lat"])
    lon_name = _find_coord_name(ds, ["longitude", "lon"])

    if lat_name is None or lon_name is None:
        raise ValueError(f"Cannot find lat/lon coordinates in dataset. coords={list(ds.coords)} dims={list(ds.dims)}")

    # lon: convert to [0, 360)
    lon = ds[lon_name]
    if float(lon.min()) < 0:
        ds = ds.assign_coords({lon_name: (lon % 360)})
        ds = ds.sortby(lon_name)

    # lat: ensure 90 -> -90
    lat = ds[lat_name]
    if float(lat[0]) < float(lat[-1]):  # increasing (-90 -> 90)
        ds = ds.sortby(lat_name, ascending=False)

    return ds


def _select_time(ds: xr.Dataset, when: np.datetime64) -> xr.Dataset:
    time_name = _find_coord_name(ds, ["time", "valid_time"])
    if time_name is None:
        raise ValueError(f"Cannot find time coordinate. coords={list(ds.coords)}")

    # Exact match; use ds.sel(..., method="nearest") if timestamps differ slightly.
    if when not in ds[time_name].values:
        raise KeyError(f"Requested time {when} not found. available={ds[time_name].values[:5]} ...")

    return ds.sel({time_name: when})


def _squeeze_singleton_dims_ds(ds: xr.Dataset) -> xr.Dataset:
    """Drop length-1 time / step / valid_time if present (typical for single-time GRIB)."""
    out = ds
    for d in ("time", "valid_time", "step"):
        if d in out.dims and out.sizes[d] == 1:
            out = out.squeeze(d, drop=True)
    return out


def _get_var(ds: xr.Dataset, varname: str) -> xr.DataArray:
    if varname not in ds.data_vars:
        raise KeyError(f"Variable '{varname}' not found. available={list(ds.data_vars)}")
    return ds[varname]


def _ensure_level_order(da: xr.DataArray, levels, level_coord_candidates=("pressure_level", "level", "isobaricInhPa")):
    lev_name = None
    for c in level_coord_candidates:
        if c in da.coords or c in da.dims:
            lev_name = c
            break
    if lev_name is None:
        raise ValueError(f"Cannot find pressure level coord in {da.name}. coords={list(da.coords)} dims={list(da.dims)}")

    # Subset and order levels for FengWu (50 … 1000 hPa).
    return da.sel({lev_name: levels})


def _to_numpy_2d(da: xr.DataArray) -> np.ndarray:
    """Project to float32 (721, 1440) with dimension order (lat, lon)."""
    ds = da.to_dataset(name="x")
    ds = _standardize_lat_lon(ds)

    lat_name = _find_coord_name(ds, ["latitude", "lat"])
    lon_name = _find_coord_name(ds, ["longitude", "lon"])

    x = ds["x"]

    if list(x.dims)[-2:] != [lat_name, lon_name]:
        x = x.transpose(..., lat_name, lon_name)

    arr = x.values.astype(np.float32)
    arr = np.squeeze(arr)  # drop length-1 time/step if any

    if arr.shape != (721, 1440):
        raise ValueError(f"Expected (721,1440), got {arr.shape} for {da.name}")

    return arr


def _to_numpy_3d_levels(da: xr.DataArray) -> np.ndarray:
    """Stack pressure levels to (13, 721, 1440) in PLEVELS order."""
    da = _ensure_level_order(da, PLEVELS)

    ds = da.to_dataset(name="x")
    ds = _standardize_lat_lon(ds)

    lat_name = _find_coord_name(ds, ["latitude", "lat"])
    lon_name = _find_coord_name(ds, ["longitude", "lon"])
    lev_name = _find_coord_name(ds, ["pressure_level", "level", "isobaricInhPa"])

    x = ds["x"].transpose(lev_name, lat_name, lon_name)
    arr = x.values.astype(np.float32)

    if arr.shape != (len(PLEVELS), 721, 1440):
        raise ValueError(f"Expected ({len(PLEVELS)},721,1440), got {arr.shape} for {da.name}")

    return arr


def build_fengwu_frame_from_datasets(ds_sfc_t: xr.Dataset, ds_up_t: xr.Dataset) -> np.ndarray:
    """One FengWu timestep: shape (69, 721, 1440). Surface u10,v10,t2m,msl; upper z,q,u,v,t × 13 levels."""
    ds_sfc_t = _squeeze_singleton_dims_ds(ds_sfc_t)
    ds_up_t = _squeeze_singleton_dims_ds(ds_up_t)

    u10 = _to_numpy_2d(_get_var(ds_sfc_t, "u10"))
    v10 = _to_numpy_2d(_get_var(ds_sfc_t, "v10"))
    t2m = _to_numpy_2d(_get_var(ds_sfc_t, "t2m"))
    msl = _to_numpy_2d(_get_var(ds_sfc_t, "msl"))

    z = _to_numpy_3d_levels(_get_var(ds_up_t, "z"))
    q = _to_numpy_3d_levels(_get_var(ds_up_t, "q"))
    u = _to_numpy_3d_levels(_get_var(ds_up_t, "u"))
    v = _to_numpy_3d_levels(_get_var(ds_up_t, "v"))
    t = _to_numpy_3d_levels(_get_var(ds_up_t, "t"))

    feats = [u10, v10, t2m, msl]
    feats += [z[i] for i in range(len(PLEVELS))]
    feats += [q[i] for i in range(len(PLEVELS))]
    feats += [u[i] for i in range(len(PLEVELS))]
    feats += [v[i] for i in range(len(PLEVELS))]
    feats += [t[i] for i in range(len(PLEVELS))]

    out = np.stack(feats, axis=0).astype(np.float32)
    if out.shape != (69, 721, 1440):
        raise ValueError(f"Expected (69,721,1440), got {out.shape}")

    return out


def _hres_surface_to_fengwu(ds: xr.Dataset) -> xr.Dataset:
    """Map cfgrib surface short names to FengWu names (u10, v10, t2m, msl)."""
    ds = ensure_lat_lon_order(ds)
    rename = {}
    if "u10" not in ds.data_vars and "10u" in ds.data_vars:
        rename["10u"] = "u10"
    if "v10" not in ds.data_vars and "10v" in ds.data_vars:
        rename["10v"] = "v10"
    if "t2m" not in ds.data_vars and "2t" in ds.data_vars:
        rename["2t"] = "t2m"
    if rename:
        ds = ds.rename(rename)
    return ds


def _hres_upper_to_fengwu(ds: xr.Dataset) -> xr.Dataset:
    """Rename GraphCast HRES upper-air fields to FengWu z, q, u, v, t (level retained)."""
    return xr.Dataset(
        {
            "z": ds["geopotential"],
            "q": ds["specific_humidity"],
            "u": ds["u_component_of_wind"],
            "v": ds["v_component_of_wind"],
            "t": ds["temperature"],
        }
    )


def build_fengwu_frame(surface_nc: Path, upper_nc: Path, when: np.datetime64) -> np.ndarray:
    """Single timestep from ERA5-style NetCDF at `when`. Shape (69, 721, 1440)."""
    ds_sfc = xr.open_dataset(surface_nc)
    ds_up = xr.open_dataset(upper_nc)

    try:
        ds_sfc_t = _select_time(ds_sfc, when)
        ds_up_t = _select_time(ds_up, when)
        return build_fengwu_frame_from_datasets(ds_sfc_t, ds_up_t)
    finally:
        ds_sfc.close()
        ds_up.close()


def _save_inputs(input1: np.ndarray, input2: np.ndarray, out_dir: Path) -> None:
    tmp_output_1 = Path("/tmp") / "input1.npy"
    tmp_output_2 = Path("/tmp") / "input2.npy"
    np.save(tmp_output_1, input1)
    np.save(tmp_output_2, input2)

    output_path_1 = out_dir / "input1.npy"
    output_path_2 = out_dir / "input2.npy"
    shutil.move(str(tmp_output_1), str(output_path_1))
    shutil.move(str(tmp_output_2), str(output_path_2))
    logger.info(f"Saved: {output_path_1}")
    logger.info(f"Saved: {output_path_2}")


def process_fengwu_data(target_date_str: str, source: str = "era5", variant: str = "v1"):
    """
    target_date_str: init time as '2023010112' or ISO datetime string.
    source: 'era5' NetCDF pairs or 'hres' GRIB (same layout as downloader_ecmwf_hres).
    variant: 'v1' or 'v2'; writes to assets/data/fengwu_{variant}/.
    """
    out_dir = assets_data_dir_for_variant(variant)
    logger.info(f"FengWu prepare: {target_date_str} (source={source}, variant={variant}, dir={out_dir})")

    try:
        t0 = dt.datetime.strptime(target_date_str, "%Y%m%d%H")
    except ValueError:
        t0 = dt.datetime.fromisoformat(target_date_str)

    t1 = t0 - dt.timedelta(hours=6)

    if source == "hres":
        t6_str = t1.strftime("%Y%m%d%H")
        t0_str = t0.strftime("%Y%m%d%H")
        required = []
        for lab in (t6_str, t0_str):
            required.append(RAW_DATA_DIR / f"surface_{lab}_hres.grib")
            required.append(RAW_DATA_DIR / f"upper_{lab}_hres.grib")
        for p in required:
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing HRES GRIB; run src/common/downloader_ecmwf_hres.py first.\nMissing: {p}"
                )

        logger.info(f"Build input1 (HRES t0-6h): {t6_str}")
        ds_sfc6 = _hres_surface_to_fengwu(load_surface_grib(RAW_DATA_DIR / f"surface_{t6_str}_hres.grib"))
        ds_up6 = _hres_upper_to_fengwu(load_upper_grib(RAW_DATA_DIR / f"upper_{t6_str}_hres.grib"))
        input1 = build_fengwu_frame_from_datasets(ds_sfc6, ds_up6)

        logger.info(f"Build input2 (HRES t0): {t0_str}")
        ds_sfc0 = _hres_surface_to_fengwu(load_surface_grib(RAW_DATA_DIR / f"surface_{t0_str}_hres.grib"))
        ds_up0 = _hres_upper_to_fengwu(load_upper_grib(RAW_DATA_DIR / f"upper_{t0_str}_hres.grib"))
        input2 = build_fengwu_frame_from_datasets(ds_sfc0, ds_up0)
    else:
        date0 = t0.strftime("%Y%m%d%H")
        surface_nc = RAW_DATA_DIR / f"surface_{date0}.nc"
        upper_nc = RAW_DATA_DIR / f"upper_{date0}.nc"

        if not surface_nc.exists() or not upper_nc.exists():
            raise FileNotFoundError(
                f"Missing NetCDF inputs; run the ERA5 downloader first.\n"
                f"Expected:\n  {surface_nc}\n  {upper_nc}"
            )

        when1 = np.datetime64(t1)
        when0 = np.datetime64(t0)

        logger.info(f"Build input1 (t0-6h): {t1.isoformat(sep=' ')}")
        input1 = build_fengwu_frame(surface_nc, upper_nc, when1)

        logger.info(f"Build input2 (t0): {t0.isoformat(sep=' ')}")
        input2 = build_fengwu_frame(surface_nc, upper_nc, when0)

    _save_inputs(input1, input2, out_dir)

    with open(out_dir / "latest_date.txt", "w") as f:
        f.write(target_date_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FengWu data preparation (ERA5 NetCDF or IFS HRES GRIB).")
    parser.add_argument(
        "--source",
        choices=["era5", "hres"],
        default="era5",
        help="era5: surface_*/upper_* .nc; hres: *_hres.grib (ECMWF Open Data as in downloader_ecmwf_hres).",
    )
    parser.add_argument(
        "--variant",
        choices=["v1", "v2"],
        default=None,
        help="Output under assets/data/fengwu_{variant}. Default: v1 for era5, v2 for hres.",
    )
    args = parser.parse_args()
    variant = args.variant or ("v1" if args.source == "era5" else "v2")

    if not DATE_FILE.exists():
        logger.error("Missing assets/target_date.txt")
        raise SystemExit(1)

    target_date = DATE_FILE.read_text().strip()
    try:
        process_fengwu_data(target_date, source=args.source, variant=variant)
    except Exception as e:
        logger.error(f"Prepare failed: {e}")
        raise SystemExit(1)

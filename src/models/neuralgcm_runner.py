"""NeuralGCM runner aligned with official data-preparation requirements."""

from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import neuralgcm
import numpy as np
import pandas as pd
import xarray as xr
from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils

from src.common.data_reader import DEFAULT_ERA5_NPY_ROOT, Era5NpyLayout, load_npy_2d

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_ROOT = Path(
    os.environ.get("NWP_WEIGHTS_ROOT", "/ecmwf-era5-datasets/nwp_bench/assets/weights")
)
DEFAULT_MODEL_FILE = "models_v1_deterministic_0_7_deg.pkl"

# Official NeuralGCM variable names (docs/API) -> local ERA5 NPY short names.
_LONG_TO_SHORT = {
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "specific_cloud_ice_water_content": "ciwc",
    "specific_cloud_liquid_water_content": "clwc",
    "sea_surface_temperature": "sst",
    "sea_ice_cover": "siconc",
}
_OPTIONAL_UPPER_INPUTS = {
    "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content",
}
_OPTIONAL_SURFACE_INPUTS = {"sea_ice_cover"}

_MODEL_CACHE: Dict[str, neuralgcm.PressureLevelModel] = {}


def _strict_inputs() -> bool:
    return os.environ.get("NWP_NEURALGCM_STRICT_INPUTS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _forcing_shift_hours() -> int:
    raw = os.environ.get("NWP_NEURALGCM_FORCING_SHIFT_HOURS", "24").strip()
    try:
        val = int(raw)
    except ValueError:
        logger.warning("Invalid NWP_NEURALGCM_FORCING_SHIFT_HOURS=%r; fallback to 24", raw)
        val = 24
    return max(0, val)


def _standardize_latlon_coords(ds: xr.Dataset) -> xr.Dataset:
    if "longitude" in ds.coords:
        lon = ds["longitude"]
        if float(lon.min()) < 0.0:
            ds = ds.assign_coords(longitude=((lon + 360.0) % 360.0))
        ds = ds.sortby("longitude")
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    return ds


def _load_model(weights_root: Path) -> neuralgcm.PressureLevelModel:
    weights_path = weights_root / "neuralgcm" / DEFAULT_MODEL_FILE
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing NeuralGCM checkpoint: {weights_path}")
    with open(weights_path, "rb") as f:
        ckpt = pickle.load(f)
    return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)


def _load_model_cached(weights_root: Path) -> neuralgcm.PressureLevelModel:
    weights_path = (weights_root / "neuralgcm" / DEFAULT_MODEL_FILE).resolve()
    key = str(weights_path)
    m = _MODEL_CACHE.get(key)
    if m is not None:
        return m
    m = _load_model(weights_root)
    _MODEL_CACHE[key] = m
    return m


def _model_levels(model: neuralgcm.PressureLevelModel) -> np.ndarray:
    return np.asarray(model.data_coords.vertical.centers, dtype=np.int32)


def _model_output_vars(model: neuralgcm.PressureLevelModel) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for long_name in model.input_variables:
        short = _LONG_TO_SHORT.get(long_name)
        if short is None:
            logger.warning("Skip unsupported NeuralGCM output variable: %s", long_name)
            continue
        out.append((short, long_name))
    if not out:
        raise RuntimeError("No supported NeuralGCM output variables were found.")
    return out


def _load_pressure_stack(
    layout: Era5NpyLayout,
    t: datetime,
    *,
    short_name: str,
    long_name: str,
    levels: np.ndarray,
    strict: bool,
    lat_desc: np.ndarray,
    lon: np.ndarray,
) -> xr.DataArray:
    slabs: List[np.ndarray] = []
    for lev in levels:
        path = layout.pressure_path(t, short_name, float(lev))
        if path.exists():
            slabs.append(load_npy_2d(path, flip_north_south=False).astype(np.float32))
            continue
        if long_name in _OPTIONAL_UPPER_INPUTS and not strict:
            logger.warning(
                "NeuralGCM optional input missing: %s (init=%s, lev=%s); using zeros.",
                path,
                t.strftime("%Y%m%d%H"),
                int(lev),
            )
            slabs.append(np.zeros((721, 1440), dtype=np.float32))
            continue
        raise FileNotFoundError(str(path))

    return xr.DataArray(
        np.stack(slabs, axis=0),
        dims=["level", "latitude", "longitude"],
        coords={"level": levels, "latitude": lat_desc, "longitude": lon},
    )


def _load_surface_field(
    layout: Era5NpyLayout,
    t: datetime,
    *,
    short_name: str,
    long_name: str,
    strict: bool,
    lat_desc: np.ndarray,
    lon: np.ndarray,
) -> xr.DataArray:
    path = layout.single_path(t, short_name)
    if path.exists():
        arr = load_npy_2d(path, flip_north_south=False).astype(np.float32)
    elif long_name in _OPTIONAL_SURFACE_INPUTS and not strict:
        logger.warning(
            "NeuralGCM optional forcing missing: %s (at %s); using zeros.",
            path,
            t.strftime("%Y%m%d%H"),
        )
        arr = np.zeros((721, 1440), dtype=np.float32)
    else:
        raise FileNotFoundError(str(path))
    return xr.DataArray(arr, dims=["latitude", "longitude"], coords={"latitude": lat_desc, "longitude": lon})


def _build_regridded_init_dataset(
    init_time: datetime,
    era5_root: Path,
    model: neuralgcm.PressureLevelModel,
) -> xr.Dataset:
    """
    Build one-time xarray dataset for model.encode()/forcings_from_xarray().

    Official alignment:
    - Uses `model.input_variables` + `model.forcing_variables` names.
    - Uses model pressure levels (`model.data_coords.vertical.centers`).
    - Applies forcing backward shift by default (24h), configurable by env.
    - Regrids to model native Gaussian grid with ConservativeRegridder(skipna=True),
      then fills NaNs with nearest.
    """
    layout = Era5NpyLayout(era5_root)
    strict = _strict_inputs()
    levels = _model_levels(model)
    lat_desc = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon = np.linspace(0.0, 360.0, 1440, endpoint=False, dtype=np.float32)

    ds_vars: Dict[str, xr.DataArray] = {}

    for long_name in model.input_variables:
        short = _LONG_TO_SHORT.get(long_name)
        if short is None:
            raise KeyError(f"Unsupported NeuralGCM input variable: {long_name}")
        ds_vars[long_name] = _load_pressure_stack(
            layout,
            init_time,
            short_name=short,
            long_name=long_name,
            levels=levels,
            strict=strict,
            lat_desc=lat_desc,
            lon=lon,
        )

    shift_hours = _forcing_shift_hours()
    forcing_src_time = init_time - timedelta(hours=shift_hours)
    for long_name in model.forcing_variables:
        short = _LONG_TO_SHORT.get(long_name)
        if short is None:
            raise KeyError(f"Unsupported NeuralGCM forcing variable: {long_name}")
        ds_vars[long_name] = _load_surface_field(
            layout,
            forcing_src_time,
            short_name=short,
            long_name=long_name,
            strict=strict,
            lat_desc=lat_desc,
            lon=lon,
        )

    ds_full = xr.Dataset(ds_vars).expand_dims(time=[np.datetime64(pd.Timestamp(init_time))])
    ds_full = _standardize_latlon_coords(ds_full)
    if "longitude" in ds_full.dims and "latitude" in ds_full.dims:
        ds_full = ds_full.transpose(..., "longitude", "latitude")

    source_grid = spherical_harmonic.Grid(
        latitude_nodes=ds_full.sizes["latitude"],
        longitude_nodes=ds_full.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(ds_full.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(ds_full.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        source_grid,
        model.data_coords.horizontal,
        skipna=True,
    )
    ds_regridded = xarray_utils.regrid(ds_full, regridder)
    return xarray_utils.fill_nan_with_nearest(ds_regridded)


def _channel_names_for_model(model: neuralgcm.PressureLevelModel) -> List[str]:
    levels = _model_levels(model)
    names: List[str] = []
    for short, _long_name in _model_output_vars(model):
        for lev in levels:
            names.append(f"{short}_{int(lev)}")
    return names


def neuralgcm_channel_names(*, weights_root: Path | None = None) -> List[str]:
    model = _load_model_cached(weights_root or DEFAULT_WEIGHTS_ROOT)
    return _channel_names_for_model(model)


def _stack_from_decoded(ds_native: xr.Dataset, model: neuralgcm.PressureLevelModel) -> np.ndarray:
    levels = _model_levels(model)
    out: List[np.ndarray] = []
    for _short, long_name in _model_output_vars(model):
        da = ds_native[long_name]
        for lev in levels:
            field = (
                da.sel(level=int(lev), method="nearest")
                .squeeze()
                .transpose("latitude", "longitude")
                .values.astype(np.float32)
            )
            # Benchmark convention: north -> south.
            out.append(np.ascontiguousarray(field[::-1, :]))
    return np.stack(out, axis=0).astype(np.float32)


def _run_with_model(
    model: neuralgcm.PressureLevelModel,
    init_time: datetime,
    lead_times_hours: List[int],
    *,
    era5_root: Path,
) -> Dict[int, np.ndarray]:
    ds_input = _build_regridded_init_dataset(init_time, era5_root, model)
    ds_init = ds_input.sel(time=np.datetime64(pd.Timestamp(init_time)), method="nearest")

    rng_key = jax.random.key(42)
    inputs = model.inputs_from_xarray(ds_init)
    input_forcings = model.forcings_from_xarray(ds_init)
    encoded_state = model.encode(inputs, input_forcings, rng_key)

    leads = sorted({int(h) for h in lead_times_hours})
    if not leads:
        return {}

    dt_hours = float(model.timestep / np.timedelta64(1, "h"))
    steps_by_lead: Dict[int, int] = {}
    for lead in leads:
        n_steps_f = lead / dt_hours
        n_steps = int(round(n_steps_f))
        if n_steps <= 0 or not np.isclose(n_steps_f, n_steps):
            raise ValueError(f"Invalid lead {lead}h for model dt={dt_hours}h")
        steps_by_lead[lead] = n_steps

    max_steps = max(steps_by_lead.values())
    all_forcings = model.forcings_from_xarray(
        ds_input.sel(time=[np.datetime64(pd.Timestamp(init_time))], method="nearest")
    )
    _, predictions = model.unroll(
        encoded_state,
        all_forcings,
        steps=max_steps,
        timedelta=model.timestep,
        start_with_input=False,
    )

    out_times = np.array(
        [pd.Timestamp(init_time) + pd.Timedelta(hours=(i + 1) * dt_hours) for i in range(max_steps)],
        dtype="datetime64[ns]",
    )
    ds_all = model.data_to_xarray(predictions, times=out_times)

    out: Dict[int, np.ndarray] = {}
    for lead in leads:
        step_idx = steps_by_lead[lead] - 1
        ds_last = ds_all.isel(time=step_idx, drop=False)
        out[lead] = _stack_from_decoded(ds_last, model)
    return out


def run_neuralgcm_forecast(
    init_time: datetime,
    lead_times_hours: List[int],
    *,
    era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
    weights_root: Path | None = None,
) -> Dict[int, np.ndarray]:
    model = _load_model_cached(weights_root or DEFAULT_WEIGHTS_ROOT)
    return _run_with_model(model, init_time, lead_times_hours, era5_root=era5_root)


class NeuralGCMForecastRunnerV2:
    """Class-style NeuralGCM runner (checkpoint/model cached per process)."""

    def __init__(
        self,
        *,
        era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
        weights_root: Path | None = None,
    ) -> None:
        self.era5_root = Path(era5_root)
        self.weights_root = Path(weights_root) if weights_root is not None else DEFAULT_WEIGHTS_ROOT
        self._model: neuralgcm.PressureLevelModel | None = None

    @property
    def model(self) -> neuralgcm.PressureLevelModel:
        if self._model is None:
            self._model = _load_model_cached(self.weights_root)
        return self._model

    def run(self, init_time: datetime, lead_times_hours: List[int]) -> Dict[int, np.ndarray]:
        return _run_with_model(self.model, init_time, lead_times_hours, era5_root=self.era5_root)


def interpolate_gt_to_neuralgcm_native(
    valid_time: datetime,
    *,
    era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
    target_hw: tuple[int, int] = (256, 512),
    model: neuralgcm.PressureLevelModel | None = None,
) -> np.ndarray:
    """
    Load ERA5 NPY ground truth using the same output variable set/order as model outputs,
    then conservatively regrid to model native Gaussian grid.
    """
    model = model or _load_model_cached(DEFAULT_WEIGHTS_ROOT)
    layout = Era5NpyLayout(era5_root)
    strict = _strict_inputs()
    levels = _model_levels(model)
    vars_out = _model_output_vars(model)

    slabs: List[np.ndarray] = []
    for short, long_name in vars_out:
        for lev in levels:
            path = layout.pressure_path(valid_time, short, float(lev))
            if path.exists():
                slabs.append(load_npy_2d(path, flip_north_south=False).astype(np.float32))
                continue
            if long_name in _OPTIONAL_UPPER_INPUTS and not strict:
                logger.warning(
                    "NeuralGCM optional GT variable missing: %s (time=%s, lev=%s); using zeros.",
                    path,
                    valid_time.strftime("%Y%m%d%H"),
                    int(lev),
                )
                slabs.append(np.zeros((721, 1440), dtype=np.float32))
                continue
            raise FileNotFoundError(str(path))

    src = np.stack(slabs, axis=0).astype(np.float32)
    src_lat_desc = np.linspace(90.0, -90.0, src.shape[1], dtype=np.float32)
    src_lon = np.linspace(0.0, 360.0, src.shape[2], endpoint=False, dtype=np.float32)

    ds = xr.Dataset(
        {
            "x": xr.DataArray(
                src.transpose(1, 2, 0),
                dims=["latitude", "longitude", "channel"],
                coords={
                    "latitude": src_lat_desc,
                    "longitude": src_lon,
                    "channel": np.arange(src.shape[0]),
                },
            )
        }
    )
    ds = _standardize_latlon_coords(ds)
    source_grid = spherical_harmonic.Grid(
        latitude_nodes=ds.sizes["latitude"],
        longitude_nodes=ds.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(ds.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(ds.longitude),
    )

    target_grid = model.data_coords.horizontal
    if (target_grid.latitude_nodes, target_grid.longitude_nodes) != target_hw:
        logger.warning(
            "Requested target_hw=%s differs from model native=(%s,%s); using model native grid.",
            target_hw,
            target_grid.latitude_nodes,
            target_grid.longitude_nodes,
        )

    regridder = horizontal_interpolation.ConservativeRegridder(
        source_grid,
        target_grid,
        skipna=True,
    )
    ds_rg = xarray_utils.regrid(ds, regridder)
    ds_rg = xarray_utils.fill_nan_with_nearest(ds_rg)
    out = ds_rg["x"].transpose("channel", "latitude", "longitude").values.astype(np.float32)
    return np.ascontiguousarray(out[:, ::-1, :])

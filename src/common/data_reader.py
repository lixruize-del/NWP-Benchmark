"""
ERA5 snapshot loader for local NPY archives (e.g. era5_np.25).

Layout (confirmed on disk)
---------------------------
Pressure-level fields::

    {root}/{year}/{YYYY-MM-DD}/{HH:MM:SS}-{var}-{level}.npy

Surface fields::

    {root}/single/{year}/{YYYY-MM-DD}/{HH:MM:SS}-{var}.npy

Each array is float32 of shape (721, 1440).

Latitude orientation depends on the provider.  By default we reverse rows
(``flip_north_south=True``) to mirror historical baseline behavior used by
the existing NPY-based runners. Set ``NWP_ERA5_FLIP=0`` to disable flipping.

Precipitation (``tp`` / ``tp6h``)
---------------------------------
The archive provides hourly ``tp`` (depth of water per ERA5 step, metres) and
``tp6h`` (6-hour accumulated depth, metres).  Models disagree on units and
which file to use:

- **GraphCast** (and this repo's ``_build_np25_input``): ``total_precipitation_6hr``
  is read from ``tp6h`` \*.npy **in metres** (ERA5-style), matching the checkpoint pipeline.
- **FuXi** (https://github.com/tpys/FuXi ``make_era5_input.py``): the ``TP``
  channel is 6h sum then ``×1000`` → **millimetres**, ``clip(0, 1000)`` — see
  ``_load_fuxi_tp_like_official_repo``.
- **Pangu / FengWu / Aurora (here)**: no precipitation input channel in the current runners.
- **Stormer**: variable list has no ERA5 precip in ``load_stormer_stack``.

Callers that need 6h depth in metres should use ``load_era5_tp6h_depth_m``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Sequence

import numpy as np
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)
TARGET_025_SHAPE = (721, 1440)

# ---------------------------------------------------------------------------
# Default data root (override via constructor / function argument)
# ---------------------------------------------------------------------------

DEFAULT_ERA5_NPY_ROOT = Path("/ecmwf-era5-datasets/era5_np.25")

# Set NWP_ERA5_FLIP=0 if your NPY latitude axis is already north-to-south (90→-90).
_DEFAULT_FLIP_NS = os.environ.get("NWP_ERA5_FLIP", "1").strip().lower() not in ("0", "false", "no")

# Pangu / FengWu shared 13 levels (hPa), low → high in FengWu channel order
PANGU_PRESSURE_LEVELS: List[int] = [
    1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50,
]
FENGWU_PRESSURE_LEVELS: List[int] = [
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
]

# FuXi (from src/fuxi/prepare.py)
FUXI_PL_NAMES: List[str] = ["z", "t", "u", "v", "r"]
FUXI_SFC_NAMES: List[str] = ["t2m", "u10", "v10", "msl", "tp"]
FUXI_LEVELS: List[int] = [
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
]

# ---------------------------------------------------------------------------
# NeuralGCM (official ERA5 pressure-level inputs)
# ---------------------------------------------------------------------------
#
# NeuralGCM data preparation requires 37 pressure levels and these atmospheric
# variables on pressure levels:
#   u_component_of_wind, v_component_of_wind, geopotential, temperature,
#   specific_humidity, specific_cloud_ice_water_content,
#   specific_cloud_liquid_water_content
#
# Surface forcings:
#   sea_surface_temperature, sea_ice_cover
#
# Refs:
# - https://neuralgcm.readthedocs.io/en/latest/data_preparation.html
# - https://neuralgcm.readthedocs.io/en/latest/inference_demo.html
NEURALGCM_PRESSURE_LEVELS_ERA5: List[int] = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875,
    900, 925, 950, 975, 1000,
]
NEURALGCM_UPPER_SHORTS: List[str] = ["z", "t", "u", "v", "q", "ciwc", "clwc"]
NEURALGCM_SURFACE_FORCING_SHORTS: List[str] = ["sst", "siconc"]


@dataclass
class Era5NpyLayout:
    """Paths for one valid time."""

    root: Path

    def pressure_path(self, dt: datetime, var: str, level_hpa: float) -> Path:
        return (
            self.root
            / str(dt.year)
            / dt.strftime("%Y-%m-%d")
            / f"{dt.strftime('%H:%M:%S')}-{var}-{float(level_hpa)}.npy"
        )

    def single_path(self, dt: datetime, var: str) -> Path:
        return (
            self.root
            / "single"
            / str(dt.year)
            / dt.strftime("%Y-%m-%d")
            / f"{dt.strftime('%H:%M:%S')}-{var}.npy"
        )


def flip_lat_ns(arr: np.ndarray) -> np.ndarray:
    """Reverse the latitude dimension (axis -2 for … x lat x lon)."""
    return np.ascontiguousarray(np.flip(arr, axis=-2))


def load_npy_2d(
    path: Path,
    *,
    flip_north_south: bool = True,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    x = np.load(path).astype(np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D field in {path}, got shape {x.shape}")
    return flip_lat_ns(x) if flip_north_south else x


def _ensure_025_shape(
    arr_2d: np.ndarray,
    *,
    field_name: str,
) -> np.ndarray:
    """Keep 0.25 shape; only resize known (2001, 4000) side-channel fields."""
    x = np.asarray(arr_2d, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D field for {field_name}, got shape {x.shape}")
    if x.shape == TARGET_025_SHAPE:
        return x
    if x.shape != (2001, 4000):
        raise ValueError(
            f"Unexpected shape for {field_name}: {x.shape}. "
            f"Expected {TARGET_025_SHAPE} or (2001, 4000)."
        )
    factors = (
        TARGET_025_SHAPE[0] / float(x.shape[0]),
        TARGET_025_SHAPE[1] / float(x.shape[1]),
    )
    logger.warning(
        "Field %s shape %s != %s; resizing with scipy.ndimage.zoom(order=1).",
        field_name,
        x.shape,
        TARGET_025_SHAPE,
    )
    out = zoom(x, factors, order=1)
    return np.asarray(out, dtype=np.float32, order="C")


def load_era5_tp6h_depth_m(
    layout: Era5NpyLayout,
    t: datetime,
    *,
    flip_north_south: bool = True,
) -> np.ndarray:
    """
    Six-hour accumulated total precipitation as **depth of water in metres** (ERA5 convention).

    Prefer ``tp6h``. If it is missing, fall back to ``tp`` with a warning (not physically
    equivalent to 6h accum).
    """
    path_6h = layout.single_path(t, "tp6h")
    if path_6h.is_file():
        x = load_npy_2d(path_6h, flip_north_south=flip_north_south).astype(np.float32)
        return _ensure_025_shape(x, field_name=f"tp6h@{t:%Y%m%d%H}")
    path_tp = layout.single_path(t, "tp")
    logger.warning(
        "ERA5: %s missing; using hourly tp — not a 6h accumulation.",
        path_6h,
    )
    x = load_npy_2d(path_tp, flip_north_south=flip_north_south).astype(np.float32)
    return _ensure_025_shape(x, field_name=f"tp@{t:%Y%m%d%H}")


def snapshot_has_pressure_files(layout: Era5NpyLayout, dt: datetime) -> bool:
    """Quick presence check (z at 500 hPa + surface t2m)."""
    try:
        p = layout.pressure_path(dt, "z", 500.0)
        s = layout.single_path(dt, "t2m")
        return p.is_file() and s.is_file()
    except Exception:
        return False


def snapshot_has_single_files(layout: Era5NpyLayout, dt: datetime) -> bool:
    """True if surface t2m exists (single-tree only)."""
    return layout.single_path(dt, "t2m").is_file()


# ---------------------------------------------------------------------------
# Pangu
# ---------------------------------------------------------------------------

PANGU_UPPER_VARS: Sequence[str] = ("z", "q", "t", "u", "v")
PANGU_SURFACE_VARS: Sequence[str] = ("msl", "u10", "v10", "t2m")


def load_pangu_inputs(
    init_time: datetime,
    *,
    root: Path = DEFAULT_ERA5_NPY_ROOT,
    flip_north_south: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Pangu ONNX inputs from NPY snapshots.

    Returns
    -------
    input_upper : np.ndarray, shape (5, 13, 721, 1440)
    input_surface : np.ndarray, shape (4, 721, 1440)
        Surface order: msl, u10, v10, t2m (matches src/pangu/prepare.py).
    """
    flip = _DEFAULT_FLIP_NS if flip_north_south is None else flip_north_south
    layout = Era5NpyLayout(root)
    slabs: List[np.ndarray] = []
    for var in PANGU_UPPER_VARS:
        for lev in PANGU_PRESSURE_LEVELS:
            slabs.append(
                load_npy_2d(
                    layout.pressure_path(init_time, var, float(lev)),
                    flip_north_south=flip,
                )
            )
    upper = np.stack(slabs, axis=0).reshape(5, len(PANGU_PRESSURE_LEVELS), 721, 1440)
    surface = np.stack(
        [
            load_npy_2d(
                layout.single_path(init_time, v),
                flip_north_south=flip,
            )
            for v in PANGU_SURFACE_VARS
        ],
        axis=0,
    )
    return upper.astype(np.float32), surface.astype(np.float32)




def pangu_channel_names() -> List[str]:
    names: List[str] = []
    for var in PANGU_UPPER_VARS:
        for lev in PANGU_PRESSURE_LEVELS:
            names.append(f"{var}_{lev}")
    names.extend(PANGU_SURFACE_VARS)
    return names


def neuralgcm_channel_names_from_reader() -> List[str]:
    """
    NeuralGCM metric channel order used in benchmark outputs.

    Order follows official atmospheric vars
    (z,t,u,v,q,ciwc,clwc) x 37 pressure levels.
    """
    names: List[str] = []
    for var in NEURALGCM_UPPER_SHORTS:
        for lev in NEURALGCM_PRESSURE_LEVELS_ERA5:
            names.append(f"{var}_{lev}")
    return names


def load_neuralgcm_upper_stack(
    t: datetime,
    *,
    root: Path = DEFAULT_ERA5_NPY_ROOT,
    flip_north_south: bool = True,
) -> np.ndarray:
    """
    Load NeuralGCM atmospheric stack from ERA5 NPY:
    (7 * 37, 721, 1440) in (z,t,u,v,q,ciwc,clwc) x levels order.
    """
    layout = Era5NpyLayout(root)
    slabs: List[np.ndarray] = []
    for var in NEURALGCM_UPPER_SHORTS:
        for lev in NEURALGCM_PRESSURE_LEVELS_ERA5:
            slabs.append(
                load_npy_2d(
                    layout.pressure_path(t, var, float(lev)),
                    flip_north_south=flip_north_south,
                )
            )
    return np.stack(slabs, axis=0).astype(np.float32)


def load_pangu_ground_truth_stack(
    valid_time: datetime,
    *,
    root: Path = DEFAULT_ERA5_NPY_ROOT,
    flip_north_south: bool = True,
) -> tuple[np.ndarray, List[str]]:
    """
    Stack the 69 Pangu channels at validation time for metrics (physics space).
    """
    u, s = load_pangu_inputs(valid_time, root=root, flip_north_south=flip_north_south)
    # Same flatten order as channel names: 65 upper + 4 surface
    upper_flat = u.reshape(5 * len(PANGU_PRESSURE_LEVELS), 721, 1440)
    pred_stack = np.concatenate([upper_flat, s], axis=0)
    return pred_stack.astype(np.float32), pangu_channel_names()




# ---------------------------------------------------------------------------
# Stormer (full variable list + ERA5 file mapping)
# ---------------------------------------------------------------------------

STORMER_PRESSURE_LEVELS: List[int] = [
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
]

# Order must match src/stormer/inference.py `variables`
STORMER_ERA5_NAMES: List[str] = (
    [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
    ]
    + [f"geopotential_{l}" for l in STORMER_PRESSURE_LEVELS]
    + [f"u_component_of_wind_{l}" for l in STORMER_PRESSURE_LEVELS]
    + [f"v_component_of_wind_{l}" for l in STORMER_PRESSURE_LEVELS]
    + [f"temperature_{l}" for l in STORMER_PRESSURE_LEVELS]
    + [f"specific_humidity_{l}" for l in STORMER_PRESSURE_LEVELS]
)


def _stormer_name_to_npy_var_and_level(
    name: str,
) -> tuple[str, float | None]:
    """Map Stormer training name to ERA5 short name + optional pressure (hPa)."""
    if name == "2m_temperature":
        return "t2m", None
    if name == "10m_u_component_of_wind":
        return "u10", None
    if name == "10m_v_component_of_wind":
        return "v10", None
    if name == "mean_sea_level_pressure":
        return "msl", None
    base, lev = name.rsplit("_", 1)
    mapping = {
        "geopotential": "z",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
        "temperature": "t",
        "specific_humidity": "q",
    }
    if base in mapping and lev.isdigit():
        return mapping[base], float(lev)
    raise ValueError(f"Unmapped Stormer variable {name}")


def load_stormer_stack(
    t: datetime,
    *,
    root: Path = DEFAULT_ERA5_NPY_ROOT,
    flip_north_south: bool = True,
) -> np.ndarray:
    """
    Stack all Stormer channels → (V, 721, 1440) in training order.
    """
    layout = Era5NpyLayout(root)
    channels: List[np.ndarray] = []
    for era_name in STORMER_ERA5_NAMES:
        short, lev = _stormer_name_to_npy_var_and_level(era_name)
        if lev is None:
            path = layout.single_path(t, short)
        else:
            path = layout.pressure_path(t, short, lev)
        channels.append(load_npy_2d(path, flip_north_south=flip_north_south))
    return np.stack(channels, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# FengWu (69 channels; needs t0 and t0-6h)
# ---------------------------------------------------------------------------


def _fengwu_single_frame(
    t: datetime,
    *,
    root: Path,
    flip_north_south: bool,
) -> np.ndarray:
    layout = Era5NpyLayout(root)
    feats: List[np.ndarray] = []
    for v in ("u10", "v10", "t2m", "msl"):
        feats.append(load_npy_2d(layout.single_path(t, v), flip_north_south=flip_north_south))
    for var in ("z", "q", "u", "v", "t"):
        for lev in FENGWU_PRESSURE_LEVELS:
            feats.append(
                load_npy_2d(
                    layout.pressure_path(t, var, float(lev)),
                    flip_north_south=flip_north_south,
                )
            )
    out = np.stack(feats, axis=0).astype(np.float32)
    if out.shape != (69, 721, 1440):
        raise RuntimeError(f"FengWu frame shape mismatch: {out.shape}")
    return out


def fengwu_channel_names() -> List[str]:
    names = ["u10", "v10", "t2m", "msl"]
    for var in ("z", "q", "u", "v", "t"):
        for lev in FENGWU_PRESSURE_LEVELS:
            names.append(f"{var}_{lev}")
    return names


def load_fengwu_snapshot(
    t: datetime,
    *,
    root: Path = DEFAULT_ERA5_NPY_ROOT,
    flip_north_south: bool = True,
) -> np.ndarray:
    """Single-time FengWu channel stack ``(69, 721, 1440)``."""
    return _fengwu_single_frame(t, root=root, flip_north_south=flip_north_south)


def load_fuxi_snapshot(
    t: datetime,
    *,
    root: Path = DEFAULT_ERA5_NPY_ROOT,
    flip_north_south: bool = True,
) -> np.ndarray:
    """FuXi channel order at one instant: ``(70, 721, 1440)``."""
    layout = Era5NpyLayout(root)
    chan_list: List[np.ndarray] = []
    for pl in FUXI_PL_NAMES:
        for lev in FUXI_LEVELS:
            chan_list.append(
                load_npy_2d(
                    layout.pressure_path(t, pl, float(lev)),
                    flip_north_south=flip_north_south,
                )
            )
    for sfc in FUXI_SFC_NAMES:
        if sfc == "tp":
            chan_list.append(_load_fuxi_tp_like_official_repo(layout, t, flip_north_south=flip_north_south))
        else:
            chan_list.append(
                load_npy_2d(
                    layout.single_path(t, sfc),
                    flip_north_south=flip_north_south,
                )
            )
    return np.stack(chan_list, axis=0).astype(np.float32)


def load_fengwu_init_pair(
    init_time: datetime,
    *,
    root: Path = DEFAULT_ERA5_NPY_ROOT,
    flip_north_south: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (input1, input2) normalized meanings: input1 = state at init-6h,
    input2 = state at init (matches src/fengwu/inference expectations).
    """
    t0 = init_time
    tm6 = init_time - timedelta(hours=6)
    return (
        _fengwu_single_frame(tm6, root=root, flip_north_south=flip_north_south),
        _fengwu_single_frame(t0, root=root, flip_north_south=flip_north_south),
    )


# ---------------------------------------------------------------------------
# FuXi (70 channels; 2 timesteps along first axis in model input)
# ---------------------------------------------------------------------------


def _fuxi_channel_order() -> List[str]:
    order: List[str] = []
    for n in FUXI_PL_NAMES:
        for lev in FUXI_LEVELS:
            # Keep variable names consistent with other models (e.g. z_500, t_850).
            order.append(f"{n.lower()}_{lev}")
    order.extend([x.lower() for x in FUXI_SFC_NAMES])
    assert len(order) == 70
    return order


def _load_fuxi_tp_like_official_repo(
    layout: Era5NpyLayout,
    t: datetime,
    *,
    flip_north_south: bool,
) -> np.ndarray:
    """
    Match tpys/FuXi ``make_era5_input.py``: 6-hour precipitation sum in metres,
    then ``* 1000`` → millimetres and ``clip(0, 1000)``.

    See https://github.com/tpys/FuXi (``make_era5_input.py``) and README (TP = 6h accum).
    """
    x = load_era5_tp6h_depth_m(layout, t, flip_north_south=flip_north_south)
    x *= np.float32(1000.0)
    np.clip(x, 0.0, 1000.0, out=x)
    return x


def load_fuxi_input_tensor(
    init_time: datetime,
    *,
    root: Path = DEFAULT_ERA5_NPY_ROOT,
    flip_north_south: bool = True,
) -> tuple[np.ndarray, List[str]]:
    """
    Build FuXi ONNX `input` tensor with shape (1, 2, 70, 721, 1440).

    Time slices: [init-6h, init].
    """
    layout = Era5NpyLayout(root)
    times = (init_time - timedelta(hours=6), init_time)
    frames: List[np.ndarray] = []
    for t in times:
        chan_list: List[np.ndarray] = []
        for pl in FUXI_PL_NAMES:
            for lev in FUXI_LEVELS:
                chan_list.append(
                    load_npy_2d(
                        layout.pressure_path(t, pl, float(lev)),
                        flip_north_south=flip_north_south,
                    )
                )
        for sfc in FUXI_SFC_NAMES:
            if sfc == "tp":
                chan_list.append(_load_fuxi_tp_like_official_repo(layout, t, flip_north_south=flip_north_south))
            else:
                chan_list.append(
                    load_npy_2d(
                        layout.single_path(t, sfc),
                        flip_north_south=flip_north_south,
                    )
                )
        stacked = np.stack(chan_list, axis=0).astype(np.float32)
        if stacked.shape != (70, 721, 1440):
            raise RuntimeError(f"FuXi slice shape {stacked.shape}")
        frames.append(stacked)
    cube = np.stack(frames, axis=0)[None, ...]  # (1,2,70,H,W)
    return cube.astype(np.float32), _fuxi_channel_order()


def fuxi_channel_names() -> List[str]:
    return _fuxi_channel_order()


def load_snapshot_by_channel_names(
    t: datetime,
    channel_names: Sequence[str],
    *,
    root: Path = DEFAULT_ERA5_NPY_ROOT,
    flip_north_south: bool = True,
) -> np.ndarray:
    """
    Load arbitrary channel list from ERA5 NPY by short names.

    Supported formats:
    - Surface short names: t2m, u10, v10, msl, tp, tp6h
    - Pressure names: z_500, t_850, u_250, v_700, q_500, r_500, w_500
    """
    layout = Era5NpyLayout(root)
    out: List[np.ndarray] = []
    for name in channel_names:
        base = name
        lev: float | None = None
        if "_" in name:
            maybe_base, maybe_lev = name.rsplit("_", 1)
            if maybe_lev.isdigit():
                base = maybe_base
                lev = float(maybe_lev)
        if lev is None:
            path = layout.single_path(t, base)
        else:
            path = layout.pressure_path(t, base, lev)
        arr = load_npy_2d(path, flip_north_south=flip_north_south)
        if lev is None and base.lower() in {"skt", "tp", "tp6h"}:
            arr = _ensure_025_shape(arr, field_name=f"{base}@{t:%Y%m%d%H}")
        out.append(arr)
    return np.stack(out, axis=0).astype(np.float32)

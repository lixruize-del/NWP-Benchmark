"""AIFS forecast runner (GRIB or ERA5 NPY -> SimpleRunner -> 0.25° fields) (GRIB -> SimpleRunner -> 0.25deg fields).

This implementation is intentionally rewritten from scratch and follows:
- Ref-A: `src/aifs/prepare.py`
- Ref-B: `src/aifs/inference.py`
- Ref-C: official notebook `run_AIFS_v1.1.ipynb`
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, Iterable, List

import earthkit.data as ekd  # Ref-A: GRIB reading via earthkit.data
import earthkit.regrid as ekr  # Ref-A/Ref-B: regridding with earthkit.regrid
import numpy as np
from scipy.ndimage import zoom
import torch
from anemoi.inference.runners.simple import SimpleRunner  # Ref-B/Ref-C: inference backend
import xarray as xr
from src.common.data_reader import DEFAULT_ERA5_NPY_ROOT, Era5NpyLayout, load_npy_2d


LOG = logging.getLogger(__name__)

# Ref-A: pressure levels and soil levels used by AIFS preprocessing.
PRESSURE_LEVELS: List[int] = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS: List[int] = [1, 2]

# Ref-A: dynamic/static/surface and upper variables used to construct input state.
DYNAMIC_SFC_PARAMS: List[str] = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
STATIC_SFC_PARAMS: List[str] = ["lsm", "z", "slor", "sdor"]
SOIL_PARAMS: List[str] = ["vsw", "sot"]
PRESSURE_PARAMS: List[str] = ["gh", "t", "u", "v", "q", "w"]

# Ref-A: soil key remap in preprocessing.
SOIL_RENAME: dict[tuple[str, int], str] = {
    ("vsw", 1): "swvl1",
    ("vsw", 2): "swvl2",
    ("sot", 1): "stl1",
    ("sot", 2): "stl2",
}

# Ref-v1.5: ERA5 short-name mapping used to build AIFS input state.
ERA5_TO_AIFS_SINGLE = {
    "10u": ("u10",),
    "10v": ("v10",),
    "2d": ("d2m",),
    "2t": ("t2m",),
    "msl": ("msl",),
    "skt": ("skt",),
    "sp": ("sp",),
    "tcw": ("tcwv", "tcw"),
    "lsm": ("lsm",),
    "z": ("z",),
    "slor": ("slor",),
    "sdor": ("sdor",),
    "swvl1": ("swvl1",),
    "swvl2": ("swvl2",),
    "stl1": ("stl1",),
    "stl2": ("stl2",),
}
AIFS_STATIC_KEYS = ["lsm", "z", "slor", "sdor"]
AIFS_SOIL_KEYS = ["swvl1", "swvl2", "stl1", "stl2"]
_STATIC_CACHE: dict[str, dict[str, np.ndarray]] = {}

# Ref-B: output key normalization used when exporting from model states.
AIFS_OUTPUT_MAP: dict[str, str] = {
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

# Ref-A/Ref-B: grid constants.
N320_POINTS = 542080
SOURCE_025 = {"grid": (0.25, 0.25)}
TARGET_N320 = {"grid": "N320"}
SOURCE_N320 = {"grid": "N320"}
TARGET_025 = {"grid": (0.25, 0.25)}
TARGET_025_SHAPE = (721, 1440)


def aifs_channel_names() -> List[str]:
    """Return benchmark channel order for AIFS evaluation.

    Ref: the order follows existing benchmark AIFS channel contract
    (surface + static + pl + tp/cp/tcc), matching v2 adapter expectation.
    """
    names: List[str] = []
    names.extend(["u10", "v10", "d2m", "t2m", "msl", "skt", "sp", "tcwv"])
    names.extend(["lsm", "z", "slor", "sdor", "swvl1", "swvl2", "stl1", "stl2"])
    for base in ("z", "t", "u", "v", "q", "w"):
        for lev in PRESSURE_LEVELS:
            names.append(f"{base}_{lev}")
    names.extend(["tp", "cp", "tcc"])
    return names


def _roll_longitude_before_n320(arr_2d: np.ndarray) -> np.ndarray:
    """Apply the same longitude roll before 0.25->N320 regrid.

    Ref-A/Ref-C: `np.roll(..., -width//2, axis=1)` prior to regridding.
    """
    return np.roll(arr_2d, -arr_2d.shape[1] // 2, axis=1)


def _roll_longitude_after_025(arr_2d: np.ndarray) -> np.ndarray:
    """Inverse longitude roll for ERA5 np.25 convention (0..360 ordering)."""
    return np.roll(arr_2d, arr_2d.shape[1] // 2, axis=1)


def _align_era5_to_ifs_longitude(arr_2d: np.ndarray) -> np.ndarray:
    """Align ERA5-style 0.25 fields to IFS-style longitude convention.

    Empirical diagnostics on all AIFS input variables show a systematic 180-degree
    offset between the ERA5 npy branch and IFS raw branch before model input build.
    We normalize ERA5-side source fields first, then both branches share the same
    `_roll_longitude_before_n320 -> regrid_025_to_n320` path.
    """
    return _roll_longitude_after_025(arr_2d)


def _regrid_025_to_n320(arr_2d: np.ndarray) -> np.ndarray:
    """0.25deg latlon -> N320 1D field.

    Ref-A: `ekr.interpolate(arr, SOURCE_GRID, TARGET_GRID)`.
    """
    out = ekr.interpolate(arr_2d, SOURCE_025, TARGET_N320)
    return np.asarray(out, dtype=np.float32)


def _ensure_025_shape(arr_2d: np.ndarray, *, field_name: str) -> np.ndarray:
    """Ensure input is 0.25-degree global shape (721, 1440).

    Some externally prepared inputs (e.g. SKT/TP6H side channels) may arrive on
    different grids. We resize them to benchmark 0.25-degree shape before
    passing to the 0.25->N320 regridder.
    """
    arr = np.asarray(arr_2d, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D field for {field_name}, got shape={arr.shape}")
    if arr.shape == TARGET_025_SHAPE:
        return arr
    h, w = arr.shape
    zoom_factors = (TARGET_025_SHAPE[0] / float(h), TARGET_025_SHAPE[1] / float(w))
    LOG.warning(
        "AIFS input %s shape %s != %s; resizing with scipy.ndimage.zoom(order=1).",
        field_name,
        arr.shape,
        TARGET_025_SHAPE,
    )
    resized = zoom(arr, zoom_factors, order=1)
    return np.asarray(resized, dtype=np.float32, order="C")


def _build_n320_to_025_regridder(device: torch.device):
    """Create sparse N320->0.25 regrid closure.

    Ref-B: `ekr.db.find(...)` + sparse CSR matmul.
    """
    weights_csr, target_shape = ekr.db.find(SOURCE_N320, TARGET_025, "linear")
    w = torch.sparse_csr_tensor(
        torch.from_numpy(weights_csr.indptr),
        torch.from_numpy(weights_csr.indices),
        torch.from_numpy(weights_csr.data),
        size=weights_csr.shape,
        device=device,
    )

    def _regrid(arr_1d: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.asarray(arr_1d, dtype=np.float64)).to(device)
        out = w.matmul(t)
        return out.detach().cpu().numpy().astype(np.float32).reshape(target_shape)

    return _regrid


def _safe_state_attr(state, key: str):
    """Read attributes from either object-style or dict-style state.

    Ref-B: compatibility helper for forecast state objects.
    """
    if hasattr(state, key):
        return getattr(state, key)
    if isinstance(state, dict) and key in state:
        return state[key]
    return None


def _norm_level_name(name: str) -> str:
    """Normalize model field names to benchmark names.

    Ref-B: output mapping logic + pressure suffix normalization.
    """
    if "_" not in name:
        return AIFS_OUTPUT_MAP.get(name, name)
    base, lev = name.rsplit("_", 1)
    if lev.isdigit():
        base_norm = AIFS_OUTPUT_MAP.get(base, base)
        return f"{base_norm}_{int(lev)}"
    return AIFS_OUTPUT_MAP.get(name, name)


def _read_grib_fields(path: Path, params: Iterable[str], levelist: Iterable[int] | None = None) -> dict[str, np.ndarray]:
    """Read GRIB fields keyed as `<param>` or `<param>_<level>`.

    Ref-A: `ekd.from_source(...).sel(param=...).sel(levelist=...)`.
    """
    ds = ekd.from_source("file", str(path))
    ds = ds.sel(param=list(params))
    if levelist is not None:
        ds = ds.sel(levelist=list(levelist))

    out: dict[str, np.ndarray] = {}
    for f in ds:
        p = str(f.metadata("param"))
        lev = f.metadata("levelist", default=None)
        key = f"{p}_{int(lev)}" if lev is not None else p
        out[key] = np.asarray(f.to_numpy(), dtype=np.float32)
    return out


def _load_single_any(layout: Era5NpyLayout, t: datetime, candidates: Iterable[str]) -> np.ndarray:
    """Load one ERA5 single-level field from candidate short names."""
    for short in candidates:
        p = layout.single_path(t, short)
        if p.exists():
            return load_npy_2d(p, flip_north_south=False)
    raise FileNotFoundError(f"No ERA5 single-level file found for {list(candidates)} at {t:%Y-%m-%d %H:%M}")


def _load_upper(layout: Era5NpyLayout, t: datetime, short: str, lev: int) -> np.ndarray:
    """Load one ERA5 pressure-level field; supports gh->z fallback."""
    p = layout.pressure_path(t, short, float(lev))
    if p.exists():
        return load_npy_2d(p, flip_north_south=False)
    if short == "z":
        p_gh = layout.pressure_path(t, "gh", float(lev))
        if p_gh.exists():
            gh = load_npy_2d(p_gh, flip_north_south=False)
            return (gh * 9.80665).astype(np.float32)
    raise FileNotFoundError(str(p))


def _load_static_from_nc(repo_root: Path) -> dict[str, np.ndarray]:
    """Load static fields from repository `static.nc` as fallback."""
    key = str(repo_root / "static.nc")
    if key in _STATIC_CACHE:
        return _STATIC_CACHE[key]
    p = repo_root / "static.nc"
    if not p.exists():
        _STATIC_CACHE[key] = {}
        return _STATIC_CACHE[key]
    ds = xr.open_dataset(p)
    try:
        out: dict[str, np.ndarray] = {}
        for name in ("z", "lsm", "slor", "sdor"):
            if name in ds.data_vars:
                out[name] = np.asarray(ds[name].isel(valid_time=0).values, dtype=np.float32)
        if "z" not in out and "geopotential_at_surface" in ds.data_vars:
            out["z"] = np.asarray(ds["geopotential_at_surface"].values, dtype=np.float32)
        if "lsm" not in out and "land_sea_mask" in ds.data_vars:
            out["lsm"] = np.asarray(ds["land_sea_mask"].values, dtype=np.float32)
        _STATIC_CACHE[key] = out
    finally:
        ds.close()
    return _STATIC_CACHE[key]


def _resolve_cycle_files(root: Path, t: datetime) -> tuple[Path, Path, Path]:
    """Resolve per-cycle surface/soil/upper GRIB paths.

    Supports both naming styles used in this workspace:
    - Ref external downloader style: `surface_*_hres.grib`, `land_*_hres.grib`, `upper_*_hres.grib`
    - Local compact downloader style: `sfc_*.grib`, `sol_*.grib`, `pl_*.grib`
    """
    tag = t.strftime("%Y%m%d%H")
    cands = [
        (root / f"surface_{tag}_hres.grib", root / f"land_{tag}_hres.grib", root / f"upper_{tag}_hres.grib"),
        (root / f"sfc_{tag}.grib", root / f"sol_{tag}.grib", root / f"pl_{tag}.grib"),
    ]
    for sfc, sol, pl in cands:
        if sfc.exists() and sol.exists() and pl.exists():
            return sfc, sol, pl
    raise FileNotFoundError(f"Cannot resolve IFS cycle GRIB files for {tag} under {root}")


class AifsForecastRunner:
    """AIFS forecast runner rewritten from reference preprocess/inference code."""

    def __init__(
        self,
        *,
        era5_root: Path | None = None,
        ifs_hres_root: Path | None = None,
        weights_root: Path | None = None,
        checkpoint: str = "aifs-single-mse-1.1.ckpt",
        device: str | None = None,
    ) -> None:
        # Ref-B: environment knobs used for stable AIFS inference.
        import os

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS", "16")

        self.era5_root = Path(era5_root) if era5_root is not None else None
        self.ifs_hres_root = Path(ifs_hres_root) if ifs_hres_root is not None else None
        self.layout = Era5NpyLayout(self.era5_root) if self.era5_root is not None else None
        self._repo_root = Path(__file__).resolve().parents[2]
        if weights_root is None:
            weights_root = Path("/ecmwf-era5-datasets/nwp_bench/assets/weights/aifs")
        self.checkpoint_path = Path(weights_root) / checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(str(self.checkpoint_path))

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._runner = SimpleRunner(checkpoint=str(self.checkpoint_path), device=self.device)
        self._torch_device = torch.device("cuda" if self.device == "cuda" else "cpu")
        self._regrid_n320_to_025 = _build_n320_to_025_regridder(self._torch_device)

        mode = "ifs" if self.ifs_hres_root is not None else "era5"
        LOG.info("AIFS loaded checkpoint=%s device=%s mode=%s", self.checkpoint_path.name, self.device, mode)

    def _assemble_input_state(
        self,
        init_time: datetime,
        *,
        load_dynamic: callable,
        load_static: callable,
        load_soil: callable,
        load_upper: callable,
    ) -> dict:
        """Unified pre-model input assembly for both ERA5 and IFS branches."""
        t6 = init_time - timedelta(hours=6)
        fields: dict[str, np.ndarray] = {}

        def _to_n320(arr_2d: np.ndarray, *, field_name: str) -> np.ndarray:
            arr_025 = _ensure_025_shape(arr_2d, field_name=field_name)
            return _regrid_025_to_n320(_roll_longitude_before_n320(arr_025))

        for key in DYNAMIC_SFC_PARAMS:
            a_t6 = _to_n320(load_dynamic(t6, key), field_name=f"{key}@{t6:%Y%m%d%H}")
            a_t0 = _to_n320(load_dynamic(init_time, key), field_name=f"{key}@{init_time:%Y%m%d%H}")
            fields[key] = np.stack([a_t6, a_t0], axis=0).astype(np.float32, copy=False)

        for key in AIFS_STATIC_KEYS:
            a_t0 = _to_n320(load_static(init_time, key), field_name=f"{key}@{init_time:%Y%m%d%H}")
            fields[key] = np.stack([a_t0, a_t0], axis=0).astype(np.float32, copy=False)

        for key in AIFS_SOIL_KEYS:
            a_t6 = _to_n320(load_soil(t6, key), field_name=f"{key}@{t6:%Y%m%d%H}")
            a_t0 = _to_n320(load_soil(init_time, key), field_name=f"{key}@{init_time:%Y%m%d%H}")
            fields[key] = np.stack([a_t6, a_t0], axis=0).astype(np.float32, copy=False)

        for lev in PRESSURE_LEVELS:
            for base_short in ("z", "t", "u", "v", "q", "w"):
                k = f"{base_short}_{lev}"
                a_t6 = _to_n320(load_upper(t6, base_short, lev), field_name=f"{k}@{t6:%Y%m%d%H}")
                a_t0 = _to_n320(load_upper(init_time, base_short, lev), field_name=f"{k}@{init_time:%Y%m%d%H}")
                fields[k] = np.stack([a_t6, a_t0], axis=0).astype(np.float32, copy=False)
                if fields[k].shape != (2, N320_POINTS):
                    raise ValueError(f"Unexpected input shape for {k}: {fields[k].shape}")

        return {"date": init_time, "fields": fields}

    def _build_input_state_from_ifs(self, init_time: datetime) -> dict:
        """Construct input state from IFS GRIB via unified assembler."""
        t0 = init_time
        t6 = init_time - timedelta(hours=6)
        sfc_t6, sol_t6, pl_t6 = _resolve_cycle_files(self.ifs_hres_root, t6)
        sfc_t0, sol_t0, pl_t0 = _resolve_cycle_files(self.ifs_hres_root, t0)

        dyn_maps = [
            _read_grib_fields(sfc_t6, DYNAMIC_SFC_PARAMS),
            _read_grib_fields(sfc_t0, DYNAMIC_SFC_PARAMS),
        ]
        static_map = _read_grib_fields(sfc_t0, STATIC_SFC_PARAMS)
        soil_raw_maps = [
            _read_grib_fields(sol_t6, SOIL_PARAMS, levelist=SOIL_LEVELS),
            _read_grib_fields(sol_t0, SOIL_PARAMS, levelist=SOIL_LEVELS),
        ]
        pl_raw_maps = [
            _read_grib_fields(pl_t6, PRESSURE_PARAMS, levelist=PRESSURE_LEVELS),
            _read_grib_fields(pl_t0, PRESSURE_PARAMS, levelist=PRESSURE_LEVELS),
        ]

        def _idx(t: datetime) -> int:
            return 0 if t == t6 else 1

        def _load_dynamic(t: datetime, key: str) -> np.ndarray:
            return dyn_maps[_idx(t)][key]

        def _load_static(_t: datetime, key: str) -> np.ndarray:
            return static_map[key]

        def _load_soil(t: datetime, key: str) -> np.ndarray:
            raw_map = soil_raw_maps[_idx(t)]
            for (param, lev), mapped in SOIL_RENAME.items():
                if mapped == key:
                    return raw_map[f"{param}_{lev}"]
            raise KeyError(key)

        def _load_upper(t: datetime, base_short: str, lev: int) -> np.ndarray:
            raw = pl_raw_maps[_idx(t)]
            if base_short == "z":
                return raw[f"gh_{lev}"] * np.float32(9.80665)
            return raw[f"{base_short}_{lev}"]

        return self._assemble_input_state(
            init_time,
            load_dynamic=_load_dynamic,
            load_static=_load_static,
            load_soil=_load_soil,
            load_upper=_load_upper,
        )

    def _build_input_state_from_era5(self, init_time: datetime) -> dict:
        """Construct input state from ERA5 npy via unified assembler."""
        assert self.layout is not None
        static_map = _load_static_from_nc(self._repo_root)

        def _load_dynamic(t: datetime, key: str) -> np.ndarray:
            arr = _load_single_any(self.layout, t, ERA5_TO_AIFS_SINGLE[key])
            return _align_era5_to_ifs_longitude(arr)

        def _load_static(t: datetime, key: str) -> np.ndarray:
            try:
                arr = _load_single_any(self.layout, t, ERA5_TO_AIFS_SINGLE[key])
            except FileNotFoundError:
                if key in static_map:
                    arr = static_map[key]
                else:
                    raise
            return _align_era5_to_ifs_longitude(arr)

        def _load_soil(t: datetime, key: str) -> np.ndarray:
            arr = _load_single_any(self.layout, t, ERA5_TO_AIFS_SINGLE[key])
            return _align_era5_to_ifs_longitude(arr)

        def _load_upper_local(t: datetime, base_short: str, lev: int) -> np.ndarray:
            arr = _load_upper(self.layout, t, base_short, lev)
            return _align_era5_to_ifs_longitude(arr)

        return self._assemble_input_state(
            init_time,
            load_dynamic=_load_dynamic,
            load_static=_load_static,
            load_soil=_load_soil,
            load_upper=_load_upper_local,
        )

    def _state_to_channel_dict_025(self, state) -> dict[str, np.ndarray]:
        """Convert one forecast state to 0.25-degree 2D channels.

        Ref-B: output processing uses N320->0.25 sparse regrid when needed.
        """
        fields = _safe_state_attr(state, "fields")
        if fields is None:
            return {}
        if not isinstance(fields, dict):
            fields = dict(fields)

        out: dict[str, np.ndarray] = {}
        for key, val in fields.items():
            x = val.detach().cpu().numpy() if hasattr(val, "detach") else np.asarray(val)
            x = np.asarray(x)
            if x.ndim > 1:
                x = np.squeeze(x)
            if x.ndim == 1 and x.size == N320_POINTS:
                x2 = self._regrid_n320_to_025(x)
            elif x.ndim == 2 and x.shape == (721, 1440):
                x2 = x.astype(np.float32, copy=False)
            else:
                continue
            out[_norm_level_name(str(key))] = x2.astype(np.float32, copy=False)
        return out

    def run(self, init_time: datetime, lead_times_hours: List[int]) -> Dict[int, np.ndarray]:
        """Run AIFS and return `{lead_h: [C,721,1440]}` in benchmark order."""
        if not lead_times_hours:
            return {}

        wanted = sorted(set(int(x) for x in lead_times_hours))
        max_lead = max(wanted)
        if max_lead % 6 != 0:
            raise ValueError("AIFS supports 6-hour step outputs; lead must be multiples of 6.")

        if self.ifs_hres_root is not None:
            input_state = self._build_input_state_from_ifs(init_time)
        elif self.layout is not None:
            input_state = self._build_input_state_from_era5(init_time)
        else:
            raise RuntimeError("AIFS requires either `ifs_hres_root` or `era5_root`.")
        forecast_iter = self._runner.run(input_state=input_state, lead_time=max_lead)

        channel_order = aifs_channel_names()
        result: Dict[int, np.ndarray] = {}

        for st in forecast_iter:
            valid_time = _safe_state_attr(st, "date")
            if valid_time is None:
                continue
            lead_h = int(round((valid_time - init_time).total_seconds() / 3600))
            if lead_h not in wanted:
                continue
            ch_map = self._state_to_channel_dict_025(st)
            stack = np.stack(
                [ch_map.get(name, np.full((721, 1440), np.nan, dtype=np.float32)) for name in channel_order],
                axis=0,
            ).astype(np.float32, copy=False)
            result[lead_h] = stack

            if len(result) == len(wanted):
                break

        missing = [x for x in wanted if x not in result]
        if missing:
            raise RuntimeError(f"AIFS missing outputs for leads={missing} at init={init_time:%Y%m%d%H}")
        return result


def run_aifs_forecast(
    init_time: datetime,
    lead_times_hours: List[int],
    *,
    era5_root: Path | None = None,
    ifs_hres_root: Path | None = None,
    weights_root: Path | None = None,
    checkpoint: str = "aifs-single-mse-1.1.ckpt",
    device: str | None = None,
    **kwargs: object,
) -> Dict[int, np.ndarray]:
    """Functional API used by ``run_large_scale*.py`` legacy ``build_adapter``."""
    del kwargs
    runner = AifsForecastRunner(
        era5_root=era5_root if era5_root is not None else DEFAULT_ERA5_NPY_ROOT,
        ifs_hres_root=ifs_hres_root,
        weights_root=weights_root,
        checkpoint=checkpoint,
        device=device,
    )
    return runner.run(init_time, lead_times_hours)

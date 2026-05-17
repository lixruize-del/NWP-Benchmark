"""Aurora runner aligned with home inference flow."""  # src:/home/NWP-Benchmark/src/aurora/inference.py:1-12

from __future__ import annotations  # src:/home/NWP-Benchmark/src/aurora/inference.py:1

import logging  # src:/home/NWP-Benchmark/src/aurora/inference.py:3
import os  # src:/home/NWP-Benchmark/src/aurora/inference.py:1
from datetime import datetime, timedelta  # src:/vepfs adapter style
from pathlib import Path  # src:/home/NWP-Benchmark/src/aurora/inference.py:10
from typing import Dict, List  # src:/home/NWP-Benchmark/src/aurora/inference.py:12
from unittest.mock import patch  # src:/home/NWP-Benchmark/src/aurora/inference.py:11

import numpy as np  # src:/home/NWP-Benchmark/src/aurora/inference.py:9
import torch  # src:/home/NWP-Benchmark/src/aurora/inference.py:7
import xarray as xr  # src:/home/NWP-Benchmark/src/aurora/inference.py:8
from aurora import Aurora, Batch, Metadata  # src:/home/NWP-Benchmark/src/aurora/inference.py:22
from aurora.rollout import rollout  # official multi-step history: cat([hist[:,1:], pred], dim=1)

from src.common.data_reader_ifs import DEFAULT_ERA5_NPY_ROOT, Era5NpyLayout, load_npy_2d  # src:/vepfs data reader

logger = logging.getLogger(__name__)  # src:/home/NWP-Benchmark/src/aurora/inference.py:35

DEFAULT_WEIGHTS_ROOT = Path(os.environ.get("NWP_WEIGHTS_ROOT", "/ecmwf-era5-datasets/nwp_bench/assets/weights"))  # src:/home/NWP-Benchmark/src/aurora/inference.py:25 adapted
from src.common.repo_paths import static_nc_path

DEFAULT_STATIC_NC = static_nc_path()  # src:/home static.nc usage adapted

AURORA_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]  # src:/home/NWP-Benchmark/src/aurora/inference.py:105
SURFACE_SHORT_ORDER = ["t2m", "u10", "v10", "msl"]  # src:/home/NWP-Benchmark/src/aurora/inference.py:112-116 mapped order
ATMOS_SHORT_ORDER = ["t", "u", "v", "q", "z"]  # src:/home/NWP-Benchmark/src/aurora/inference.py:123-127 mapped order


def aurora_channel_names() -> List[str]:  # src:/home/NWP-Benchmark/src/aurora/inference.py:209-231 output mapping logic
    names: List[str] = []  # src:/home/NWP-Benchmark/src/aurora/inference.py:206
    names.extend(SURFACE_SHORT_ORDER)  # src:/home/NWP-Benchmark/src/aurora/inference.py:209-218
    for var in ATMOS_SHORT_ORDER:  # src:/home/NWP-Benchmark/src/aurora/inference.py:223
        for lev in AURORA_PRESSURE_LEVELS:  # src:/home/NWP-Benchmark/src/aurora/inference.py:227
            names.append(f"{var}_{lev}")  # src:/home/NWP-Benchmark/src/aurora/inference.py:229
    return names  # src:/home/NWP-Benchmark/src/aurora/inference.py:231


def _setup_device() -> str:  # src:/home/NWP-Benchmark/src/aurora/inference.py:70-79
    if torch.cuda.is_available():  # src:/home/NWP-Benchmark/src/aurora/inference.py:72
        return "cuda"  # src:/home/NWP-Benchmark/src/aurora/inference.py:75
    return "cpu"  # src:/home/NWP-Benchmark/src/aurora/inference.py:78


def _latlon_721_1440() -> tuple[np.ndarray, np.ndarray]:  # src:/home/NWP-Benchmark/src/aurora/inference.py:130-131
    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)  # src:/home/NWP-Benchmark/src/aurora/inference.py:130 adapted from np.25 coords
    lon = np.linspace(0.0, 360.0, 1440, endpoint=False, dtype=np.float32)  # src:/vepfs-dev/.../run_large_scale.py:88-89
    return lat, lon  # src:/helper


def _load_static_aurora_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # src:/home/NWP-Benchmark/src/aurora/inference.py:117-121 static_vars
    ds = xr.open_dataset(DEFAULT_STATIC_NC)  # src:/home static file path usage
    rename_map = {"valid_time": "time", "latitude": "lat", "longitude": "lon"}  # src:/home/NWP-Benchmark/src/graphcast/prepare.py:114-117 style
    ds = ds.rename({k: v for k, v in rename_map.items() if k in ds.dims or k in ds.coords})  # src:/home prepare rename style
    z = ds["z"].isel(time=0).values if "time" in ds["z"].dims else ds["z"].values  # src:/home/NWP-Benchmark/src/aurora/inference.py:118
    lsm = ds["lsm"].isel(time=0).values if "time" in ds["lsm"].dims else ds["lsm"].values  # src:/home/NWP-Benchmark/src/aurora/inference.py:120
    slt = ds["slt"].isel(time=0).values if "time" in ds["slt"].dims else ds["slt"].values  # src:/home/NWP-Benchmark/src/aurora/inference.py:119
    return z.astype(np.float32), slt.astype(np.float32), lsm.astype(np.float32)  # src:/home static vars pass-through


def _load_batch_from_np25(init_time: datetime, device: str, era5_root: Path) -> Batch:  # src:/home/NWP-Benchmark/src/aurora/inference.py:80-137 adapted to np.25
    layout = Era5NpyLayout(era5_root)  # src:/vepfs data layout
    t_hist = [init_time - timedelta(hours=6), init_time]  # src:/home/NWP-Benchmark/src/aurora/inference.py:109-110
    z_sfc, slt_sfc, lsm_sfc = _load_static_aurora_fields()  # src:/home/NWP-Benchmark/src/aurora/inference.py:117-121

    surf_vars: Dict[str, torch.Tensor] = {}  # src:/home/NWP-Benchmark/src/aurora/inference.py:111
    for name in SURFACE_SHORT_ORDER:  # src:/home/NWP-Benchmark/src/aurora/inference.py:112-116
        arr721 = np.stack([load_npy_2d(layout.single_path(t, name), flip_north_south=False) for t in t_hist], axis=0).astype(np.float32)  # src:/home load two steps
        surf_vars["2t" if name == "t2m" else "10u" if name == "u10" else "10v" if name == "v10" else "msl"] = torch.from_numpy(arr721[None]).float().to(device)  # src:/home/NWP-Benchmark/src/aurora/inference.py:112-116 mapping

    atmos_vars: Dict[str, torch.Tensor] = {}  # src:/home/NWP-Benchmark/src/aurora/inference.py:122
    for name in ATMOS_SHORT_ORDER:  # src:/home/NWP-Benchmark/src/aurora/inference.py:123-127
        arr721 = np.stack([np.stack([load_npy_2d(layout.pressure_path(t, name, float(lev)), flip_north_south=False) for lev in AURORA_PRESSURE_LEVELS], axis=0) for t in t_hist], axis=0).astype(np.float32)  # src:/home level-select behavior
        atmos_vars[name] = torch.from_numpy(arr721[None]).float().to(device)  # src:/home/NWP-Benchmark/src/aurora/inference.py:123-127

    lat, lon = _latlon_721_1440()  # src:/home/NWP-Benchmark/src/aurora/inference.py:130-131
    batch = Batch(  # src:/home/NWP-Benchmark/src/aurora/inference.py:110-135
        surf_vars=surf_vars,  # src:/home/NWP-Benchmark/src/aurora/inference.py:111
        static_vars={"z": torch.from_numpy(z_sfc).float().to(device), "slt": torch.from_numpy(slt_sfc).float().to(device), "lsm": torch.from_numpy(lsm_sfc).float().to(device)},  # src:/home/NWP-Benchmark/src/aurora/inference.py:117-121
        atmos_vars=atmos_vars,  # src:/home/NWP-Benchmark/src/aurora/inference.py:122
        metadata=Metadata(lat=torch.from_numpy(lat).to(device), lon=torch.from_numpy(lon).to(device), time=(init_time,), atmos_levels=tuple(AURORA_PRESSURE_LEVELS)),  # src:/home/NWP-Benchmark/src/aurora/inference.py:129-134; aurora docs: Metadata.time is datetime for current step
    )
    # Aurora docs: surf=(B,T,H,W), static=(H,W), atmos=(B,T,C,H,W), T=2 history steps.
    for v in batch.surf_vars.values():
        if v.ndim != 4 or v.shape[1] != 2:
            raise ValueError(f"Invalid surf_vars shape {tuple(v.shape)}; expected (B,2,H,W)")
    for v in batch.static_vars.values():
        if v.ndim != 2:
            raise ValueError(f"Invalid static_vars shape {tuple(v.shape)}; expected (H,W)")
    for v in batch.atmos_vars.values():
        if v.ndim != 5 or v.shape[1] != 2 or v.shape[2] != len(AURORA_PRESSURE_LEVELS):
            raise ValueError(f"Invalid atmos_vars shape {tuple(v.shape)}; expected (B,2,{len(AURORA_PRESSURE_LEVELS)},H,W)")
    return batch  # src:/return batch


def _load_model(device: str) -> Aurora:  # src:/home/NWP-Benchmark/src/aurora/inference.py:138-165
    weights_path = DEFAULT_WEIGHTS_ROOT / "aurora" / "aurora-0.25-finetuned.ckpt"  # src:/home/NWP-Benchmark/src/aurora/inference.py:142-143 adapted
    if not weights_path.exists():  # src:/home/NWP-Benchmark/src/aurora/inference.py:145
        raise FileNotFoundError(f"Checkpoint missing: {weights_path}")  # src:/home/NWP-Benchmark/src/aurora/inference.py:146
    # IFS finetuned checkpoint contains LoRA adapter weights.
    model = Aurora(use_lora=True)

    def fake_download(*_args, **_kwargs):  # src:/home/NWP-Benchmark/src/aurora/inference.py:152-153
        return str(weights_path)  # src:/home/NWP-Benchmark/src/aurora/inference.py:153

    with patch("aurora.model.aurora.hf_hub_download", side_effect=fake_download):  # src:/home/NWP-Benchmark/src/aurora/inference.py:157
        model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")  # src:/home/NWP-Benchmark/src/aurora/inference.py:158
    return model.to(device).eval()  # src:/home/NWP-Benchmark/src/aurora/inference.py:164


def _pack_output(pred: Batch) -> np.ndarray:  # src:/home/NWP-Benchmark/src/aurora/inference.py:190-247
    data_list: List[np.ndarray] = []  # src:/home/NWP-Benchmark/src/aurora/inference.py:206
    for k in ["2t", "10u", "10v", "msl"]:  # src:/home/NWP-Benchmark/src/aurora/inference.py:211-217 stable order
        data_list.append(pred.surf_vars[k][0, -1, ...].detach().cpu().numpy().astype(np.float32))  # src:/home/NWP-Benchmark/src/aurora/inference.py:215-216
    levels = pred.metadata.atmos_levels  # src:/home/NWP-Benchmark/src/aurora/inference.py:221
    for k in ATMOS_SHORT_ORDER:  # src:/home/NWP-Benchmark/src/aurora/inference.py:223
        val_levels = pred.atmos_vars[k][0, -1, ...].detach().cpu().numpy()  # src:/home/NWP-Benchmark/src/aurora/inference.py:225
        for i, _level in enumerate(levels):  # src:/home/NWP-Benchmark/src/aurora/inference.py:227
            data_list.append(val_levels[i].astype(np.float32))  # src:/home/NWP-Benchmark/src/aurora/inference.py:230
    return np.stack(data_list, axis=0).astype(np.float32)  # src:/home/NWP-Benchmark/src/aurora/inference.py:234


def run_aurora_forecast(init_time: datetime, lead_times_hours: List[int], *, era5_root: Path = DEFAULT_ERA5_NPY_ROOT) -> Dict[int, np.ndarray]:  # src:/vepfs run_large_scale.py:192 contract
    wanted = sorted({int(h) for h in lead_times_hours})  # src:/vepfs runner contract style
    if not wanted:  # src:/contract empty case
        return {}  # src:/contract empty case
    if any(h % 6 != 0 for h in wanted):  # src:/home/NWP-Benchmark/src/aurora/inference.py:289 +6h stepping
        raise ValueError(f"Aurora lead times must be multiples of 6h: {wanted}")  # src:/validation

    device = _setup_device()  # src:/home/NWP-Benchmark/src/aurora/inference.py:268
    batch = _load_batch_from_np25(init_time, device, era5_root)  # src:/home/NWP-Benchmark/src/aurora/inference.py:282 adapted
    model = _load_model(device)  # src:/home/NWP-Benchmark/src/aurora/inference.py:286

    max_steps = max(wanted) // 6  # src:/+6h step conversion
    out: Dict[int, np.ndarray] = {}  # src:/adapter output
    # Must use ``aurora.rollout.rollout``: each step needs a 2×6h history window
    # ``[t-6h, t]`` built as ``cat([batch[:,1:], pred], dim=1)``. Using only
    # ``pred = model.forward(batch)`` as the next ``batch`` leaves T=1 and blows up long leads.
    with torch.inference_mode():  # src:/home/NWP-Benchmark/src/aurora/inference.py:290
        for step, pred_batch in enumerate(rollout(model, batch, max_steps), start=1):
            lead_h = step * 6  # src:/+6h stepping
            if lead_h in wanted:  # src:/requested lead filter
                out[lead_h] = _pack_output(pred_batch)  # src:/home/NWP-Benchmark/src/aurora/inference.py:295 adapted return array
    return out  # src:/vepfs contract


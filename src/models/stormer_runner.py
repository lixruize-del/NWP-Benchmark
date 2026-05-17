"""Stormer runner with explicit path mapping comments."""  # src:/home/NWP-Benchmark/src/stormer/inference.py:1-5

from __future__ import annotations  # src:/home/NWP-Benchmark/src/stormer/inference.py:1

import logging  # src:/home/NWP-Benchmark/src/stormer/inference.py:4
import os  # src:/home/NWP-Benchmark/src/stormer/inference.py:1
import sys  # src:/home/NWP-Benchmark/src/stormer/inference.py:1
from datetime import datetime  # src:/home/NWP-Benchmark/src/stormer/inference.py:1
from pathlib import Path  # src:/home/NWP-Benchmark/src/stormer/inference.py:5
from typing import Dict, List  # src:/home/NWP-Benchmark/src/stormer/inference.py:6

import numpy as np  # src:/home/NWP-Benchmark/src/stormer/inference.py:9
import torch  # src:/home/NWP-Benchmark/src/stormer/inference.py:11
import xarray as xr  # src:/home/NWP-Benchmark/src/stormer/prepare.py:5

from src.common.data_reader import DEFAULT_ERA5_NPY_ROOT, load_stormer_stack  # src:/vepfs.../data_reader.py:239-258
from src.stormer import regridding  # src:/home/NWP-Benchmark/src/stormer/prepare.py:11
from src.stormer.utils.data_utils import CONSTANTS  # src:/home/NWP-Benchmark/src/stormer/utils/data_utils.py:1-20

logger = logging.getLogger(__name__)  # src:/home/NWP-Benchmark/src/stormer/inference.py:25

REPO_ROOT = Path(__file__).resolve().parents[2]  # src:/home/NWP-Benchmark/src/stormer/inference.py:14-16 (adapted path root)
DEFAULT_WEIGHTS_ROOT = Path(os.environ.get("NWP_WEIGHTS_ROOT", "/ecmwf-era5-datasets/nwp_bench/assets/weights"))  # src:/vepfs old runner weights root

if str(REPO_ROOT) not in sys.path:  # src:/home/NWP-Benchmark/src/stormer/inference.py:16
    sys.path.insert(0, str(REPO_ROOT))  # src:/home/NWP-Benchmark/src/stormer/inference.py:16

import src.stormer.inference as stormer_inference  # src:/home/NWP-Benchmark/src/stormer/inference.py:117 import side-effect patches
from src.stormer.inference import build_channel_mapping, ensure_batch_bvhw, load_model, variables  # src:/home/NWP-Benchmark/src/stormer/inference.py:165,189,198,125


def _target_grid() -> tuple[np.ndarray, np.ndarray]:  # src:/home/NWP-Benchmark/src/stormer/prepare.py:56-63
    ddeg_out = 1.40625  # src:/home/NWP-Benchmark/src/stormer/prepare.py:20
    lat_start = -90 + ddeg_out / 2  # src:/home/NWP-Benchmark/src/stormer/prepare.py:57
    lat_stop = 90 - ddeg_out / 2  # src:/home/NWP-Benchmark/src/stormer/prepare.py:58
    n_lat = int(180 / ddeg_out)  # src:/home/NWP-Benchmark/src/stormer/prepare.py:59
    n_lon = int(360 / ddeg_out)  # src:/home/NWP-Benchmark/src/stormer/prepare.py:60
    new_lat = np.linspace(lat_start, lat_stop, num=n_lat, endpoint=True)  # src:/home/NWP-Benchmark/src/stormer/prepare.py:61
    new_lon = np.linspace(0, 360, num=n_lon, endpoint=False)  # src:/home/NWP-Benchmark/src/stormer/prepare.py:62
    return new_lat.astype(np.float32), new_lon.astype(np.float32)  # src:/home/NWP-Benchmark/src/stormer/prepare.py:63


def _build_regridder():  # src:/home/NWP-Benchmark/src/stormer/prepare.py:66-72
    src_lon = np.linspace(0.0, 360.0, 1440, endpoint=False, dtype=np.float32)  # src grid inferred from np.25
    src_lat_desc = np.linspace(90.0, -90.0, 721, dtype=np.float32)  # src grid inferred from np.25
    src_grid = regridding.Grid.from_degrees(lon=src_lon, lat=np.sort(src_lat_desc))  # src:/home/NWP-Benchmark/src/stormer/prepare.py:70
    tgt_lat, tgt_lon = _target_grid()  # src:/home/NWP-Benchmark/src/stormer/prepare.py:69
    tgt_grid = regridding.Grid.from_degrees(lon=tgt_lon, lat=tgt_lat)  # src:/home/NWP-Benchmark/src/stormer/prepare.py:71
    return regridding.ConservativeRegridder(src_grid, tgt_grid), tgt_lat, tgt_lon  # src:/home/NWP-Benchmark/src/stormer/prepare.py:72


_REGRIDDER, _TGT_LAT, _TGT_LON = _build_regridder()  # src:/home/NWP-Benchmark/src/stormer/prepare.py:create_regridder usage


def interpolate_stormer_to_721(
    stack_stormer: np.ndarray,
    *,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
) -> np.ndarray:
    """Resample Stormer native ``(V,128,256)`` to ERA5 0.25° ``(V,721,1440)``.

    Matches :mod:`src.common.era5_eval_regrid`: scipy linear interpolation + periodic
    longitude wrap — **not** JAX conservative regrid (that gave incompatible z500 etc.).

    If ``lat`` / ``lon`` are omitted, uses Stormer training grid from :func:`_target_grid`.
    Pass coordinates read from the forecast NetCDF for strict parity with offline eval.
    """
    from src.common.era5_eval_regrid import stack_native_to_era5_025

    x = np.asarray(stack_stormer, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected (V,H,W), got {x.shape}")
    hs, ws = x.shape[1], x.shape[2]
    if (hs, ws) != (int(_TGT_LAT.shape[0]), int(_TGT_LON.shape[0])):
        raise ValueError(
            f"Expected Stormer native ({_TGT_LAT.shape[0]},{_TGT_LON.shape[0]}), got {(hs, ws)}"
        )

    lat_arr = np.asarray(lat if lat is not None else _target_grid()[0], dtype=np.float64)
    lon_arr = np.asarray(lon if lon is not None else _target_grid()[1], dtype=np.float64)
    return stack_native_to_era5_025(x, lat_arr, lon_arr)


def _interpolate_to_stormer_grid(x: torch.Tensor) -> torch.Tensor:  # src:/home/NWP-Benchmark/src/stormer/prepare.py:75-77
    arr = x.detach().cpu().numpy().astype(np.float32)  # src:/home/NWP-Benchmark/src/stormer/inference.py uses numpy<->torch conversions
    out = np.empty((arr.shape[0], _TGT_LAT.shape[0], _TGT_LON.shape[0]), dtype=np.float32)  # target (V,128,256) per prepare
    src_lat_desc = np.linspace(90.0, -90.0, 721, dtype=np.float32)  # np.25 layout
    src_lon = np.linspace(0.0, 360.0, 1440, endpoint=False, dtype=np.float32)  # np.25 layout
    for i in range(arr.shape[0]):  # per-channel regrid following prepare field extraction
        da = xr.DataArray(arr[i], dims=["lat", "lon"], coords={"lat": src_lat_desc, "lon": src_lon})  # xarray field container
        if da["lat"][0] > da["lat"][-1]:  # src:/home/NWP-Benchmark/src/stormer/regridding.py:68-71 increasing lat required
            da = da.isel(lat=slice(None, None, -1))  # src:/home/NWP-Benchmark/src/stormer/regridding.py:70
        ds = xr.Dataset({"x": da})  # regridder API expects dataset, src:/home/NWP-Benchmark/src/stormer/regridding.py:66
        ds_rg = _REGRIDDER.regrid_dataset(ds).transpose("lat", "lon")  # src:/home/NWP-Benchmark/src/stormer/prepare.py:75-77
        out[i] = ds_rg["x"].values.astype(np.float32)  # extract channel result
    return torch.from_numpy(out)  # back to torch tensor


def _prepare_input_tensor(stack_721: np.ndarray, mean_cpu: torch.Tensor, std_cpu: torch.Tensor) -> torch.Tensor:  # src:/home/NWP-Benchmark/src/stormer/inference.py:318-324
    t = torch.from_numpy(stack_721).to(dtype=torch.float32)  # src:/home/NWP-Benchmark/src/stormer/inference.py:344
    t = _interpolate_to_stormer_grid(t)  # mapped to prepare conservative regrid convention
    return (t - mean_cpu.squeeze(0)) / std_cpu.squeeze(0)  # src:/home/NWP-Benchmark/src/stormer/models/iterative_module.py:173 reverse of denorm


def _load_norm_tensors(device: torch.device):  # src:/home/NWP-Benchmark/src/stormer/inference.py:216-261
    norm_dir = REPO_ROOT / "src" / "stormer" / "normalization_constants"  # src:/home/NWP-Benchmark/src/stormer/inference.py:21
    mean_npz = dict(np.load(norm_dir / "normalize_mean.npz"))  # src:/home/NWP-Benchmark/src/stormer/inference.py:221
    std_npz = dict(np.load(norm_dir / "normalize_std.npz"))  # src:/home/NWP-Benchmark/src/stormer/inference.py:222
    normalize_mean = np.concatenate([mean_npz[v] for v in variables], axis=0).astype(np.float32)  # src:/home/NWP-Benchmark/src/stormer/inference.py:228
    normalize_std = np.concatenate([std_npz[v] for v in variables], axis=0).astype(np.float32)  # src:/home/NWP-Benchmark/src/stormer/inference.py:229
    mean_t = torch.from_numpy(normalize_mean).to(device=device, dtype=torch.float32).view(1, -1, 1, 1)  # src:/home/NWP-Benchmark/src/stormer/inference.py:244
    std_t = torch.from_numpy(normalize_std).to(device=device, dtype=torch.float32).view(1, -1, 1, 1)  # src:/home/NWP-Benchmark/src/stormer/inference.py:245
    diff_std_by_interval: Dict[int, torch.Tensor] = {}  # src:/home/NWP-Benchmark/src/stormer/inference.py:247
    for interval in [6, 12, 24]:  # src:/home/NWP-Benchmark/src/stormer/inference.py:248
        diff_std_npz = dict(np.load(norm_dir / f"normalize_diff_std_{interval}.npz"))  # src:/home/NWP-Benchmark/src/stormer/inference.py:238
        diff_std = np.concatenate([diff_std_npz[v] for v in variables], axis=0).astype(np.float32)  # src:/home/NWP-Benchmark/src/stormer/inference.py:239
        diff_std_by_interval[interval] = torch.from_numpy(diff_std).to(device=device, dtype=torch.float32).view(1, -1, 1, 1)  # src:/home/NWP-Benchmark/src/stormer/inference.py:250-253
    return mean_t, std_t, diff_std_by_interval  # src:/home/NWP-Benchmark/src/stormer/inference.py:261


def _stormer_pad(x: torch.Tensor, patch_size: int):  # src:/home/NWP-Benchmark/src/stormer/inference.py:264-272
    h = x.shape[-2]  # src:/home/NWP-Benchmark/src/stormer/inference.py:265
    if h % patch_size != 0:  # src:/home/NWP-Benchmark/src/stormer/inference.py:266
        pad_size = patch_size - h % patch_size  # src:/home/NWP-Benchmark/src/stormer/inference.py:267
        padded_x = torch.nn.functional.pad(x, (0, 0, pad_size, 0), "constant", 0)  # src:/home/NWP-Benchmark/src/stormer/inference.py:268
    else:  # src:/home/NWP-Benchmark/src/stormer/inference.py:269
        padded_x = x  # src:/home/NWP-Benchmark/src/stormer/inference.py:270
        pad_size = 0  # src:/home/NWP-Benchmark/src/stormer/inference.py:271
    return padded_x, pad_size  # src:/home/NWP-Benchmark/src/stormer/inference.py:272


def _stormer_predict_residual(net, x: torch.Tensor, interval_hours: int) -> torch.Tensor:  # src:/home/NWP-Benchmark/src/stormer/inference.py:275-285
    interval_tensor = torch.tensor([interval_hours], device=x.device, dtype=x.dtype) / 10.0  # src:/home/NWP-Benchmark/src/stormer/inference.py:282-283
    interval_tensor = interval_tensor.repeat(x.shape[0])  # src:/home/NWP-Benchmark/src/stormer/inference.py:283
    padded_x, pad_size = _stormer_pad(x, net.patch_size)  # src:/home/NWP-Benchmark/src/stormer/inference.py:284
    return net(padded_x, variables, interval_tensor)[:, :, pad_size:]  # src:/home/NWP-Benchmark/src/stormer/inference.py:285


def _replace_constant_channels(yhat: torch.Tensor) -> torch.Tensor:  # src:/home/NWP-Benchmark/src/stormer/inference.py:288-292
    for i, name in enumerate(variables):  # src:/home/NWP-Benchmark/src/stormer/inference.py:289
        if name in CONSTANTS:  # src:/home/NWP-Benchmark/src/stormer/inference.py:290
            yhat[:, i] = 0.0  # src:/home/NWP-Benchmark/src/stormer/inference.py:291
    return yhat  # src:/home/NWP-Benchmark/src/stormer/inference.py:292


def _forward_validation_explicit(net, x_norm: torch.Tensor, interval: int, steps: int, mean_t: torch.Tensor, std_t: torch.Tensor, diff_std_t: torch.Tensor) -> torch.Tensor:  # src:/home/NWP-Benchmark/src/stormer/inference.py:295-311
    x = x_norm  # src:/home/NWP-Benchmark/src/stormer/inference.py:305
    for _ in range(steps):  # src:/home/NWP-Benchmark/src/stormer/inference.py:306
        pred_diff_norm = _stormer_predict_residual(net, x, interval)  # src:/home/NWP-Benchmark/src/stormer/inference.py:306
        pred_diff_norm = _replace_constant_channels(pred_diff_norm)  # src:/home/NWP-Benchmark/src/stormer/inference.py:307
        pred_diff_phys = pred_diff_norm * diff_std_t  # src:/home/NWP-Benchmark/src/stormer/inference.py:308
        pred_phys = x * std_t + mean_t + pred_diff_phys  # src:/home/NWP-Benchmark/src/stormer/inference.py:309
        x = (pred_phys - mean_t) / std_t  # src:/home/NWP-Benchmark/src/stormer/inference.py:310
    return x  # src:/home/NWP-Benchmark/src/stormer/inference.py:311


def run_stormer_forecast(init_time: datetime, lead_times_hours: List[int], *, era5_root: Path = DEFAULT_ERA5_NPY_ROOT, weights_ckpt: Path | None = None, list_intervals: List[int] | None = None) -> Dict[int, np.ndarray]:  # src:/home/NWP-Benchmark/src/stormer/inference.py:314
    intervals = list_intervals or [6, 12, 24]  # src:/home/NWP-Benchmark/src/stormer/inference.py:314
    # Only warn when no step size in ``intervals`` divides ``lead`` (reachable lead times).
    for lead in lead_times_hours:
        if not any(lead % it == 0 for it in intervals):
            logger.warning(
                "lead %sh is not divisible by any interval in %s — will fail at rollout",
                lead,
                intervals,
            )
    ckpt = weights_ckpt or (DEFAULT_WEIGHTS_ROOT / "stormer" / "stormer_1.40625_patch_size_2.ckpt")  # src:/home/NWP-Benchmark/src/stormer/inference.py:19
    stormer_inference.WEIGHTS_FILE = Path(ckpt)  # src:/home/NWP-Benchmark/src/stormer/inference.py:19 symbol override
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # src:/home/NWP-Benchmark/src/stormer/inference.py:315
    model = load_model(device)  # src:/home/NWP-Benchmark/src/stormer/inference.py:316
    mean_t, std_t, diff_std_by_interval = _load_norm_tensors(device)  # src:/home/NWP-Benchmark/src/stormer/inference.py:318
    mean_cpu = mean_t.detach().cpu()  # follows explicit CPU preprocessing path
    std_cpu = std_t.detach().cpu()  # follows explicit CPU preprocessing path
    raw = load_stormer_stack(init_time, root=era5_root, flip_north_south=False)  # src:/home/NWP-Benchmark/src/stormer/prepare.py lat already north->south on np.25
    inp_b = ensure_batch_bvhw(_prepare_input_tensor(raw, mean_cpu, std_cpu)).to(device=device, dtype=torch.float32)  # src:/home/NWP-Benchmark/src/stormer/inference.py:344
    out: Dict[int, np.ndarray] = {}  # src:/home/NWP-Benchmark/src/stormer/inference.py:346
    wanted = sorted({int(h) for h in lead_times_hours})  # src:/home/NWP-Benchmark/src/stormer/inference.py:347 sorted loop
    with torch.no_grad():  # src:/home/NWP-Benchmark/src/stormer/inference.py:352
        n_wanted = len(wanted)
        for li, lead in enumerate(wanted):  # src:/home/NWP-Benchmark/src/stormer/inference.py:347
            logger.info(
                "Stormer lead %sh (%d/%d), intervals=%s — forward may take minutes on CPU",
                lead,
                li + 1,
                n_wanted,
                intervals,
            )
            preds = []  # src:/home/NWP-Benchmark/src/stormer/inference.py:348
            for interval in intervals:  # src:/home/NWP-Benchmark/src/stormer/inference.py:349
                if lead % interval != 0:  # src:/home/NWP-Benchmark/src/stormer/inference.py:350
                    continue  # src:/home/NWP-Benchmark/src/stormer/inference.py:350
                steps = lead // interval  # src:/home/NWP-Benchmark/src/stormer/inference.py:351
                pred_norm = _forward_validation_explicit(model.net, inp_b, interval, steps, mean_t, std_t, diff_std_by_interval[interval])  # src:/home/NWP-Benchmark/src/stormer/inference.py:353-362
                preds.append(pred_norm)  # src:/home/NWP-Benchmark/src/stormer/inference.py:364
            if not preds:  # src:/home/NWP-Benchmark/src/stormer/inference.py:365
                raise RuntimeError(f"No valid interval divides lead {lead} with intervals={intervals}")  # src:/home/NWP-Benchmark/src/stormer/inference.py:366
            mean_pred_norm = torch.stack(preds, dim=0).mean(0)  # src:/home/NWP-Benchmark/src/stormer/inference.py:369
            denorm = (mean_pred_norm * std_t + mean_t).squeeze(0).detach().cpu().numpy().astype(np.float32)  # src:/home/NWP-Benchmark/src/stormer/inference.py:371-372
            out[lead] = denorm  # src:/home/NWP-Benchmark/src/stormer/inference.py:372
    return out  # src:/home/NWP-Benchmark/src/stormer/inference.py:388 analogous return path


def stormer_channel_names() -> List[str]:  # src:/home/NWP-Benchmark/src/stormer/inference.py:165-176
    return build_channel_mapping(variables)  # src:/home/NWP-Benchmark/src/stormer/inference.py:165-176


def interpolate_721_to_stormer(stack721: np.ndarray) -> np.ndarray:  # src:/home/NWP-Benchmark/src/stormer/prepare.py:75-77 mapping
    t = torch.from_numpy(stack721.astype(np.float32))  # tensor wrapping for reusable interpolation
    return _interpolate_to_stormer_grid(t).detach().cpu().numpy().astype(np.float32)  # output for GT path

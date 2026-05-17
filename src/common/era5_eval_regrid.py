"""ERA5 0.25° evaluation grid regridding — canonical benchmark definition (AgentCast-era parity).

Stormer and other offline paths use this module so metrics agree on:
  • scipy ``RegularGridInterpolator`` (linear), not JAX conservative regrid
  • periodic longitude wrap when native lon does not reach the last ERA5 column
  • output rows ordered 90°N → 90°S (same as raw ERA5 ``*.npy`` layout)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

_EVAL_H, _EVAL_W = 721, 1440
# N→S rows on 721×1440 eval grid
_EVAL_LAT = np.linspace(90.0, -90.0, _EVAL_H, dtype=np.float32)
_EVAL_LON = np.linspace(0.0, 359.75, _EVAL_W, dtype=np.float32)

# Mean-normalised cosine latitude weights on eval grid
_W1D_EVAL: np.ndarray = (
    np.cos(np.deg2rad(_EVAL_LAT)).clip(0.0).astype(np.float64)
)
_W1D_EVAL /= _W1D_EVAL.mean() + 1e-12

_INTERP_PTS_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def eval_lon_grid_1440() -> np.ndarray:
    """Longitude centers for 0.25° global grid (0 … 359.75°)."""
    return np.linspace(0.0, 359.75, _EVAL_W, dtype=np.float32)


def regrid_to_era5_025(
    arr: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
) -> np.ndarray | None:
    """Resample a single (H, W) field to ERA5 native 721×1440 (N→S lat).

    Regrid rules (legacy standalone eval parity):
      • 721×1440 with lat[0] > lat[-1] → identity; else flip rows
      • else scipy bilinear + periodic lon wrap
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D field, got {arr.shape}")

    H, W = arr.shape

    if H == _EVAL_H and W == _EVAL_W:
        arr32 = arr.astype(np.float32)
        return arr32 if float(src_lat[0]) > float(src_lat[-1]) else np.ascontiguousarray(arr32[::-1, :])

    if H == _EVAL_H - 1 and W == _EVAL_W:
        arr32 = arr.astype(np.float32)
        if float(src_lat[0]) > float(src_lat[-1]):
            return np.concatenate([arr32, arr32[-1:, :]], axis=0)
        return np.concatenate([arr32[:1, :], arr32], axis=0)[::-1, :]

    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError:
        return None

    lat = np.asarray(src_lat, dtype=np.float64)
    lon = np.asarray(src_lon, dtype=np.float64)
    data = np.asarray(arr, dtype=np.float64)

    if lat[0] > lat[-1]:
        lat = lat[::-1]
        data = data[::-1, :]

    if lon[-1] < float(_EVAL_LON[-1]):
        lon = np.append(lon, 360.0)
        data = np.concatenate([data, data[:, :1]], axis=1)

    key = (H, W)
    if key not in _INTERP_PTS_CACHE:
        tgt_lat = np.clip(_EVAL_LAT[::-1].astype(np.float64), lat[0], lat[-1])
        tgt_lon = _EVAL_LON.astype(np.float64)
        pts_lon, pts_lat = np.meshgrid(tgt_lon, tgt_lat)
        _INTERP_PTS_CACHE[key] = np.stack([pts_lat.ravel(), pts_lon.ravel()], axis=1)

    interp = RegularGridInterpolator(
        (lat, lon),
        data,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    out = interp(_INTERP_PTS_CACHE[key]).reshape(_EVAL_H, _EVAL_W).astype(np.float32)[::-1, :]
    return np.ascontiguousarray(out)


def wrmse_721_eval_style(pred_hw: np.ndarray, gt_hw: np.ndarray) -> float:
    """Latitude-weighted RMSE on 721×1440 (float64 diff, cosine weights / mean)."""
    if pred_hw.shape != (_EVAL_H, _EVAL_W) or gt_hw.shape != (_EVAL_H, _EVAL_W):
        raise ValueError(
            f"Expected ({_EVAL_H}, {_EVAL_W}) arrays, got pred={pred_hw.shape} gt={gt_hw.shape}"
        )
    d = pred_hw.astype(np.float64) - gt_hw.astype(np.float64)
    return float(np.sqrt((_W1D_EVAL[:, None] * (d**2)).mean()))


def wrmse_stack_eval_style(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-channel WRMSE for ``(C,721,1440)`` stacks (:func:`wrmse_721_eval_style` per channel)."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError(f"Expected matching (C,H,W), got pred={pred.shape} gt={gt.shape}")
    c = int(pred.shape[0])
    out = np.empty(c, dtype=np.float64)
    for i in range(c):
        out[i] = wrmse_721_eval_style(pred[i], gt[i])
    return out


def stack_native_to_era5_025(
    stack: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
) -> np.ndarray:
    """``(V,H,W)`` → ``(V,721,1440)`` using :func:`regrid_to_era5_025`."""
    x = np.asarray(stack, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected (V,H,W), got {x.shape}")
    v = x.shape[0]
    out = np.empty((v, _EVAL_H, _EVAL_W), dtype=np.float32)
    for i in range(v):
        rg = regrid_to_era5_025(x[i], src_lat, src_lon)
        if rg is None:
            raise RuntimeError("regrid_to_era5_025 failed (install scipy?)")
        out[i] = rg
    return out


# Models whose inference output is not ERA5 0.25°; metrics compare on 721×1440 after regrid.
NATIVE_GRID_EVAL_MODELS = frozenset({"stormer", "aurora", "neuralgcm"})


def native_lat_lon_for_pred(model: str, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Latitude/longitude centers for a model forecast field shape ``(height, width)``."""
    m = model.lower()
    if m == "stormer":
        if (height, width) != (128, 256):
            raise ValueError(f"Stormer expected (128, 256), got ({height}, {width})")
        ddeg = 1.40625
        lat = np.linspace(-90.0 + ddeg / 2.0, 90.0 - ddeg / 2.0, 128, dtype=np.float64)
        lon = np.linspace(0.0, 360.0, 256, endpoint=False, dtype=np.float64)
        return lat, lon
    if m == "aurora":
        if width != 1440 or height not in (720, 721):
            raise ValueError(f"Aurora expected H=720|721 and W=1440, got ({height}, {width})")
        if height == 720:
            lat = np.linspace(90.0, -90.0, 721, dtype=np.float64)[:-1]
        else:
            lat = np.linspace(90.0, -90.0, 721, dtype=np.float64)
        lon = np.linspace(0.0, 360.0, 1440, endpoint=False, dtype=np.float64)
        return lat, lon
    if m == "neuralgcm":
        lat = np.linspace(90.0, -90.0, height, dtype=np.float64)
        lon = np.linspace(0.0, 360.0, width, endpoint=False, dtype=np.float64)
        return lat, lon
    raise ValueError(f"Not a native-grid eval model: {model}")


def regrid_model_pred_eval_to_era5_025(
    model: str,
    pred_eval: np.ndarray,
    *,
    src_lat: np.ndarray | None = None,
    src_lon: np.ndarray | None = None,
) -> np.ndarray:
    """Resample ``(C,H,W)`` model forecast to ERA5 0.25° ``(C,721,1440)`` for metrics."""
    x = np.asarray(pred_eval, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {x.shape}")
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if (h, w) == (_EVAL_H, _EVAL_W):
        return np.ascontiguousarray(x)

    m = model.lower()
    if m == "stormer":
        from src.models.stormer_runner import interpolate_stormer_to_721

        if src_lat is None or src_lon is None:
            src_lat, src_lon = native_lat_lon_for_pred(m, h, w)
        return interpolate_stormer_to_721(x, lat=src_lat, lon=src_lon)

    if src_lat is None or src_lon is None:
        src_lat, src_lon = native_lat_lon_for_pred(m, h, w)
    return stack_native_to_era5_025(x, src_lat, src_lon)

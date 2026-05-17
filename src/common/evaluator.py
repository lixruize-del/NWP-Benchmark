"""
src/common/evaluator.py
=======================
Latitude-weighted deterministic forecast metrics.

Implemented metrics
-------------------
- WRMSE   : Weighted Root Mean Square Error
- BIAS    : Weighted mean bias (pred - gt)
- MAE     : Weighted Mean Absolute Error
- Activity: Placeholder for now (returns NaN by request).
- ACC     : Anomaly Correlation Coefficient (placeholder – requires climatology).
            Returns NaN until climatology_path is supplied by the user.

All metrics operate on single-variable 2-D spatial fields (H, W) and return a
scalar float.  The public function `compute_all_metrics` accepts arrays of shape
(V, H, W) and returns a dict whose values are 1-D arrays of length V.

Latitude weighting
------------------
Following earth2studio (NVIDIA) and WeatherBench2:
    w[i] = cos(lat[i] * π / 180)
    w    = w / mean(w)          <- normalised so weights sum to H

References
----------
- NVIDIA earth2studio statistics/weights.py : lat_weight formula
- WeatherBench2 metrics.py : weighted_mean_bias, weighted_rmse, acc
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Latitude weights
# ---------------------------------------------------------------------------

def lat_weight(lat: np.ndarray) -> np.ndarray:
    """
    Compute normalised cosine latitude weights.

    Parameters
    ----------
    lat : np.ndarray, shape (H,)
        Latitude values in degrees, typically 90 → -90 (north to south).

    Returns
    -------
    np.ndarray, shape (H,)
        Non-negative weights whose mean equals 1.0.
    """
    w = np.cos(np.deg2rad(lat)).astype(np.float64)
    w = np.clip(w, 0.0, None)   # guard against tiny negatives at poles
    w = w / w.mean()
    return w


# ---------------------------------------------------------------------------
# Core metric functions  (operate on a single variable field)
# ---------------------------------------------------------------------------

def _weighted_mean(x: np.ndarray, w2d: np.ndarray) -> float:
    """Weighted spatial mean of a (H, W) field."""
    return float(np.sum(w2d * x) / np.sum(w2d))


def _weighted_std(x: np.ndarray, w2d: np.ndarray) -> float:
    """Weighted spatial standard deviation of a (H, W) field."""
    mu = _weighted_mean(x, w2d)
    var = _weighted_mean((x - mu) ** 2, w2d)
    return float(np.sqrt(max(var, 0.0)))


def wrmse(pred: np.ndarray, gt: np.ndarray, w2d: np.ndarray) -> float:
    """
    Weighted Root Mean Square Error.

    Parameters
    ----------
    pred, gt : np.ndarray, shape (H, W)
    w2d      : np.ndarray, shape (H, W) – pre-broadcast latitude weights

    Returns
    -------
    float
    """
    return float(np.sqrt(_weighted_mean((pred - gt) ** 2, w2d)))


def bias(pred: np.ndarray, gt: np.ndarray, w2d: np.ndarray) -> float:
    """
    Weighted mean bias: mean(w * (pred - gt)).

    Positive values indicate systematic over-prediction.
    """
    return _weighted_mean(pred - gt, w2d)


def mae(pred: np.ndarray, gt: np.ndarray, w2d: np.ndarray) -> float:
    """Weighted Mean Absolute Error."""
    return _weighted_mean(np.abs(pred - gt), w2d)


def activity(pred: np.ndarray, gt: np.ndarray, w2d: np.ndarray) -> float:
    """
    Placeholder metric for Activity.

    Per current project decision, Activity is intentionally deferred and should
    not influence model ranking yet. This function returns NaN for every call.
    """
    _ = (pred, gt, w2d)
    return float("nan")


def acc(
    pred: np.ndarray,
    gt: np.ndarray,
    clim: np.ndarray,
    w2d: np.ndarray,
) -> float:
    """
    Anomaly Correlation Coefficient.

    ACC = weighted_corr(pred - clim, gt - clim)

    Parameters
    ----------
    pred, gt, clim : np.ndarray, shape (H, W)
        All three must be on the same grid.
    w2d : np.ndarray, shape (H, W)

    Returns
    -------
    float in [-1, 1].  Returns NaN if denominator is zero.
    """
    fa = (pred - clim).astype(np.float64)
    oa = (gt   - clim).astype(np.float64)

    num   = _weighted_mean(fa * oa, w2d)
    denom = np.sqrt(_weighted_mean(fa ** 2, w2d) * _weighted_mean(oa ** 2, w2d))

    if denom < 1e-12:
        return float("nan")
    return float(num / denom)


# ---------------------------------------------------------------------------
# Batch interface: operate over (V, H, W) arrays
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    lat: np.ndarray,
    climatology: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """
    Compute all implemented metrics for every variable channel.

    Parameters
    ----------
    pred : np.ndarray, shape (V, H, W)
        Denormalised model forecast.
    gt : np.ndarray, shape (V, H, W)
        Ground-truth ERA5 field on the same grid.
    lat : np.ndarray, shape (H,)
        Latitude values in degrees (north-to-south, e.g., 90 → -90).
    climatology : np.ndarray or None, shape (V, H, W)
        Climatological mean for the valid time.  Required for ACC.
        If None, ACC values will be NaN.

    Returns
    -------
    dict with keys 'wrmse', 'bias', 'mae', 'activity', 'acc', each mapping to
    a np.ndarray of shape (V,).

    Notes
    -----
    - All inputs must be on the same spatial grid.
    - Latitude must be monotonically decreasing (standard north-to-south).
    """
    assert pred.shape == gt.shape, (
        f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}"
    )
    V, H, W = pred.shape
    assert lat.shape == (H,), f"lat must have shape ({H},), got {lat.shape}"

    # Build (H, W) weight array
    w = lat_weight(lat)             # (H,)
    w2d = w[:, None] * np.ones((1, W), dtype=np.float64)   # (H, W)

    pred_f = pred.astype(np.float64)
    gt_f   = gt.astype(np.float64)

    out = {
        "wrmse":    np.empty(V, dtype=np.float64),
        "bias":     np.empty(V, dtype=np.float64),
        "mae":      np.empty(V, dtype=np.float64),
        "activity": np.empty(V, dtype=np.float64),
        "acc":      np.full(V, np.nan, dtype=np.float64),
    }

    for v in range(V):
        p, g = pred_f[v], gt_f[v]
        out["wrmse"][v]    = wrmse(p, g, w2d)
        out["bias"][v]     = bias(p, g, w2d)
        out["mae"][v]      = mae(p, g, w2d)
        out["activity"][v] = activity(p, g, w2d)

        if climatology is not None:
            c = climatology[v].astype(np.float64)
            out["acc"][v] = acc(p, g, c, w2d)

    return out


# ---------------------------------------------------------------------------
# Quick unit test (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)

    H, W, V = 721, 1440, 5
    lat = np.linspace(90.0, -90.0, H)

    gt = rng.standard_normal((V, H, W)).astype(np.float32)
    pred = gt.copy()
    res = compute_all_metrics(pred, gt, lat)
    assert np.allclose(res["wrmse"], 0.0, atol=1e-6), res["wrmse"]
    assert np.allclose(res["bias"], 0.0, atol=1e-6), res["bias"]
    assert np.allclose(res["mae"], 0.0, atol=1e-6), res["mae"]
    assert np.all(np.isnan(res["activity"])), res["activity"]
    print("Test 1 passed: perfect forecast -> WRMSE=BIAS=MAE=0, Activity=NaN placeholder")

    pred2 = np.zeros((V, H, W), dtype=np.float32)
    gt2 = rng.standard_normal((V, H, W)).astype(np.float32)
    res2 = compute_all_metrics(pred2, gt2, lat)
    assert np.all(np.isnan(res2["activity"])), res2["activity"]
    print("Test 2 passed: Activity placeholder returns NaN")

    clim = rng.standard_normal((V, H, W)).astype(np.float32)
    pred3 = gt.copy()
    res3 = compute_all_metrics(pred3, gt, lat, climatology=clim)
    assert np.allclose(res3["acc"], 1.0, atol=1e-6), res3["acc"]
    print("Test 3 passed: perfect forecast -> ACC=1")

    print("All evaluator unit tests passed.")
    sys.exit(0)

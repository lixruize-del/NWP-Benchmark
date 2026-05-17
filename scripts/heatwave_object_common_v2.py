#!/usr/bin/env python3
"""Core object-based heatwave matching utilities (v2).

This module implements the algorithmic core requested for heatwave event
verification on 1D daily series:
1) Extract contiguous events with minimum duration.
2) Build Temporal-IoU candidates across GT/Pred events.
3) Perform IoU-descending greedy 1-to-1 matching.
4) Return TP/FP/FN with strict count conservation.
5) Derive ML + meteorological metrics with explicit NaN/0 edge semantics.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    import numba as nb

    NUMBA_AVAILABLE = True
    _njit = nb.njit
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def _njit(*args, **kwargs):  # type: ignore
        def _decorator(func):
            return func

        return _decorator


@_njit(cache=True)
def _extract_event_spans_numba(mask_1d: np.ndarray, min_duration_days: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return [start, end] spans (inclusive) for contiguous True runs."""
    n = mask_1d.shape[0]
    starts = np.empty(n, dtype=np.int32)
    ends = np.empty(n, dtype=np.int32)
    k = 0
    i = 0

    while i < n:
        if mask_1d[i]:
            s = i
            while i + 1 < n and mask_1d[i + 1]:
                i += 1
            e = i
            if (e - s + 1) >= min_duration_days:
                starts[k] = s
                ends[k] = e
                k += 1
        i += 1

    return starts[:k], ends[:k]


@_njit(cache=True)
def _temporal_iou_from_spans(s1: int, e1: int, s2: int, e2: int) -> float:
    """Compute Temporal IoU for two inclusive spans."""
    inter = min(e1, e2) - max(s1, s2) + 1
    if inter <= 0:
        return 0.0
    len1 = e1 - s1 + 1
    len2 = e2 - s2 + 1
    union = len1 + len2 - inter
    if union <= 0:
        return 0.0
    return inter / union


@_njit(cache=True)
def _match_event_spans_greedy_numba(
    gt_starts: np.ndarray,
    gt_ends: np.ndarray,
    pred_starts: np.ndarray,
    pred_ends: np.ndarray,
    iou_threshold: float,
) -> Tuple[int, int, int, int, int]:
    """Greedy 1-to-1 matching by descending IoU."""
    gt_total = gt_starts.shape[0]
    pred_total = pred_starts.shape[0]

    if gt_total == 0 and pred_total == 0:
        return 0, 0, 0, 0, 0

    # Worst-case number of candidate pairs.
    max_pairs = gt_total * pred_total
    cand_iou = np.empty(max_pairs, dtype=np.float64)
    cand_gt = np.empty(max_pairs, dtype=np.int32)
    cand_pred = np.empty(max_pairs, dtype=np.int32)
    m = 0

    for g in range(gt_total):
        gs = int(gt_starts[g])
        ge = int(gt_ends[g])
        for p in range(pred_total):
            ps = int(pred_starts[p])
            pe = int(pred_ends[p])
            iou = _temporal_iou_from_spans(gs, ge, ps, pe)
            if iou >= iou_threshold:
                cand_iou[m] = iou
                cand_gt[m] = g
                cand_pred[m] = p
                m += 1

    used_gt = np.zeros(gt_total, dtype=np.uint8)
    used_pred = np.zeros(pred_total, dtype=np.uint8)
    tp = 0

    if m > 0:
        order = np.argsort(cand_iou[:m])[::-1]  # descending IoU
        for kk in range(order.shape[0]):
            j = int(order[kk])
            g = int(cand_gt[j])
            p = int(cand_pred[j])
            if used_gt[g] == 0 and used_pred[p] == 0:
                used_gt[g] = 1
                used_pred[p] = 1
                tp += 1

    fn = gt_total - tp
    fp = pred_total - tp
    return tp, fp, fn, gt_total, pred_total


@_njit(cache=True)
def _build_event_id_series_numba(mask_1d: np.ndarray, min_duration_days: int) -> np.ndarray:
    """Build 1D event-id series (0=no event, 1..N for kept events)."""
    n = mask_1d.shape[0]
    out = np.zeros(n, dtype=np.int32)
    starts, ends = _extract_event_spans_numba(mask_1d, min_duration_days)
    for k in range(starts.shape[0]):
        s = int(starts[k])
        e = int(ends[k])
        eid = k + 1
        for i in range(s, e + 1):
            out[i] = eid
    return out


@_njit(cache=True)
def _match_temporal_iou_batch_numba(
    gt_mask_2d: np.ndarray,
    pred_mask_2d: np.ndarray,
    min_duration_days: int,
    iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batch match over all points.

    Inputs:
      gt_mask_2d, pred_mask_2d: shape (time, n_points), bool
    Returns:
      tp, fp, fn, gt_total, pred_total: int32 arrays with shape (n_points,)
    """
    n_points = gt_mask_2d.shape[1]
    tp = np.zeros(n_points, dtype=np.int32)
    fp = np.zeros(n_points, dtype=np.int32)
    fn = np.zeros(n_points, dtype=np.int32)
    gt_total = np.zeros(n_points, dtype=np.int32)
    pred_total = np.zeros(n_points, dtype=np.int32)

    for p in range(n_points):
        gt_col = gt_mask_2d[:, p]
        pred_col = pred_mask_2d[:, p]
        gt_starts, gt_ends = _extract_event_spans_numba(gt_col, min_duration_days)
        pred_starts, pred_ends = _extract_event_spans_numba(pred_col, min_duration_days)
        _tp, _fp, _fn, _gt, _pred = _match_event_spans_greedy_numba(
            gt_starts, gt_ends, pred_starts, pred_ends, iou_threshold
        )
        tp[p] = _tp
        fp[p] = _fp
        fn[p] = _fn
        gt_total[p] = _gt
        pred_total[p] = _pred

    return tp, fp, fn, gt_total, pred_total


def extract_event_spans(mask_1d: np.ndarray, min_duration_days: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Public wrapper for contiguous event extraction on 1D bool series."""
    if min_duration_days <= 0:
        raise ValueError("min_duration_days must be positive")
    arr = np.asarray(mask_1d, dtype=np.bool_)
    if arr.ndim != 1:
        raise ValueError(f"mask_1d must be 1D, got shape={arr.shape}")
    return _extract_event_spans_numba(arr, int(min_duration_days))


def match_temporal_iou_greedy_1d(
    gt_mask_1d: np.ndarray,
    pred_mask_1d: np.ndarray,
    *,
    min_duration_days: int = 3,
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int, int, int]:
    """Match GT/Pred events at one grid point and return TP/FP/FN.

    Returns:
      (tp, fp, fn, gt_total, pred_total)
    """
    if min_duration_days <= 0:
        raise ValueError("min_duration_days must be positive")
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be in [0, 1]")

    gt = np.asarray(gt_mask_1d, dtype=np.bool_)
    pred = np.asarray(pred_mask_1d, dtype=np.bool_)
    if gt.ndim != 1 or pred.ndim != 1:
        raise ValueError(f"gt/pred must be 1D; got gt={gt.shape}, pred={pred.shape}")
    if gt.shape[0] != pred.shape[0]:
        raise ValueError(f"gt/pred length mismatch: {gt.shape[0]} vs {pred.shape[0]}")

    gt_starts, gt_ends = _extract_event_spans_numba(gt, int(min_duration_days))
    pred_starts, pred_ends = _extract_event_spans_numba(pred, int(min_duration_days))

    tp, fp, fn, gt_total, pred_total = _match_event_spans_greedy_numba(
        gt_starts, gt_ends, pred_starts, pred_ends, float(iou_threshold)
    )

    # Conservation checks required by the evaluation contract.
    if tp + fn != gt_total:
        raise RuntimeError("Conservation broken: TP + FN != GT_total")
    if tp + fp != pred_total:
        raise RuntimeError("Conservation broken: TP + FP != Pred_total")

    return int(tp), int(fp), int(fn), int(gt_total), int(pred_total)


def build_event_id_series_1d(mask_1d: np.ndarray, min_duration_days: int = 3) -> np.ndarray:
    """Return event-id sequence for one 1D mask (0=no-event, 1..N=event id)."""
    if min_duration_days <= 0:
        raise ValueError("min_duration_days must be positive")
    arr = np.asarray(mask_1d, dtype=np.bool_)
    if arr.ndim != 1:
        raise ValueError(f"mask_1d must be 1D, got shape={arr.shape}")
    return _build_event_id_series_numba(arr, int(min_duration_days))


def match_temporal_iou_greedy_batch(
    gt_mask_2d: np.ndarray,
    pred_mask_2d: np.ndarray,
    *,
    min_duration_days: int = 3,
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batch greedy matching over all points.

    Inputs:
      gt_mask_2d, pred_mask_2d: (time, n_points) bool arrays
    Returns:
      tp, fp, fn, gt_total, pred_total: int32 arrays of shape (n_points,)
    """
    if min_duration_days <= 0:
        raise ValueError("min_duration_days must be positive")
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be in [0, 1]")

    gt = np.asarray(gt_mask_2d, dtype=np.bool_)
    pred = np.asarray(pred_mask_2d, dtype=np.bool_)
    if gt.ndim != 2 or pred.ndim != 2:
        raise ValueError(f"gt/pred must be 2D (time, n_points); got gt={gt.shape}, pred={pred.shape}")
    if gt.shape != pred.shape:
        raise ValueError(f"gt/pred shape mismatch: {gt.shape} vs {pred.shape}")

    tp, fp, fn, gt_total, pred_total = _match_temporal_iou_batch_numba(
        gt, pred, int(min_duration_days), float(iou_threshold)
    )
    # Global conservation checks
    if int(tp.sum() + fn.sum()) != int(gt_total.sum()):
        raise RuntimeError("Conservation broken in batch: sum(TP)+sum(FN) != sum(GT_total)")
    if int(tp.sum() + fp.sum()) != int(pred_total.sum()):
        raise RuntimeError("Conservation broken in batch: sum(TP)+sum(FP) != sum(Pred_total)")
    return tp, fp, fn, gt_total, pred_total


def metrics_from_event_counts(
    tp: int,
    fp: int,
    fn: int,
    gt_total: int,
    pred_total: int,
) -> Dict[str, float]:
    """Compute P/R/F1 and FAR/POD/CSI with strict edge-case semantics."""
    tp_f = float(tp)
    fp_f = float(fp)
    fn_f = float(fn)
    gt_f = float(gt_total)
    pred_f = float(pred_total)

    out: Dict[str, float] = {
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "far": float("nan"),
        "pod": float("nan"),
        "csi": float("nan"),
    }

    # Case 1: no GT and no Pred -> all NaN
    if gt_total == 0 and pred_total == 0:
        return out

    # Case 2: no GT, but Pred exists
    if gt_total == 0 and pred_total > 0:
        out["precision"] = 0.0
        out["far"] = 1.0
        out["csi"] = 0.0
        return out

    # Case 3: GT exists, but no Pred
    if gt_total > 0 and pred_total == 0:
        out["recall"] = 0.0
        out["pod"] = 0.0
        out["csi"] = 0.0
        return out

    # General case: both GT and Pred exist
    denom_pr = tp_f + fp_f
    denom_re = tp_f + fn_f
    denom_f1 = 2.0 * tp_f + fp_f + fn_f
    denom_csi = tp_f + fp_f + fn_f

    out["precision"] = tp_f / denom_pr if denom_pr > 0 else float("nan")
    out["recall"] = tp_f / denom_re if denom_re > 0 else float("nan")
    out["pod"] = out["recall"]
    out["far"] = fp_f / denom_pr if denom_pr > 0 else float("nan")
    out["f1"] = 2.0 * tp_f / denom_f1 if denom_f1 > 0 else float("nan")
    out["csi"] = tp_f / denom_csi if denom_csi > 0 else float("nan")

    # Additional consistency guardrails.
    if not np.isclose(tp_f + fn_f, gt_f):
        raise RuntimeError("Conservation broken: TP + FN != GT_total")
    if not np.isclose(tp_f + fp_f, pred_f):
        raise RuntimeError("Conservation broken: TP + FP != Pred_total")

    return out


"""Stormer v2 runner with class-based runtime cache."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from src.common.data_reader import DEFAULT_ERA5_NPY_ROOT, load_stormer_stack
import src.models.stormer_runner as base

logger = logging.getLogger(__name__)

_STORMER_ATTN_PATCHED = False


def _patch_stormer_attention_for_cpu() -> None:
    """Patch xformers attention call to a torch SDPA fallback on CPU."""
    global _STORMER_ATTN_PATCHED
    if _STORMER_ATTN_PATCHED:
        return
    import src.stormer.models.hub.stormer as stormer_hub

    def _sdpa_fallback(q, k, v, attn_bias=None):
        # xformers format: [B, M, H, K] -> torch sdpa: [B, H, M, K]
        del attn_bias
        qh = q.permute(0, 2, 1, 3).contiguous()
        kh = k.permute(0, 2, 1, 3).contiguous()
        vh = v.permute(0, 2, 1, 3).contiguous()
        oh = F.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        return oh.permute(0, 2, 1, 3).contiguous()

    stormer_hub.memory_efficient_attention = _sdpa_fallback
    _STORMER_ATTN_PATCHED = True
    logger.info("Stormer(v2) patched attention: xformers -> torch SDPA fallback")


class StormerForecastRunnerV2:
    """Class-style Stormer runner that reuses model + normalization tensors."""

    def __init__(
        self,
        *,
        era5_root: Path = DEFAULT_ERA5_NPY_ROOT,
        weights_ckpt: Path | None = None,
        list_intervals: List[int] | None = None,
    ) -> None:
        self.era5_root = era5_root
        self.weights_ckpt = weights_ckpt
        self.list_intervals = list_intervals or [6]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._mean_t: torch.Tensor | None = None
        self._std_t: torch.Tensor | None = None
        self._diff_std_by_interval: Dict[int, torch.Tensor] | None = None
        self._mean_cpu: torch.Tensor | None = None
        self._std_cpu: torch.Tensor | None = None

    def _ensure_runtime(self) -> None:
        if self._model is not None:
            return
        if self.device.type == "cpu":
            _patch_stormer_attention_for_cpu()
        ckpt = self.weights_ckpt or (
            base.DEFAULT_WEIGHTS_ROOT / "stormer" / "stormer_1.40625_patch_size_2.ckpt"
        )
        base.stormer_inference.WEIGHTS_FILE = Path(ckpt)

        t0 = time.perf_counter()
        self._model = base.load_model(self.device)
        mean_t, std_t, diff_std_by_interval = base._load_norm_tensors(self.device)
        self._mean_t = mean_t
        self._std_t = std_t
        self._diff_std_by_interval = diff_std_by_interval
        self._mean_cpu = mean_t.detach().cpu()
        self._std_cpu = std_t.detach().cpu()
        logger.info(
            "Stormer(v2) runtime initialized ckpt=%s device=%s took %.3fs",
            ckpt,
            self.device,
            time.perf_counter() - t0,
        )

    def run(self, init_time: datetime, lead_times_hours: List[int]) -> Dict[int, np.ndarray]:
        self._ensure_runtime()
        assert self._model is not None
        assert self._mean_t is not None
        assert self._std_t is not None
        assert self._diff_std_by_interval is not None
        assert self._mean_cpu is not None
        assert self._std_cpu is not None

        wanted = sorted({int(h) for h in lead_times_hours})
        if not wanted:
            return {}
        for lead in wanted:
            if not any(lead % it == 0 for it in self.list_intervals):
                raise RuntimeError(
                    f"No valid interval divides lead {lead} with intervals={self.list_intervals}"
                )

        t0 = time.perf_counter()
        raw = load_stormer_stack(init_time, root=self.era5_root, flip_north_south=False)
        inp_b = base.ensure_batch_bvhw(
            base._prepare_input_tensor(raw, self._mean_cpu, self._std_cpu)
        ).to(device=self.device, dtype=torch.float32)

        out: Dict[int, np.ndarray] = {}
        with torch.no_grad():
            # Fast path for common production setting (single 6h interval):
            # do one autoregressive rollout and cache requested leads, instead
            # of recomputing from init for each lead.
            if len(self.list_intervals) == 1:
                interval = int(self.list_intervals[0])
                max_steps = max(wanted) // interval
                wanted_set = set(wanted)
                x = inp_b
                for step in range(1, max_steps + 1):
                    lead = step * interval
                    logger.info(
                        "Stormer(v2) init=%s lead=%sh (%d/%d), intervals=%s",
                        init_time.strftime("%Y%m%d%H"),
                        lead,
                        step,
                        max_steps,
                        self.list_intervals,
                    )
                    pred_diff_norm = base._stormer_predict_residual(self._model.net, x, interval)
                    pred_diff_norm = base._replace_constant_channels(pred_diff_norm)
                    pred_diff_phys = pred_diff_norm * self._diff_std_by_interval[interval]
                    pred_phys = x * self._std_t + self._mean_t + pred_diff_phys
                    x = (pred_phys - self._mean_t) / self._std_t
                    if lead in wanted_set:
                        out[lead] = (
                            pred_phys.squeeze(0).detach().cpu().numpy().astype(np.float32)
                        )
                logger.info(
                    "Stormer(v2) init=%s leads=%d total_s=%.3f",
                    init_time.strftime("%Y%m%d%H"),
                    len(wanted),
                    time.perf_counter() - t0,
                )
                return out

            for li, lead in enumerate(wanted):
                logger.info(
                    "Stormer(v2) init=%s lead=%sh (%d/%d), intervals=%s",
                    init_time.strftime("%Y%m%d%H"),
                    lead,
                    li + 1,
                    len(wanted),
                    self.list_intervals,
                )
                preds = []
                for interval in self.list_intervals:
                    if lead % interval != 0:
                        continue
                    steps = lead // interval
                    pred_norm = base._forward_validation_explicit(
                        self._model.net,
                        inp_b,
                        interval,
                        steps,
                        self._mean_t,
                        self._std_t,
                        self._diff_std_by_interval[interval],
                    )
                    preds.append(pred_norm)
                mean_pred_norm = torch.stack(preds, dim=0).mean(0)
                denorm = (
                    (mean_pred_norm * self._std_t + self._mean_t)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                out[lead] = denorm
        logger.info(
            "Stormer(v2) init=%s leads=%d total_s=%.3f",
            init_time.strftime("%Y%m%d%H"),
            len(wanted),
            time.perf_counter() - t0,
        )
        return out


def stormer_channel_names_v2() -> List[str]:
    return base.stormer_channel_names()


def interpolate_721_to_stormer_v2(stack721: np.ndarray) -> np.ndarray:
    return base.interpolate_721_to_stormer(stack721)


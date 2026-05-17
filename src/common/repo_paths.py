"""Repository path helpers without cluster-specific hardcoded defaults."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Benchmark repo root (override with ``NWP_BENCHMARK_ROOT``)."""
    explicit = os.environ.get("NWP_BENCHMARK_ROOT", "").strip()
    if explicit:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parents[2]


def static_nc_path() -> Path:
    """GraphCast/Aurora static fields (``NWP_GRAPHCAST_STATIC_NC`` / ``NWP_AURORA_STATIC_NC``)."""
    for key in ("NWP_GRAPHCAST_STATIC_NC", "NWP_AURORA_STATIC_NC", "NWP_STATIC_NC"):
        val = os.environ.get(key, "").strip()
        if val:
            return Path(val)
    return repo_root() / "static.nc"


def nwp_outputs_dir() -> Path:
    """Top-level ``nwp_outputs`` (override with ``NWP_OUTPUTS_DIR``)."""
    explicit = os.environ.get("NWP_OUTPUTS_DIR", "").strip()
    if explicit:
        return Path(explicit)
    return repo_root() / "nwp_outputs"


def data_dir() -> Path:
    """``data/`` under repo (override with ``NWP_DATA_DIR``)."""
    explicit = os.environ.get("NWP_DATA_DIR", "").strip()
    if explicit:
        return Path(explicit)
    return repo_root() / "data"


def tc_eval_results_dir() -> Path:
    """TC evaluation outputs (override with ``NWP_TC_EVAL_DIR``)."""
    explicit = os.environ.get("NWP_TC_EVAL_DIR", "").strip()
    if explicit:
        return Path(explicit)
    return repo_root() / "tc_eval_results"


def debug_log_path(filename: str = "debug.log") -> Path:
    """Optional agent/debug log path (``NWP_DEBUG_LOG``)."""
    explicit = os.environ.get("NWP_DEBUG_LOG", "").strip()
    if explicit:
        return Path(explicit)
    return repo_root() / ".cursor" / filename

"""Resolve IBTrACS CSV path without hardcoding cluster-specific directories.

Priority:
1. Environment variable ``IBTRACS_CSV`` — absolute path to any IBTrACS list CSV.
2. ``ERA5_ROOT`` — use ``<root>/tc_data/ibtracs.last3years.list.v04r01.csv``.
"""

from __future__ import annotations

import os

DEFAULT_IBTRACS_LAST3YEARS = "ibtracs.last3years.list.v04r01.csv"


def ibtracs_csv_path(era5_root: str | None = None) -> str:
    """Return path to the IBTrACS CSV used for BestTracker."""
    explicit = os.environ.get("IBTRACS_CSV")
    if explicit:
        return explicit

    root = era5_root if era5_root is not None else os.environ.get("ERA5_ROOT")
    if root:
        p = os.path.join(root, "tc_data", DEFAULT_IBTRACS_LAST3YEARS)
        if os.path.isfile(p):
            return p
        raise FileNotFoundError(
            f"Expected IBTrACS file not found: {p}. "
            "Download from NCEI or set IBTRACS_CSV to the CSV path."
        )

    raise FileNotFoundError(
        "IBTrACS CSV not found. Set IBTRACS_CSV, or ERA5_ROOT with "
        f"<ERA5_ROOT>/tc_data/{DEFAULT_IBTRACS_LAST3YEARS}."
    )

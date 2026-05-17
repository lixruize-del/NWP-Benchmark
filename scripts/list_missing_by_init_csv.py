#!/usr/bin/env python3
"""List missing per-init split CSV files under a ``by_init`` directory (metrics backfill helper)."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import re


def _parse_model_from_name(name: str) -> str | None:
    # Expected split file name: <model>_<YYYYMMDDHH>.csv
    m = re.match(r"^([A-Za-z0-9]+)_(\d{10})\.csv$", name)
    if not m:
        return None
    return m.group(1).lower()


def _parse_model_hours(items: list[str]) -> dict[str, list[int]]:
    """
    Parse args like:
      pangu:0,6,12,18
      aurora:0,12
    """
    out: dict[str, list[int]] = {}
    for raw in items:
        if ":" not in raw:
            raise ValueError(f"Invalid --model-hours item: {raw!r}")
        model, hours_str = raw.split(":", 1)
        model = model.strip().lower()
        hours = [int(x.strip()) for x in hours_str.split(",") if x.strip()]
        if not hours:
            raise ValueError(f"No hours parsed for model {model!r}")
        out[model] = sorted(set(hours))
    return out


def _iter_expected_inits(start: datetime, end: datetime, hours: list[int]) -> list[str]:
    out: list[str] = []
    cur = start.date()
    end_date = end.date()
    hours = sorted(set(hours))
    while cur <= end_date:
        for h in hours:
            t = datetime(cur.year, cur.month, cur.day, h, 0, 0)
            if start <= t <= end:
                out.append(t.strftime("%Y%m%d%H"))
        cur += timedelta(days=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="List missing per-init CSV files under by_init.")
    ap.add_argument(
        "--by-init-dir",
        type=Path,
        required=True,
        help="Directory containing <model>_<YYYYMMDDHH>.csv files",
    )
    ap.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument(
        "--init-hours",
        type=int,
        nargs="+",
        default=[0, 6, 12, 18],
        help="Default init hours for all models",
    )
    ap.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Optional model list. Default: infer from by_init filenames.",
    )
    ap.add_argument(
        "--model-hours",
        type=str,
        nargs="*",
        default=[],
        help="Optional per-model hours override, e.g. pangu:0,6,12,18 aurora:0,12",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <by-init-dir>/missing_by_init.csv)",
    )
    args = ap.parse_args()

    by_init_dir = args.by_init_dir
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23, minute=59)
    default_hours = sorted(set(args.init_hours))
    model_hours_override = _parse_model_hours(args.model_hours)

    if args.models:
        models = sorted(set(m.strip().lower() for m in args.models if m.strip()))
    else:
        inferred: set[str] = set()
        for p in by_init_dir.glob("*.csv"):
            m = _parse_model_from_name(p.name)
            if m is not None:
                inferred.add(m)
        models = sorted(inferred)

    missing_rows: list[dict[str, str]] = []
    missing_count_by_model: dict[str, int] = defaultdict(int)
    expected_count_by_model: dict[str, int] = {}

    for model in models:
        hours = model_hours_override.get(model, default_hours)
        expected_inits = _iter_expected_inits(start, end, hours)
        expected_count_by_model[model] = len(expected_inits)
        for init in expected_inits:
            expected_name = f"{model}_{init}.csv"
            expected_path = by_init_dir / expected_name
            if not expected_path.exists():
                missing_rows.append(
                    {
                        "model": model,
                        "init_time": init,
                        "expected_file": expected_name,
                        "expected_path": str(expected_path),
                    }
                )
                missing_count_by_model[model] += 1

    out_csv = args.out_csv or (by_init_dir / "missing_by_init.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "init_time", "expected_file", "expected_path"])
        w.writeheader()
        w.writerows(missing_rows)

    print("MISSING_SUMMARY_START")
    print(f"by_init_dir={by_init_dir}")
    print(f"range={args.start}..{args.end}")
    print(f"default_init_hours={','.join(str(x) for x in default_hours)}")
    print(f"output_csv={out_csv}")
    print(f"models={len(models)}")
    print(f"missing_total={len(missing_rows)}")
    for model in models:
        exp = expected_count_by_model.get(model, 0)
        miss = missing_count_by_model.get(model, 0)
        print(f"model={model} expected={exp} missing={miss} present={exp-miss}")
    print("MISSING_SUMMARY_END")


if __name__ == "__main__":
    main()


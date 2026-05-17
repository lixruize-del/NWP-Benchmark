#!/usr/bin/env python3
"""Run storm-centric TC evaluation in parallel by model.

This wrapper launches one subprocess per model (up to max_parallel_models)
to execute scripts/evaluate_tc_by_storm.py with identical settings.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "evaluate_tc_by_storm.py"
DEFAULT_MODELS = ["aifs", "aurora", "fengwu", "fuxi", "graphcast", "pangu", "stormer"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parallel TC storm-centric evaluation by model.")
    p.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to launch evaluate_tc_by_storm.py",
    )
    p.add_argument(
        "--forecast-root",
        type=Path,
        default=Path("nwp_outputs/ifs_monthly_202506_v2/forecasts"),
    )
    p.add_argument("--source", type=str, default="ifs")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--start-date", type=str, default="2025-06-01")
    p.add_argument("--end-date", type=str, default="2025-12-31")
    p.add_argument("--lead-step", type=int, default=6)
    p.add_argument("--max-lead", type=int, default=240)
    p.add_argument("--distance-threshold", type=float, default=560.0)
    p.add_argument("--wind-threshold", type=float, default=8.0)
    p.add_argument("--resolution", type=float, default=0.25)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "tc_eval_results" / "storm_centric",
    )
    p.add_argument("--basins", nargs="*", default=None)
    p.add_argument("--subbasins", nargs="*", default=None)
    p.add_argument("--sid", nargs="*", default=None)
    p.add_argument("--storm-name", nargs="*", default=None)
    p.add_argument("--max-parallel-models", type=int, default=3)
    return p


def _build_cmd(args: argparse.Namespace, model: str) -> list[str]:
    cmd: list[str] = [
        str(args.python_bin),
        str(EVAL_SCRIPT),
        "--forecast-root",
        str(args.forecast_root),
        "--source",
        str(args.source),
        "--models",
        model,
        "--season",
        str(args.season),
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--lead-step",
        str(args.lead_step),
        "--max-lead",
        str(args.max_lead),
        "--distance-threshold",
        str(args.distance_threshold),
        "--wind-threshold",
        str(args.wind_threshold),
        "--resolution",
        str(args.resolution),
        "--out-dir",
        str(args.out_dir),
    ]

    if args.basins:
        cmd.append("--basins")
        cmd.extend(args.basins)
    if args.subbasins:
        cmd.append("--subbasins")
        cmd.extend(args.subbasins)
    if args.sid:
        cmd.append("--sid")
        cmd.extend(args.sid)
    if args.storm_name:
        cmd.append("--storm-name")
        cmd.extend(args.storm_name)
    return cmd


def _run_one(args: argparse.Namespace, model: str, log_dir: Path) -> tuple[str, int, Path]:
    log_path = log_dir / f"{model}.log"
    cmd = _build_cmd(args, model)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    return model, int(proc.returncode), log_path


def main() -> None:
    args = _build_parser().parse_args()
    if not EVAL_SCRIPT.exists():
        raise SystemExit(f"Missing script: {EVAL_SCRIPT}")
    if args.max_parallel_models < 1:
        raise SystemExit("--max-parallel-models must be >= 1")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.out_dir / args.source.lower() / f"parallel_logs_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[parallel-start] models={len(args.models)} max_parallel={args.max_parallel_models} "
        f"logs={log_dir}"
    )
    results: list[tuple[str, int, Path]] = []
    with ThreadPoolExecutor(max_workers=int(args.max_parallel_models)) as pool:
        futures = {pool.submit(_run_one, args, m, log_dir): m for m in args.models}
        for fut in as_completed(futures):
            model, code, log_path = fut.result()
            results.append((model, code, log_path))
            status = "ok" if code == 0 else "fail"
            print(f"[{status}] model={model} exit_code={code} log={log_path}")

    failed = [r for r in results if r[1] != 0]
    if failed:
        print(f"[parallel-done] failed={len(failed)}/{len(results)}")
        for model, code, log_path in failed:
            print(f"  - {model}: exit_code={code} log={log_path}")
        raise SystemExit(1)
    print(f"[parallel-done] success={len(results)}/{len(results)}")


if __name__ == "__main__":
    main()

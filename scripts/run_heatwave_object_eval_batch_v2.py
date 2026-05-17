#!/usr/bin/env python3
"""Batch runner for object-based heatwave verification pipeline (v2)."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.common.repo_paths import nwp_outputs_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run heatwave object-eval v2 in batch.")
    p.add_argument(
        "--forecast-root",
        type=Path,
        default=nwp_outputs_dir() / "era5_monthly_202506_v2/forecasts",
    )
    p.add_argument(
        "--init-source",
        type=str,
        default="era5",
        choices=("era5", "ifs"),
        help="Initial-condition source tag used in output folder layout.",
    )
    p.add_argument(
        "--baseline-file",
        type=Path,
        required=True,
        help="p90 baseline nc produced by build_heatwave_baseline_percentile.py",
    )
    p.add_argument("--baseline-var", type=str, default="")
    p.add_argument("--event-type", type=str, choices=("heatwave", "coldwave"), default="heatwave")
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["aifs", "aurora", "fuxi", "fengwu", "pangu", "graphcast", "stormer"],
    )
    p.add_argument("--lead-days", type=int, nargs="+", default=[1, 3, 7, 10])
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--start-date", type=str, default="", help="Inclusive target date YYYY-MM-DD")
    p.add_argument("--end-date", type=str, default="", help="Inclusive target date YYYY-MM-DD")
    p.add_argument("--min-duration-days", type=int, default=3)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/ecmwf-era5-datasets/era5_np.25"),
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=nwp_outputs_dir() / "era5_monthly_202506_v2" / "metrics/heatwave_object_v2",
    )
    p.add_argument(
        "--gt-cache-dir",
        type=Path,
        default=Path(""),
        help="Optional shared directory for yearly GT cache. Default: <out-root>/<init-source>/_shared",
    )
    p.add_argument("--skip-missing", action="store_true")
    p.add_argument("--write-event-ids", action="store_true")
    p.add_argument(
        "--skip-phase-a",
        action="store_true",
        help="Skip fused Step1 build and reuse existing step1 outputs + GT cache.",
    )
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--phase-a-workers", type=int, default=2, help="Concurrent workers for Phase A fused Step1.")
    p.add_argument("--phase-b-workers", type=int, default=3, help="Concurrent workers for Phase B Step2/3.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _run(cmd: list[str], dry_run: bool, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n")
    if dry_run:
        return
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"START: {datetime.utcnow().isoformat(timespec='seconds')}Z\n")
        lf.flush()
        subprocess.check_call(cmd, stdout=lf, stderr=lf)
        lf.write(f"END: {datetime.utcnow().isoformat(timespec='seconds')}Z\n")
        lf.flush()


def _ensure_gt_cache(
    args: argparse.Namespace,
    step1: Path,
    gt_cache_file: Path,
    model: str,
    lead_day: int,
) -> None:
    """Build shared GT cache once before concurrent workers start."""
    if args.dry_run or gt_cache_file.exists():
        return
    warmup_dir = gt_cache_file.parent / "_cache_warmup"
    warmup_log = gt_cache_file.parent / "gt_cache_build.log"
    cmd = [
        args.python_bin,
        str(step1),
        "--gt-only",
        "--year",
        str(int(args.year)),
        "--start-date",
        args.start_date.strip() if args.start_date.strip() else f"{int(args.year)}-01-01",
        "--end-date",
        args.end_date.strip() if args.end_date.strip() else f"{int(args.year)}-12-31",
        "--gt-root",
        str(args.gt_root),
        "--gt-cache-file",
        str(gt_cache_file),
        "--out-dir",
        str(warmup_dir),
    ]
    if args.skip_missing:
        cmd.append("--skip-missing")
    print(
        f"GT cache not found. Building once with model={model}, lead_day={lead_day}: {gt_cache_file}",
        flush=True,
    )
    _run(cmd, args.dry_run, warmup_log)
    if not gt_cache_file.exists():
        raise RuntimeError(f"GT cache build failed: {gt_cache_file} not created")


def _build_task_context(
    args: argparse.Namespace,
    model: str,
    lead_day: int,
    step2: Path,
    step3: Path,
    gt_cache_file: Path,
) -> dict[str, Any]:
    run_dir = args.out_root / args.init_source / model / f"lead_day_{int(lead_day)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    step1_dir = run_dir / "step1"
    step2_dir = run_dir / "step2"
    step3_dir = run_dir / "step3"
    task_log = run_dir / "eval_task.log"
    is_heat = args.event_type == "heatwave"
    pred_file = step1_dir / ("pred_tmax_daily.nc" if is_heat else "pred_tmin_daily.nc")
    gt_file = gt_cache_file if gt_cache_file.exists() else (step1_dir / ("gt_tmax_daily.nc" if is_heat else "gt_tmin_daily.nc"))
    masks_file = step2_dir / ("hot_masks_p90.nc" if is_heat else "cold_masks_p10.nc")

    cmd2 = [
        args.python_bin,
        str(step2),
        "--event-type",
        args.event_type,
        "--pred-file",
        str(pred_file),
        "--gt-file",
        str(gt_file),
        "--baseline-file",
        str(args.baseline_file),
        "--out-dir",
        str(step2_dir),
        "--min-duration-days",
        str(int(args.min_duration_days)),
    ]
    if args.baseline_var:
        cmd2.extend(["--baseline-var", args.baseline_var])
    if is_heat:
        cmd2.extend(["--pred-var", "tmax_pred_c", "--gt-var", "tmax_gt_c"])
    else:
        cmd2.extend(["--pred-var", "tmin_pred_c", "--gt-var", "tmin_gt_c"])
    if args.write_event_ids:
        cmd2.append("--write-event-ids")

    cmd3 = [
        args.python_bin,
        str(step3),
        "--event-type",
        args.event_type,
        "--masks-file",
        str(masks_file),
        "--iou-threshold",
        str(float(args.iou_threshold)),
        "--min-duration-days",
        str(int(args.min_duration_days)),
        "--out-dir",
        str(step3_dir),
    ]
    return {
        "model": model,
        "lead_day": int(lead_day),
        "run_dir": run_dir,
        "task_log": task_log,
        "step3_dir": step3_dir,
        "commands": [cmd2, cmd3],
    }


def _run_one_task(task: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    model = str(task["model"])
    lead_day = int(task["lead_day"])
    run_dir = Path(task["run_dir"])
    task_log = Path(task["task_log"])
    step3_dir = Path(task["step3_dir"])
    commands = list(task["commands"])

    with open(task_log, "a", encoding="utf-8") as lf:
        lf.write(
            f"TASK_START model={model} lead_day={lead_day} at={datetime.utcnow().isoformat(timespec='seconds')}Z\n"
        )

    for cmd in commands:
        _run(cmd, dry_run, task_log)

    row: dict[str, Any] = {
        "init_source": "",
        "model": model,
        "lead_day": lead_day,
        "run_dir": str(run_dir),
        "status": "ok",
        "task_log": str(task_log),
    }
    if not dry_run:
        gfile = step3_dir / "metrics_global.json"
        if gfile.exists():
            with open(gfile, "r", encoding="utf-8") as f:
                gobj = json.load(f)
            mm = gobj.get("metric_global_weighted_mean", {})
            row.update(
                {
                    "precision": mm.get("precision"),
                    "recall": mm.get("recall"),
                    "f1": mm.get("f1"),
                    "pod": mm.get("pod"),
                    "far": mm.get("far"),
                    "csi": mm.get("csi"),
                }
            )
    with open(task_log, "a", encoding="utf-8") as lf:
        lf.write(f"TASK_END model={model} lead_day={lead_day} at={datetime.utcnow().isoformat(timespec='seconds')}Z\n")
    return row


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.phase_a_workers <= 0 or args.phase_b_workers <= 0:
        raise ValueError("phase-a-workers and phase-b-workers must be positive")
    if args.phase_a_workers > 4:
        raise ValueError("--phase-a-workers should be <= 4 to avoid NAS thrashing")
    if args.phase_b_workers > 6:
        raise ValueError("--phase-b-workers should be <= 6")

    scripts_dir = Path(__file__).resolve().parent
    step1 = scripts_dir / "build_heatwave_lead_timeseries_v2.py"
    step2 = scripts_dir / "extract_heatwave_events_v2.py"
    step3 = scripts_dir / "eval_heatwave_object_metrics_v2.py"

    rows: list[dict[str, Any]] = []
    use_custom_gt_cache_dir = str(args.gt_cache_dir).strip() not in ("", ".")
    shared_dir = args.gt_cache_dir if use_custom_gt_cache_dir else (args.out_root / args.init_source / "_shared")
    shared_dir.mkdir(parents=True, exist_ok=True)
    cache_start = args.start_date.strip().replace("-", "") if args.start_date.strip() else f"{int(args.year)}0101"
    cache_end = args.end_date.strip().replace("-", "") if args.end_date.strip() else f"{int(args.year)}1231"
    gt_cache_file = shared_dir / f"gt_tmax_daily_{int(args.year)}_{cache_start}_{cache_end}.nc"

    # Ensure shared GT cache exists before launching workers.
    warm_model = str(args.models[0])
    warm_lead = int(args.lead_days[0])
    _ensure_gt_cache(args, step1, gt_cache_file, warm_model, warm_lead)

    # Phase A: Step1 once per model (fused lead-days and max/min outputs).
    if not args.skip_phase_a:
        phase_a_jobs: list[tuple[str, list[str], Path]] = []
        for model in args.models:
            model_log = args.out_root / args.init_source / str(model) / "step1_fused.log"
            cmd = [
                args.python_bin,
                str(step1),
                "--forecast-root",
                str(args.forecast_root),
                "--model",
                str(model),
                "--lead-days",
                *[str(int(x)) for x in args.lead_days],
                "--year",
                str(int(args.year)),
                "--start-date",
                args.start_date.strip() if args.start_date.strip() else f"{int(args.year)}-01-01",
                "--end-date",
                args.end_date.strip() if args.end_date.strip() else f"{int(args.year)}-12-31",
                "--gt-root",
                str(args.gt_root),
                "--gt-cache-file",
                str(gt_cache_file),
                "--no-write-local-gt",
                "--out-dir",
                str(args.out_root / args.init_source),
            ]
            if args.skip_missing:
                cmd.append("--skip-missing")
            phase_a_jobs.append((str(model), cmd, model_log))

        print(
            f"PhaseA start: models={len(phase_a_jobs)}, workers={args.phase_a_workers}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=int(args.phase_a_workers)) as pool:
            fut_map = {
                pool.submit(_run, cmd, args.dry_run, log): model for model, cmd, log in phase_a_jobs
            }
            a_done = 0
            for fut in as_completed(fut_map):
                model = fut_map[fut]
                fut.result()
                a_done += 1
                print(f"PhaseA model[{model}] done. {a_done}/{len(phase_a_jobs)}", flush=True)

    # Phase B: Step2/3 per lead in parallel (lighter on NAS).
    tasks: list[dict[str, Any]] = []
    for model in args.models:
        for lead_day in args.lead_days:
            tasks.append(
                _build_task_context(
                    args=args,
                    model=str(model),
                    lead_day=int(lead_day),
                    step2=step2,
                    step3=step3,
                    gt_cache_file=gt_cache_file,
                )
            )

    total_tasks = len(tasks)
    done = 0
    print(
        f"Starting PhaseB tasks={total_tasks}, workers={args.phase_b_workers}, "
        f"gt_cache={gt_cache_file}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=int(args.phase_b_workers)) as pool:
        future_map = {pool.submit(_run_one_task, t, args.dry_run): t for t in tasks}
        for fut in as_completed(future_map):
            task = future_map[fut]
            model = task["model"]
            lead_day = task["lead_day"]
            try:
                row = fut.result()
                row["init_source"] = args.init_source
                rows.append(row)
                done += 1
                print(f"Task[{model} - lead {lead_day}] finished. {done}/{total_tasks} completed.", flush=True)
            except Exception as e:
                rows.append(
                    {
                        "init_source": args.init_source,
                        "model": model,
                        "lead_day": int(lead_day),
                        "run_dir": str(task["run_dir"]),
                        "status": "failed",
                        "task_log": str(task["task_log"]),
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                done += 1
                print(
                    f"Task[{model} - lead {lead_day}] failed. {done}/{total_tasks} completed. "
                    f"log={task['task_log']}",
                    flush=True,
                )

    summary_csv = args.out_root / args.init_source / "summary_all.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    meta = {
        "forecast_root": str(args.forecast_root),
        "init_source": args.init_source,
        "start_date": args.start_date.strip() if args.start_date.strip() else f"{int(args.year)}-01-01",
        "end_date": args.end_date.strip() if args.end_date.strip() else f"{int(args.year)}-12-31",
        "baseline_file": str(args.baseline_file),
        "event_type": args.event_type,
        "models": [str(m) for m in args.models],
        "lead_days": [int(x) for x in args.lead_days],
        "year": int(args.year),
        "gt_cache_file": str(gt_cache_file),
        "skip_phase_a": bool(args.skip_phase_a),
        "phase_a_workers": int(args.phase_a_workers),
        "phase_b_workers": int(args.phase_b_workers),
        "total_tasks": int(total_tasks),
        "min_duration_days": int(args.min_duration_days),
        "iou_threshold": float(args.iou_threshold),
        "dry_run": bool(args.dry_run),
        "summary_csv": str(summary_csv),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with open(args.out_root / args.init_source / "batch_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    print("Batch pipeline done.")
    print(f"  - summary: {summary_csv}")


if __name__ == "__main__":
    main()


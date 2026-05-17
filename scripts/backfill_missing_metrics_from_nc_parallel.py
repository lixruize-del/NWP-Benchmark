#!/usr/bin/env python3
"""
Append metrics rows from on-disk forecast NetCDF using the same stack as ``run_large_scale``.

Per-init parallelism:
  * ``ThreadPoolExecutor`` over lead hours (NC read + optional regrid + GT/clim + metrics).
  * Optional ``--init-workers`` > 1 processes multiple inits concurrently (locked caches).

Model-specific regrid (matches monthly inference outputs on ERA5 0.25°):
  * ``stormer`` / ``aurora`` / ``neuralgcm``: ``regrid_model_pred_eval_to_era5_025`` (lat/lon from NC when present).
GT for ``stormer`` / ``aurora`` / ``neuralgcm`` uses ``load_snapshot_by_channel_names`` on 721×1440;
other models use ``load_gt_subset_by_model`` with the usual adapter layout.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Nested parallelism kill: ThreadPoolExecutor(N) × OpenMP/BLAS inside each worker
# → thousands of threads / OOM / silent death with no Python traceback.
# Must set before importing numpy (and before run_large_scale → torch).
for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_large_scale as base
from src.common.saver import select_pressure_hpa

try:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# xarray → netCDF4 → libHDF5: concurrent opens from ThreadPoolExecutor segfault on many
# builds (concurrent HDF5 opens are unsafe). Serialize all NC access.
_HDF5_THREAD_LOCK = threading.Lock()


def _parse_init(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%Y%m%d%H")


def _infer_eval_names(model: str, by_init_dir: Path, fallback: List[str]) -> List[str]:
    files = sorted(by_init_dir.glob(f"{model}_*.csv"))
    for p in files:
        try:
            df = pd.read_csv(p, usecols=["variable"])
        except Exception:
            continue
        if df.empty:
            continue
        names = [str(x).strip() for x in df["variable"].tolist() if str(x).strip()]
        out = list(dict.fromkeys(names))
        if model == "fuxi":
            out = [
                x
                for x in out
                if not base._normalize_fuxi_eval_name(str(x)).startswith("r_")
            ]
        if out:
            return out
    fb = list(fallback)
    if model == "fuxi":
        fb = [
            x
            for x in fb
            if not base._normalize_fuxi_eval_name(str(x)).startswith("r_")
        ]
    return fb


def _lead_from_name(name: str) -> int | None:
    m = re.search(r"-(\d{2,3})(?:_\d+)?\.nc$", name)
    if not m:
        return None
    return int(m.group(1))


def _lead_files_in_dir(init_dir: Path) -> List[Tuple[int, Path]]:
    lead_files: List[Tuple[int, Path]] = []
    for p in init_dir.glob("*.nc"):
        lf = _lead_from_name(p.name)
        if lf is None:
            continue
        lead_files.append((lf, p))
    lead_files.sort(key=lambda x: x[0])
    return lead_files


def _pick_forecast_init_dir(
    forecasts_root: Path,
    model: str,
    init_str: str,
) -> tuple[Optional[Path], List[Tuple[int, Path]]]:
    """Return ``(init_dir, lead_files)`` under ``forecasts_root/model/init``."""
    primary = forecasts_root / model / init_str
    lf_primary = _lead_files_in_dir(primary) if primary.is_dir() else []
    if lf_primary:
        return primary, lf_primary
    return None, []


def read_pred_lat_lon_one_nc_open(
    nc_path: Path, eval_names: List[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One locked ``open_dataset`` per lead: planes + lat/lon (half the HDF5 critical sections)."""
    with _HDF5_THREAD_LOCK:
        ds = xr.open_dataset(nc_path)
        try:
            lat = None
            lon = None
            for lk in ("latitude", "lat"):
                if lk in ds.coords or lk in ds.variables:
                    lat = np.asarray(ds[lk].values, dtype=np.float64)
                    break
            for lk in ("longitude", "lon"):
                if lk in ds.coords or lk in ds.variables:
                    lon = np.asarray(ds[lk].values, dtype=np.float64)
                    break
            if lat is None or lon is None:
                raise ValueError(f"Missing lat/lon in {nc_path}")

            out: list[np.ndarray] = []
            for nm in eval_names:
                n = nm.strip().lower()
                base_name = n
                lev = None
                m = re.match(r"^([a-z0-9]+)_(\d{2,4})$", n)
                if m:
                    base_name = m.group(1)
                    lev = float(m.group(2))
                if base_name not in ds:
                    raise KeyError(f"{base_name} not found in {nc_path}")
                da = ds[base_name]
                if "time" in da.dims:
                    da = da.isel(time=0)
                if lev is not None:
                    da = select_pressure_hpa(da, lev)
                arr = np.asarray(da.values, dtype=np.float32)
                out.append(arr)
            pred = np.stack(out, axis=0).astype(np.float32)
            return pred, lat, lon
        finally:
            ds.close()


def _available_eval_names_from_nc(nc_path: Path) -> List[str]:
    with _HDF5_THREAD_LOCK:
        ds = xr.open_dataset(nc_path)
        try:
            out: list[str] = []
            for var in ds.data_vars:
                da = ds[var]
                plev_dims = [d for d in da.dims if d == "isobaricInhPa" or d.startswith("plev_")]
                if not plev_dims:
                    out.append(str(var).lower())
                    continue
                plev_dim = plev_dims[0]
                levels = np.asarray(da.coords[plev_dim].values, dtype=np.float64)
                for lv in levels:
                    out.append(f"{str(var).lower()}_{int(round(float(lv)))}")
            return list(dict.fromkeys(out))
        finally:
            ds.close()


def _climatology_for_model(
    model: str,
    valid_time: datetime,
    eval_names: List[str],
    pred_eval: np.ndarray,
    lat: np.ndarray,
    *,
    clim_era5_root: Path,
    flip_north_south: bool,
) -> np.ndarray | None:
    try:
        clim721 = base._load_climatology_721(
            valid_time,
            eval_names,
            era5_root=clim_era5_root,
            flip_north_south=flip_north_south,
        )
    except Exception:
        return None

    if pred_eval.shape[-2:] == (721, 1440):
        return clim721
    return None


def _build_adapter(model: str, era5_root: Path, ifs_root: Path, fengwu_onnx: str):
    return base.build_adapter(model, era5_root, ifs_root, fengwu_onnx)


def _cache_get_or_put(
    cache: Dict[tuple, np.ndarray],
    key: tuple,
    producer: Callable[[], np.ndarray],
    *,
    max_items: int = 256,
) -> np.ndarray:
    if key in cache:
        return cache[key]
    val = producer()
    if len(cache) >= max_items:
        cache.clear()
    cache[key] = val
    return val


def _locked_cache_get_or_put(
    lock: threading.Lock,
    cache: Dict[tuple, Any],
    key: tuple,
    producer: Callable[[], Any],
    *,
    max_items: int = 256,
) -> Any:
    with lock:
        return _cache_get_or_put(cache, key, producer, max_items=max_items)


LeadResult = Tuple[int, Optional[List[Dict[str, Any]]], Optional[str]]


def _process_one_lead(
    *,
    lead: int,
    nc_path: Path,
    model: str,
    init_str: str,
    init_time: datetime,
    eval_names_local: List[str],
    era5_root: Path,
    clim_root: Path,
    adapter: object,
    cache_lock: threading.Lock,
    gt_cache: Dict[tuple, np.ndarray],
    clim_cache: Dict[tuple, np.ndarray | None],
) -> LeadResult:
    """Compute all metric rows for one (init, lead). Thread-safe cache access."""
    # Parallel NC reads + metrics per thread; here we also run
    # scipy interp + PyTorch metrics per thread — catch everything so one bad lead
    # cannot kill the whole process (HDF5 / torch can raise across threads).
    try:
        try:
            pred_eval, lat_nc, lon_nc = read_pred_lat_lon_one_nc_open(
                nc_path, eval_names_local
            )
        except Exception as e:
            return lead, None, f"pred_read_error:{type(e).__name__}:{e}"

        if model in ("stormer", "aurora", "neuralgcm") and pred_eval.shape[-2:] != (721, 1440):
            from src.common.era5_eval_regrid import regrid_model_pred_eval_to_era5_025

            pred_eval = regrid_model_pred_eval_to_era5_025(
                model, pred_eval, src_lat=lat_nc, src_lon=lon_nc
            )

        valid_time = init_time + timedelta(hours=int(lead))

        def _get_gt() -> np.ndarray:
            if model in ("stormer", "aurora", "neuralgcm"):
                return base.load_snapshot_by_channel_names(
                    valid_time,
                    eval_names_local,
                    root=era5_root,
                    flip_north_south=False,
                )
            return base.load_gt_subset_by_model(
                model,
                valid_time,
                eval_names_local,
                era5_root=era5_root,
                adapter=adapter,
            )

        gt_key = (model, valid_time.strftime("%Y%m%d%H"), tuple(eval_names_local))
        try:
            gt_stack = _locked_cache_get_or_put(
                cache_lock,
                gt_cache,
                gt_key,
                _get_gt,
            )
        except FileNotFoundError:
            return lead, None, None
        except Exception as e:
            return lead, None, f"gt_error:{type(e).__name__}:{e}"

        if gt_stack.shape != pred_eval.shape:
            return lead, None, "shape_mismatch"

        clim_key = (
            model,
            valid_time.strftime("%Y%m%d%H"),
            tuple(eval_names_local),
            pred_eval.shape,
        )

        def _get_clim() -> np.ndarray | None:
            return _climatology_for_model(
                model=model,
                valid_time=valid_time,
                eval_names=eval_names_local,
                pred_eval=pred_eval,
                lat=adapter.lat,
                clim_era5_root=clim_root,
                flip_north_south=False,
            )

        clim = _locked_cache_get_or_put(cache_lock, clim_cache, clim_key, _get_clim)
        if clim is not None and clim.shape != pred_eval.shape:
            clim = None

        m = base._compute_weighted_metrics(pred_eval, gt_stack, clim, use_eval_wrmse=True)
        rows: List[Dict[str, Any]] = []
        for vi, var in enumerate(eval_names_local):
            rows.append(
                {
                    "init_time": init_str,
                    "valid_time": valid_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "lead_hours": int(lead),
                    "variable": var,
                    "wrmse": float(m["wrmse"][vi]),
                    "bias": float(m["bias"][vi]),
                    "mae": float(m["mae"][vi]),
                    "activity": float(m["activity"][vi]),
                    "acc": float(m["acc"][vi]),
                }
            )
        return lead, rows, None
    except Exception as e:
        return lead, None, f"lead_fatal:{type(e).__name__}:{e}"


def _run_single_init(
    *,
    r: pd.Series,
    args: argparse.Namespace,
    clim_root: Path,
    adapters: Dict[str, object],
    eval_names_cache: Dict[str, List[str]],
    adapter_lock: threading.Lock,
    gt_cache: Dict[tuple, np.ndarray],
    clim_cache: Dict[tuple, np.ndarray | None],
    cache_lock: threading.Lock,
    unresolved_lock: threading.Lock,
    unresolved: List[Tuple[str, str, str]],
    lead_workers: int,
) -> Tuple[str, str, int, str] | None:
    model = str(r["model"]).strip().lower()
    init_str = str(r["init_time"]).strip()
    out_csv = args.by_init_dir / f"{model}_{init_str}.csv"
    if out_csv.exists() and not args.overwrite:
        return None

    try:
        with adapter_lock:
            if model not in adapters:
                adapters[model] = _build_adapter(
                    model=model,
                    era5_root=args.era5_root,
                    ifs_root=args.ifs_hres_root,
                    fengwu_onnx=args.fengwu_onnx,
                )
                adapter = adapters[model]
                if args.variables:
                    eval_names_cache[model] = args.variables
                else:
                    eval_names_cache[model] = _infer_eval_names(
                        model, args.by_init_dir, adapter.channel_names
                    )
            adapter = adapters[model]
            eval_names = eval_names_cache[model]

        init_dir, lead_files = _pick_forecast_init_dir(
            args.forecasts_root, model, init_str
        )
        if not lead_files or init_dir is None:
            with unresolved_lock:
                unresolved.append(
                    (
                        model,
                        init_str,
                        "no_forecast_nc",
                    )
                )
            return None

        init_time = _parse_init(init_str)

        available_names = set(_available_eval_names_from_nc(lead_files[0][1]))
        eval_names_local = [x for x in eval_names if x in available_names]
        if not eval_names_local:
            eval_names_local = sorted(available_names)
        # FuXi NC stores moisture as r_* (RH); ERA5 metrics here follow q/t/z/u/v — skip RH for GT load.
        if model == "fuxi":
            eval_names_local = [
                x
                for x in eval_names_local
                if not base._normalize_fuxi_eval_name(str(x)).startswith("r_")
            ]
        if not eval_names_local:
            with unresolved_lock:
                unresolved.append((model, init_str, "no_eval_names_after_nc_intersection"))
            return None

        futures = {}
        n_workers = max(1, min(lead_workers, len(lead_files)))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for lead, nc_path in lead_files:
                fut = pool.submit(
                    _process_one_lead,
                    lead=lead,
                    nc_path=nc_path,
                    model=model,
                    init_str=init_str,
                    init_time=init_time,
                    eval_names_local=eval_names_local,
                    era5_root=args.era5_root,
                    clim_root=clim_root,
                    adapter=adapter,
                    cache_lock=cache_lock,
                    gt_cache=gt_cache,
                    clim_cache=clim_cache,
                )
                futures[fut] = lead

            batches: List[LeadResult] = []
            for fut in as_completed(futures):
                lead_tag = futures[fut]
                try:
                    batches.append(fut.result())
                except Exception as ex:
                    batches.append(
                        (lead_tag, None, f"future_error:{type(ex).__name__}:{ex}")
                    )

        init_rows: List[Dict[str, Any]] = []
        batches.sort(key=lambda x: x[0])
        for lead, rows, err in batches:
            if err:
                with unresolved_lock:
                    unresolved.append((model, init_str, err))
                continue
            if rows:
                init_rows.extend(rows)

        cols = [
            "init_time",
            "valid_time",
            "lead_hours",
            "variable",
            "wrmse",
            "bias",
            "mae",
            "activity",
            "acc",
        ]
        pd.DataFrame(init_rows, columns=cols).to_csv(out_csv, index=False)
        return model, init_str, len(init_rows), str(out_csv)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        with unresolved_lock:
            unresolved.append((model, init_str, f"init_fatal:{type(e).__name__}:{e}"))
        return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Parallel backfill of by-init metrics CSVs from forecast NC "
            "(accelerated; original script unchanged)."
        )
    )
    ap.add_argument("--missing-csv", type=Path, required=True)
    ap.add_argument("--forecasts-root", type=Path, required=True)
    ap.add_argument("--by-init-dir", type=Path, required=True)
    ap.add_argument("--era5-root", type=Path, default=base.DEFAULT_ERA5_NPY_ROOT)
    ap.add_argument("--ifs-hres-root", type=Path, default=base.DEFAULT_IFS_HRES_ROOT)
    ap.add_argument("--fengwu-onnx", type=str, default="fengwu_v1.onnx")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--workers",
        type=int,
        default=6,
        help=(
            "Threads per init for parallel lead-hours. Keep modest (e.g. 4–8): each "
            "thread runs scipy + PyTorch; high values × BLAS threads risk OOM / worker kill."
        ),
    )
    ap.add_argument(
        "--init-workers",
        type=int,
        default=1,
        help="How many inits to process concurrently (each init still uses --workers). Default: 1.",
    )
    ap.add_argument("--variables", type=str, nargs="+", default=None)
    ap.add_argument(
        "--clim-era5-root",
        type=Path,
        default=None,
        help="Climatology root for ACC/activity (default: canonical ERA5).",
    )
    args = ap.parse_args()

    clim_root = args.clim_era5_root if args.clim_era5_root is not None else base.DEFAULT_ERA5_NPY_ROOT

    args.by_init_dir.mkdir(parents=True, exist_ok=True)
    missing_df = pd.read_csv(args.missing_csv)

    adapters: Dict[str, object] = {}
    eval_names_cache: Dict[str, List[str]] = {}
    gt_cache: Dict[tuple, np.ndarray] = {}
    clim_cache: Dict[tuple, np.ndarray | None] = {}
    cache_lock = threading.Lock()
    adapter_lock = threading.Lock()
    unresolved_lock = threading.Lock()
    unresolved: List[Tuple[str, str, str]] = []
    done = defaultdict(int)
    rows: List[Tuple[str, str, int, str]] = []
    init_workers = max(1, args.init_workers)

    def _one(row: pd.Series) -> Optional[Tuple[str, str, int, str]]:
        return _run_single_init(
            r=row,
            args=args,
            clim_root=clim_root,
            adapters=adapters,
            eval_names_cache=eval_names_cache,
            adapter_lock=adapter_lock,
            gt_cache=gt_cache,
            clim_cache=clim_cache,
            cache_lock=cache_lock,
            unresolved_lock=unresolved_lock,
            unresolved=unresolved,
            lead_workers=args.workers,
        )

    if init_workers == 1:
        for _, r in tqdm(missing_df.iterrows(), total=len(missing_df), desc="Backfilling inits"):
            try:
                out = _one(r)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                continue
            if out:
                m, init_str, n, path = out
                done[m] += 1
                rows.append(out)
    else:
        with ThreadPoolExecutor(max_workers=init_workers) as pool:
            futs = [pool.submit(_one, r) for _, r in missing_df.iterrows()]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Backfilling inits"):
                try:
                    out = fut.result()
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                    continue
                if out:
                    m, init_str, n, path = out
                    done[m] += 1
                    rows.append(out)

    unresolved_csv = args.by_init_dir / "missing_backfill_unresolved.csv"
    with unresolved_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "init_time", "reason"])
        w.writerows(unresolved)

    print("BACKFILL_SUMMARY_START")
    print(f"filled_files={len(rows)}")
    for model in sorted(done):
        print(f"model={model} filled={done[model]}")
    print(f"unresolved_rows={len(unresolved)}")
    print(f"unresolved_csv={unresolved_csv}")
    print("BACKFILL_SUMMARY_END")


if __name__ == "__main__":
    import faulthandler

    faulthandler.enable(all_threads=True)
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

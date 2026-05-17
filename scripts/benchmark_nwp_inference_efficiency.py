#!/usr/bin/env python3
"""
Measure parameter counts (static weight elements), inference latency (warmup + repeats),
and peak GPU memory for NWP forecast models used in era5_monthly v2.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_nwp_inference_efficiency.py \\
    --models pangu stormer fengwu fuxi aifs graphcast neuralgcm \\
    --era5-root /path/to/era5_np.25 --init-time 2025060100

  Multi-GPU (sequential, one model per GPU): bash scripts/run_benchmark_inference_efficiency_gpus_4_7.sh

  Operational reporting context (see ``--batch-size`` and ``*.meta.json``):
    2025 full year, 365 calendar days, 4 initial times per day, 40 forecast steps of 6 h per init.
    Per-step latency below uses the model's native 6 h update; ``batch_size`` is the spatial/instance
    batch in this microbenchmark (default 1), not the number of parallel inits.

Notes:
  - ``params_m`` column is **millions of scalar weights** (divide raw count by 1e6), not MiB.
  - Parameter counts: ONNX ``graph.initializer`` element products (matches external fp16/fp32 blobs);
    PyTorch ``sum(p.numel())``; JAX Haiku ``ckpt.params`` leaves only (GraphCast / NeuralGCM).
  - Latency is wall-clock after warmup; model/load/session creation is outside timed sections.
  - Per-model timed scope:
      pangu: one ORT call on fixed inputs (6h ONNX step).
      stormer: one _stormer_predict_residual(..., 6h); peak VRAM via torch.cuda.max_memory_allocated.
      fengwu: one ORT run on sliding window (curr reset each repeat).
      fuxi: one ORT run on ``short.onnx`` only (6 h cascade stepwise; params column = short only, ~1563M).
      aifs: one ``anemoi`` ``predict_step`` (6 h); ERA5→input tensor prep once outside timed loop.
      graphcast: rollout.chunked_prediction with 1 step; jax.block_until_ready(pred).
      neuralgcm: ``encode`` + ``unroll`` for native 6h lead only; ERA5 load + regrid + inputs_from_xarray
        are done once outside the timed loop (pure forward timing, analogous to dummy-input ONNX benches).

  - Hopper/H20 (e.g. sm90): unless you override env, we append ``--xla_gpu_enable_triton_gemm=false``
    to ``XLA_FLAGS`` and set ``CUDNN_FRONTEND_HEURISTIC_ENABLED=0`` before other imports load CUDA/JAX,
    to avoid XLA/cuDNN ``INTERNAL: unsupported value`` failures on some driver stacks.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os


def _apply_gpu_benchmark_runtime_defaults() -> None:
    """Defaults for JAX GraphCast/NeuralGCM on Hopper-class GPUs (see module docstring)."""
    extra_xla = "--xla_gpu_enable_triton_gemm=false"
    prev = os.environ.get("XLA_FLAGS", "").strip()
    if extra_xla not in prev:
        os.environ["XLA_FLAGS"] = f"{prev} {extra_xla}".strip()
    # Avoid cuDNN frontend picking unsupported algorithm configs on some stacks (NeuralGCM unroll).
    os.environ.setdefault("CUDNN_FRONTEND_HEURISTIC_ENABLED", "0")


_apply_gpu_benchmark_runtime_defaults()

import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS = ["pangu", "stormer", "fengwu", "fuxi", "aifs", "graphcast", "neuralgcm"]

# Reporting context for publications (matches operational NWP 2025 rollout scope).
OPERATIONAL_REF_YEAR = 2025
OPERATIONAL_CALENDAR_DAYS = 365
OPERATIONAL_INITS_PER_DAY = 4
OPERATIONAL_STEPS_PER_INIT = 40
OPERATIONAL_STEP_HOURS = 6


def _physical_gpu_index() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip().split(",")[0].strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def _nvidia_smi_used_mib(gpu_index: int) -> Optional[float]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return float(out.strip().split("\n")[0].strip())
    except Exception:
        return None


def _sample_peak_mib_during(
    gpu_index: int, fn: Callable[[], None], samples: int = 5, interval_s: float = 0.05
) -> Optional[float]:
    """Poll nvidia-smi while fn() runs (fn should be short)."""
    peak: Optional[float] = None
    t_end = time.perf_counter() + 30.0
    import threading

    done = threading.Event()

    def _poll() -> None:
        nonlocal peak
        while not done.is_set() and time.perf_counter() < t_end:
            v = _nvidia_smi_used_mib(gpu_index)
            if v is not None:
                peak = v if peak is None else max(peak, v)
            time.sleep(interval_s)

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()
    try:
        fn()
    finally:
        done.set()
        poller.join(timeout=2.0)
    return peak


def count_onnx_initializers_elements(paths: List[Path]) -> tuple[int, Dict[str, int]]:
    """Return total initializer element count per path (dims from protobuf; matches loaded weights)."""
    import onnx

    per: Dict[str, int] = {}
    total = 0
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(str(p))
        m = onnx.load(str(p), load_external_data=False)
        n = 0
        for init in m.graph.initializer:
            dims = [int(d) for d in init.dims]
            n += int(np.prod(dims)) if dims else 0
        per[p.name] = n
        total += n
    return total, per


def count_onnx_initializers_bytes(paths: List[Path]) -> int:
    """Sum ONNX initializer element counts (scalar weights/biases); dims read from model proto."""
    total, _ = count_onnx_initializers_elements(paths)
    return total


def count_jax_leaves(x: Any) -> int:
    import jax

    return int(sum(int(a.size) for a in jax.tree_util.tree_leaves(x)))


def _mean_std_ms(samples: List[float]) -> tuple[float, float]:
    if not samples:
        return float("nan"), float("nan")
    if len(samples) == 1:
        return samples[0] * 1000.0, 0.0
    ms = [s * 1000.0 for s in samples]
    return float(statistics.mean(ms)), float(statistics.stdev(ms))


def _collect_hardware_meta() -> Dict[str, Any]:
    """Best-effort host/GPU summary for tables like 'Efficiency comparisons'."""
    import platform

    out: Dict[str, Any] = {"python_platform": platform.platform()}
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        out["nvidia_smi_gpus"] = [ln.strip() for ln in smi.strip().splitlines() if ln.strip()]
    except Exception as e:
        out["nvidia_smi_gpus"] = []
        out["nvidia_smi_error"] = str(e)
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    out["host_ram_kib"] = int(parts[1])
                    break
    except Exception:
        pass
    try:
        lp = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL, timeout=5)
        for line in lp.splitlines():
            if line.startswith("Model name:"):
                out["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return out


@dataclass
class Row:
    model: str
    params_m: float
    inference_ms_mean: float
    inference_ms_std: float
    peak_gpu_mem_mib: Optional[float]
    gflops_per_step: Optional[float]
    notes: str
    error: str = ""
    batch_size: int = 1


def bench_pangu(
    *,
    weights_root: Path,
    era5_root: Path,
    init_time: datetime,
    warmup: int,
    repeats: int,
    gpu_index: int,
) -> Row:
    import onnxruntime as ort
    import torch
    from src.models.pangu_runner_class import MODEL_STEPS, _run_step, _session_options, _providers

    onnx_paths = [weights_root / "pangu" / MODEL_STEPS[6]]
    params = count_onnx_initializers_bytes(onnx_paths)
    path = onnx_paths[0]
    sess = ort.InferenceSession(
        str(path), sess_options=_session_options(), providers=_providers()
    )
    from src.common.data_reader import load_pangu_inputs

    upper, surface = load_pangu_inputs(init_time, root=era5_root, flip_north_south=False)
    upper = upper.astype(np.float32)
    surface = surface.astype(np.float32)

    def one_step() -> None:
        _run_step(sess, upper, surface)

    for _ in range(warmup):
        one_step()
    times: List[float] = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    mean_ms, std_ms = _mean_std_ms(times)
    peak = _sample_peak_mib_during(gpu_index, one_step)
    return Row(
        "pangu",
        params / 1e6,
        mean_ms,
        std_ms,
        peak,
        None,
        "single ONNX step pangu_weather_6.onnx (6h)",
    )


def bench_fengwu(
    *,
    weights_root: Path,
    era5_root: Path,
    init_time: datetime,
    onnx_name: str,
    warmup: int,
    repeats: int,
    gpu_index: int,
) -> Row:
    import torch
    from src.models.fengwu_runner import _get_session, _get_norm

    model_path = weights_root / "fengwu" / onnx_name
    params = count_onnx_initializers_bytes([model_path])
    session = _get_session(model_path)
    norm_dir = REPO_ROOT / "src" / "fengwu" / "normalization_constants"
    data_mean, data_std = _get_norm(norm_dir / "data_mean.npy", norm_dir / "data_std.npy")
    from src.common.data_reader import load_fengwu_init_pair

    input1, input2 = load_fengwu_init_pair(init_time, root=era5_root, flip_north_south=False)
    input1n = (input1 - data_mean) / data_std
    input2n = (input2 - data_mean) / data_std
    curr = np.concatenate([input1n, input2n], axis=0)[None, ...].astype(np.float32)

    def one_step() -> None:
        nonlocal curr
        out_arr = session.run(None, {"input": curr})[0]
        curr = np.concatenate((curr[:, 69:], out_arr[:, :69]), axis=1).astype(np.float32)

    # restore curr each timed iteration for comparable single-step cost
    base_curr = np.concatenate([input1n, input2n], axis=0)[None, ...].astype(np.float32)

    for _ in range(warmup):
        curr = base_curr.copy()
        one_step()
    times = []
    for _ in range(repeats):
        curr = base_curr.copy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    mean_ms, std_ms = _mean_std_ms(times)

    def peak_fn() -> None:
        c = base_curr.copy()
        session.run(None, {"input": c})

    peak = _sample_peak_mib_during(gpu_index, peak_fn)
    return Row(
        "fengwu",
        params / 1e6,
        mean_ms,
        std_ms,
        peak,
        None,
        f"single ORT step; onnx={onnx_name}",
    )


def bench_fuxi(
    *,
    weights_root: Path,
    era5_root: Path,
    init_time: datetime,
    warmup: int,
    repeats: int,
    gpu_index: int,
) -> Row:
    import onnxruntime as ort
    import pandas as pd
    import torch
    from src.models import fuxi_runner as fx

    short_onnx = weights_root / "fuxi" / "short.onnx"
    _, per_onnx = count_onnx_initializers_elements([short_onnx])
    params_short = per_onnx.get("short.onnx", 0)
    if not params_short:
        raise KeyError("short.onnx initializer count missing")

    total_steps = 1
    tembs = fx._time_encoding(pd.Timestamp(init_time), total_steps)
    from src.common.data_reader import load_fuxi_input_tensor

    input_np, _ = load_fuxi_input_tensor(init_time, root=era5_root, flip_north_south=False)
    sess = fx._session(short_onnx)

    def one_short_step() -> None:
        temb = tembs[0]
        sess.run(None, {"input": input_np, "temb": temb})

    for _ in range(warmup):
        one_short_step()
    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_short_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    mean_ms, std_ms = _mean_std_ms(times)
    peak = _sample_peak_mib_during(gpu_index, one_short_step)
    note = (
        f"6 h stepwise path: short.onnx only; params = short.onnx initializers (~{params_short/1e6:.0f}M scalars). "
        "medium/long not used in this row."
    )
    return Row(
        "fuxi",
        params_short / 1e6,
        mean_ms,
        std_ms,
        peak,
        None,
        note,
    )


def bench_aifs(
    *,
    weights_root: Path,
    era5_root: Path,
    init_time: datetime,
    warmup: int,
    repeats: int,
    gpu_index: int,
    checkpoint: str = "aifs-single-mse-1.1.ckpt",
) -> Row:
    import torch
    from src.models.aifs_runner import AifsForecastRunner

    impl = AifsForecastRunner(
        era5_root=era5_root,
        weights_root=weights_root / "aifs",
        checkpoint=checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    sr = impl._runner
    params = sum(p.numel() for p in sr.model.parameters())

    # ERA5 → model input fields (untimed; matches production path once per forecast init).
    input_state = impl._build_input_state_from_era5(init_time)
    # Runner.run() normally sets these before prepare_input_tensor — mirror that setup.
    sr.constant_forcings_inputs = sr.checkpoint.constant_forcings_inputs(sr, input_state)
    sr.dynamic_forcings_inputs = sr.checkpoint.dynamic_forcings_inputs(sr, input_state)
    sr.boundary_forcings_inputs = sr.checkpoint.boundary_forcings_inputs(sr, input_state)
    input_tensor_numpy = sr.prepare_input_tensor(input_state)

    input_tensor_torch = torch.from_numpy(
        np.swapaxes(input_tensor_numpy, -2, -1)[np.newaxis, ...]
    ).to(sr.device)

    dev_type = "cuda" if str(sr.device).startswith("cuda") else "cpu"
    autocast_dtype = sr.autocast

    def one_step() -> None:
        sr.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=dev_type, dtype=autocast_dtype):
                sr.predict_step(sr.model, input_tensor_torch, fcstep=0)
        if torch.cuda.is_available() and dev_type == "cuda":
            torch.cuda.synchronize()

    cuda_dev = input_tensor_torch.device

    for _ in range(warmup):
        one_step()
    times = []
    if torch.cuda.is_available() and dev_type == "cuda":
        torch.cuda.reset_peak_memory_stats(cuda_dev)
    for _ in range(repeats):
        t0 = time.perf_counter()
        one_step()
        times.append(time.perf_counter() - t0)
    mean_ms, std_ms = _mean_std_ms(times)
    peak_mib: Optional[float] = None
    if torch.cuda.is_available() and dev_type == "cuda":
        peak_mib = torch.cuda.max_memory_allocated(cuda_dev) / (1024.0**2)
    else:
        peak_mib = _sample_peak_mib_during(gpu_index, one_step)

    return Row(
        "aifs",
        params / 1e6,
        mean_ms,
        std_ms,
        peak_mib,
        None,
        f"single predict_step (6 h); Anemoi SimpleRunner; ckpt={checkpoint}; ERA5 prep once outside timed loop",
    )


def bench_stormer(
    *,
    weights_root: Path,
    era5_root: Path,
    init_time: datetime,
    warmup: int,
    repeats: int,
    gpu_index: int,
) -> Row:
    import torch
    from src.stormer.inference import load_model, variables
    from src.models.stormer_runner import (
        _load_norm_tensors,
        _prepare_input_tensor,
        _stormer_predict_residual,
        ensure_batch_bvhw,
        load_stormer_stack,
    )
    import src.stormer.inference as stormer_inference

    ckpt = weights_root / "stormer" / "stormer_1.40625_patch_size_2.ckpt"
    stormer_inference.WEIGHTS_FILE = Path(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    params = sum(p.numel() for p in model.parameters())
    mean_t, std_t, _ = _load_norm_tensors(device)
    mean_cpu = mean_t.detach().cpu()
    std_cpu = std_t.detach().cpu()
    raw = load_stormer_stack(init_time, root=era5_root, flip_north_south=False)
    inp_b = ensure_batch_bvhw(_prepare_input_tensor(raw, mean_cpu, std_cpu)).to(
        device=device, dtype=torch.float32
    )

    def one_step() -> None:
        with torch.no_grad():
            _stormer_predict_residual(model.net, inp_b, 6)

    for _ in range(warmup):
        one_step()
    times = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    mean_ms, std_ms = _mean_std_ms(times)
    peak_mib = None
    if torch.cuda.is_available():
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    else:
        peak_mib = _sample_peak_mib_during(gpu_index, one_step)

    return Row(
        "stormer",
        params / 1e6,
        mean_ms,
        std_ms,
        peak_mib,
        None,
        "single _stormer_predict_residual interval=6h",
    )


def bench_graphcast(
    *,
    weights_root: Path,
    era5_root: Path,
    init_time: datetime,
    warmup: int,
    repeats: int,
    gpu_index: int,
) -> Row:
    import jax
    from graphcast import rollout

    from src.models.graphcast_runner import (
        STEP_HOURS,
        _build_np25_input,
        _get_model_bundle,
        _get_predictor,
        _prepare_forcings_and_targets,
    )

    bundle = _get_model_bundle(weights_root)
    _params = bundle[0]
    task_config = bundle[3]
    params_n = count_jax_leaves(_params)

    ds_long = _build_np25_input(init_time, era5_root)
    steps = 1
    inputs, forcings, targets_template = _prepare_forcings_and_targets(
        ds_long, init_time, steps, task_config
    )
    run_forward_jitted = _get_predictor(weights_root)
    rng = jax.random.PRNGKey(0)

    def run_gc() -> None:
        pred = rollout.chunked_prediction(
            run_forward_jitted,
            rng=rng,
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
        )
        jax.block_until_ready(pred)

    for _ in range(warmup):
        run_gc()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_gc()
        times.append(time.perf_counter() - t0)
    mean_ms, std_ms = _mean_std_ms(times)
    peak = _sample_peak_mib_during(gpu_index, run_gc)
    return Row(
        "graphcast",
        params_n / 1e6,
        mean_ms,
        std_ms,
        peak,
        None,
        "chunked_prediction 1x6h; params = ckpt.params JAX leaves (~36M matches published GraphCast 0.25° 2–6 mesh)",
    )


def bench_neuralgcm(
    *,
    weights_root: Path,
    era5_root: Path,
    init_time: datetime,
    warmup: int,
    repeats: int,
    gpu_index: int,
) -> Row:
    import pickle

    import jax
    import pandas as pd

    from src.models.neuralgcm_runner import (
        _build_regridded_init_dataset,
        _load_model_cached,
    )

    ckpt_path = weights_root / "neuralgcm" / "models_v1_deterministic_0_7_deg.pkl"
    params_n = 0
    try:
        with open(ckpt_path, "rb") as f:
            raw_ckpt = pickle.load(f)
        if isinstance(raw_ckpt, dict) and "params" in raw_ckpt:
            params_n = count_jax_leaves(raw_ckpt["params"])
        else:
            logger.warning("NeuralGCM checkpoint missing params key; param count set to 0")
    except Exception as e:
        logger.warning("NeuralGCM param count failed: %s", e)

    model = _load_model_cached(weights_root)
    # Build inputs once (ERA5 read + conservative regrid): excluded from latency repeats.
    ds_input = _build_regridded_init_dataset(init_time, era5_root, model)
    t0_np = np.datetime64(pd.Timestamp(init_time))
    ds_init = ds_input.sel(time=t0_np, method="nearest")
    inputs = model.inputs_from_xarray(ds_init)
    input_forcings = model.forcings_from_xarray(ds_init)
    all_forcings = model.forcings_from_xarray(
        ds_input.sel(time=[t0_np], method="nearest")
    )

    lead_hours = 6
    dt_hours = float(model.timestep / np.timedelta64(1, "h"))
    n_steps_f = lead_hours / dt_hours
    max_steps = int(round(n_steps_f))
    if max_steps <= 0 or not np.isclose(n_steps_f, max_steps):
        raise ValueError(
            f"NeuralGCM benchmark: lead {lead_hours}h incompatible with model dt={dt_hours}h"
        )

    rng_key = jax.random.key(42)

    def run_infer() -> None:
        encoded_state = model.encode(inputs, input_forcings, rng_key)
        _, predictions = model.unroll(
            encoded_state,
            all_forcings,
            steps=max_steps,
            timedelta=model.timestep,
            start_with_input=False,
        )
        jax.block_until_ready(predictions)

    for _ in range(warmup):
        run_infer()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_infer()
        times.append(time.perf_counter() - t0)
    mean_ms, std_ms = _mean_std_ms(times)
    peak = _sample_peak_mib_during(gpu_index, run_infer)

    return Row(
        "neuralgcm",
        (params_n / 1e6) if params_n else float("nan"),
        mean_ms,
        std_ms,
        peak,
        None,
        "encode+unroll only (6h lead); ERA5/regrid/xarray prep once outside timed loop; "
        "params = pickle ckpt['params'] Haiku leaves (~31M)",
    )


BENCHERS: Dict[str, Any] = {
    "pangu": bench_pangu,
    "fengwu": bench_fengwu,
    "fuxi": bench_fuxi,
    "aifs": bench_aifs,
    "stormer": bench_stormer,
    "graphcast": bench_graphcast,
    "neuralgcm": bench_neuralgcm,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="NWP inference efficiency benchmark")
    ap.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=list(BENCHERS.keys()),
        help="Models to benchmark",
    )
    ap.add_argument("--weights-root", type=Path, default=None, help="Override NWP_WEIGHTS_ROOT")
    ap.add_argument("--era5-root", type=Path, default=None)
    ap.add_argument("--init-time", type=str, default="2025060100", help="YYYYMMDDHH")
    ap.add_argument("--fengwu-onnx", type=str, default="fengwu_v1.onnx")
    ap.add_argument(
        "--aifs-checkpoint",
        type=str,
        default="aifs-single-mse-1.1.ckpt",
        help="Filename under <weights-root>/aifs/ (e.g. aifs-single-mse-1.1.ckpt).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Spatial/instance batch for this microbenchmark (report as-measured; default 1).",
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help="If set, write nwp_inference_benchmark_<stem>.{csv,md,meta.json} instead of a timestamp stem",
    )
    args = ap.parse_args()

    if args.weights_root is None:
        args.weights_root = Path(
            os.environ.get("NWP_WEIGHTS_ROOT", "/ecmwf-era5-datasets/nwp_bench/assets/weights")
        )
    if args.era5_root is None:
        from src.common.data_reader import DEFAULT_ERA5_NPY_ROOT

        args.era5_root = DEFAULT_ERA5_NPY_ROOT

    init_time = datetime.strptime(args.init_time, "%Y%m%d%H")
    gpu_idx = _physical_gpu_index()
    out_dir = args.output_dir or (REPO_ROOT / "nwp_outputs" / "benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "physical_gpu_index_query": gpu_idx,
        "weights_root": str(args.weights_root),
        "era5_root": str(args.era5_root),
        "init_time": args.init_time,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "python": sys.version.split()[0],
        "operational_reference_scenario": {
            "year": OPERATIONAL_REF_YEAR,
            "calendar_days": OPERATIONAL_CALENDAR_DAYS,
            "initial_times_per_day": OPERATIONAL_INITS_PER_DAY,
            "steps_per_initial_time": OPERATIONAL_STEPS_PER_INIT,
            "step_hours": OPERATIONAL_STEP_HOURS,
            "native_step_hours_measured": 6,
            "total_six_hour_updates_if_full_rollout_sequential": OPERATIONAL_CALENDAR_DAYS
            * OPERATIONAL_INITS_PER_DAY
            * OPERATIONAL_STEPS_PER_INIT,
            "description": (
                f"{OPERATIONAL_REF_YEAR} full year; {OPERATIONAL_INITS_PER_DAY} inits/day; "
                f"{OPERATIONAL_STEPS_PER_INIT} steps × {OPERATIONAL_STEP_HOURS} h per init "
                "(table latency is one native 6 h forward per row unless noted)."
            ),
        },
        "hardware": _collect_hardware_meta(),
    }
    rows: List[Row] = []

    for name in args.models:
        logger.info("=== Benchmarking %s ===", name)
        try:
            fn = BENCHERS[name]
            row = replace(
                fn(
                    weights_root=args.weights_root,
                    era5_root=args.era5_root,
                    init_time=init_time,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    gpu_index=gpu_idx,
                **({"onnx_name": args.fengwu_onnx} if name == "fengwu" else {}),
                **({"checkpoint": args.aifs_checkpoint} if name == "aifs" else {}),
            ),
                batch_size=args.batch_size,
            )
            rows.append(row)
        except Exception as e:
            logger.exception("Model %s failed: %s", name, e)
            rows.append(
                Row(
                    name,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    None,
                    None,
                    "",
                    error=str(e),
                    batch_size=args.batch_size,
                )
            )

    ts = args.output_stem if args.output_stem else datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"nwp_inference_benchmark_{ts}.csv"
    md_path = out_dir / f"nwp_inference_benchmark_{ts}.md"
    meta_path = out_dir / f"nwp_inference_benchmark_{ts}.meta.json"

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "batch_size",
                "params_m",
                "inference_ms_mean",
                "inference_ms_std",
                "peak_gpu_mem_mib",
                "gflops_per_step",
                "notes",
                "error",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    lines = [
        "# NWP inference efficiency benchmark",
        "",
        f"Generated: {ts}" + (" (output-stem)" if args.output_stem else ""),
        "",
        "| Model | Batch | Parameters (M) | Inference ms (mean ± std) | Peak GPU MiB | GFLOPs/step | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.error:
            lines.append(
                f"| {r.model} | {r.batch_size} | — | — | — | — | **ERROR:** {r.error[:80]} |"
            )
            continue
        gfl = f"{r.gflops_per_step:.2f}" if r.gflops_per_step is not None else "—"
        peak = f"{r.peak_gpu_mem_mib:.0f}" if r.peak_gpu_mem_mib is not None else "—"
        lines.append(
            f"| {r.model} | {r.batch_size} | {r.params_m:.3f} | {r.inference_ms_mean:.2f} ± {r.inference_ms_std:.2f} | {peak} | {gfl} | {r.notes} |"
        )
    lines.extend(["", "## Environment", "", f"```json", json.dumps(meta, indent=2), "```"])
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()

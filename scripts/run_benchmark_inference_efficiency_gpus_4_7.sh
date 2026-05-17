#!/usr/bin/env bash
# Run the inference benchmark with **one model at a time**, each on a **dedicated physical GPU**
# from a small pool (default: 4,5,6,7). Models run **sequentially** (never two models on the same
# card at once). Round-robin assigns GPU 4→first model, 5→second, … wrapping after GPU 7.
#
# Usage:
#   bash scripts/run_benchmark_inference_efficiency_gpus_4_7.sh
#   WARMUP=2 REPEATS=5 BATCH_SIZE=1 bash scripts/run_benchmark_inference_efficiency_gpus_4_7.sh
#   BENCH_GPUS=6,7 MODELS="pangu stormer" bash scripts/run_benchmark_inference_efficiency_gpus_4_7.sh
# Parallelism: one subprocess per model (sequential); each uses one GPU from BENCH_GPUS (no two models on one GPU).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${NWP_PYTHON:-python3}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/nwp_outputs/benchmarks}"

# GraphCast/NeuralGCM on Hopper/H20: benchmark script also sets safe defaults; explicit exports optional:
#   export CUDNN_FRONTEND_HEURISTIC_ENABLED="${CUDNN_FRONTEND_HEURISTIC_ENABLED:-0}"
WARMUP="${WARMUP:-2}"
REPEATS="${REPEATS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"

# Comma-separated physical GPU indices (defaults to idle-tier 4–7 on shared nodes).
BENCH_GPUS="${BENCH_GPUS:-4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$BENCH_GPUS"

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "BENCH_GPUS is empty" >&2
  exit 1
fi

if [[ -n "${MODELS:-}" ]]; then
  read -r -a MODEL_LIST <<< "${MODELS}"
else
    MODEL_LIST=(pangu stormer fengwu fuxi aifs graphcast neuralgcm)
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

echo "RUN_ID=$RUN_ID  OUT_DIR=$OUT_DIR  BATCH_SIZE=$BATCH_SIZE  GPUs=${GPUS[*]}  models=${MODEL_LIST[*]}"

for i in "${!MODEL_LIST[@]}"; do
  m="${MODEL_LIST[$i]}"
  g="${GPUS[$((i % ${#GPUS[@]}))]}"
  STEM="${RUN_ID}__gpu${g}__${m}"
  echo ""
  echo "---------- ${m} on CUDA_VISIBLE_DEVICES=${g} (physical GPU ${g}) ----------"
  CUDA_VISIBLE_DEVICES="$g" "$PYTHON_BIN" "$REPO_ROOT/scripts/benchmark_nwp_inference_efficiency.py" \
    --models "$m" \
    --batch-size "$BATCH_SIZE" \
    --warmup "$WARMUP" \
    --repeats "$REPEATS" \
    --output-dir "$OUT_DIR" \
    --output-stem "$STEM"
done

MERGED_CSV="$OUT_DIR/nwp_inference_benchmark_merged_${RUN_ID}.csv"
MERGED_MD="$OUT_DIR/nwp_inference_benchmark_merged_${RUN_ID}.md"
MERGED_META="$OUT_DIR/nwp_inference_benchmark_merged_${RUN_ID}.meta.json"

first=true
for i in "${!MODEL_LIST[@]}"; do
  m="${MODEL_LIST[$i]}"
  g="${GPUS[$((i % ${#GPUS[@]}))]}"
  STEM="${RUN_ID}__gpu${g}__${m}"
  f="$OUT_DIR/nwp_inference_benchmark_${STEM}.csv"
  if [[ ! -f "$f" ]]; then
    echo "Missing shard CSV: $f" >&2
    exit 1
  fi
  if $first; then
    cp "$f" "$MERGED_CSV"
    first=false
  else
    tail -n +2 "$f" >> "$MERGED_CSV"
  fi
done

"$PYTHON_BIN" - "$MERGED_CSV" "$MERGED_MD" "$MERGED_META" "$RUN_ID" "$BENCH_GPUS" "$BATCH_SIZE" <<'PY'
import csv, json, sys
from pathlib import Path

merged_csv, merged_md, merged_meta, run_id, bench_gpus, batch_size = sys.argv[1:7]
rows = list(csv.DictReader(open(merged_csv, encoding="utf-8")))
lines = [
    "# NWP inference efficiency benchmark (merged, one model per GPU shard)",
    "",
    f"RUN_ID: {run_id}",
    f"BENCH_GPUS: {bench_gpus}",
    f"BATCH_SIZE (microbenchmark): {batch_size}",
    "",
    "Operational reference (meta.json on each shard): 2025 full year, 365 days, 4 inits/day, "
    "40 steps × 6 h per init.",
    "",
    "| Model | Batch | Parameters (M) | Inference ms (mean ± std) | Peak GPU MiB | GFLOPs/step | Notes |",
    "|---|---:|---:|---:|---:|---:|---|",
]
for r in rows:
    err = (r.get("error") or "").strip()
    bs = r.get("batch_size", batch_size)
    if err:
        lines.append(f"| {r['model']} | {bs} | — | — | — | — | **ERROR:** {err[:80]} |")
        continue
    raw_gfl = (r.get("gflops_per_step") or "").strip()
    gfl = raw_gfl if raw_gfl and raw_gfl.lower() != "nan" else "—"
    raw_peak = (r.get("peak_gpu_mem_mib") or "").strip()
    peak = raw_peak if raw_peak and raw_peak.lower() != "nan" else "—"
    notes = r.get("notes", "").replace("|", "\\|")
    lines.append(
        f"| {r['model']} | {bs} | {float(r['params_m']):.3f} | {float(r['inference_ms_mean']):.2f} ± {float(r['inference_ms_std']):.2f} | {peak} | {gfl} | {notes} |"
    )
Path(merged_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

meta = {
    "run_id": run_id,
    "bench_gpus": bench_gpus,
    "batch_size_reported": int(batch_size),
    "parallelism": "sequential subprocesses; one model at a time; one GPU per subprocess",
    "merged_csv": merged_csv,
    "shard_pattern": f"nwp_inference_benchmark_{run_id}__gpu*__*.csv",
}
Path(merged_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
PY

echo ""
echo "Merged CSV:  $MERGED_CSV"
echo "Merged MD:   $MERGED_MD"
echo "Merged meta: $MERGED_META"

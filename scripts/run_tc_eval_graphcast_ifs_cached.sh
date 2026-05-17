#!/usr/bin/env bash
# GraphCast IFS TC eval: LRU-open NetCDF handles per path (same idea as backfill_missing_metrics_from_nc_parallel).
# Slow NFS pays heavily for repeated xarray open_dataset on the same files.
#
# Usage:
#   bash scripts/run_tc_eval_graphcast_ifs_cached.sh
# Optional env overrides:
#   FORECAST_NC_CACHE_SIZE=96 bash scripts/run_tc_eval_graphcast_ifs_cached.sh
#   RESUME=1 bash scripts/run_tc_eval_graphcast_ifs_cached.sh   # resume (skip finished inits)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NWP_BENCHMARK_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

: "${PYTHON:=${NWP_PYTHON:-python3}}"
: "${PRIMARY_IFS_ROOT:=${NWP_FORECAST_ROOT:-${ROOT}/nwp_outputs/ifs_monthly_202506_v2/forecasts}}"
: "${OUT_ROOT:=${NWP_TC_EVAL_DIR:-${ROOT}/tc_eval_results/storm_centric}}"
: "${FORECAST_NC_CACHE_SIZE:=96}"
: "${RESUME:=0}"

EVAL_PY="${ROOT}/scripts/evaluate_tc_by_storm.py"

EXTRA_ARGS=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "yes" || "${RESUME}" == "true" ]]; then
  EXTRA_ARGS+=(--resume)
fi

export PYTHONUNBUFFERED=1
# Single-process eval: modest OMP default; set to 1 if you add outer parallelism.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

exec "${PYTHON}" -u "${EVAL_PY}" \
  --forecast-root "${PRIMARY_IFS_ROOT}" \
  --source ifs \
  --models graphcast \
  --forecast-nc-cache-size "${FORECAST_NC_CACHE_SIZE}" \
  --season 2025 \
  --start-date 2025-06-01 \
  --end-date 2025-12-31 \
  --out-dir "${OUT_ROOT}" \
  "${EXTRA_ARGS[@]}"

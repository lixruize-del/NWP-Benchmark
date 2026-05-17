#!/usr/bin/env bash
# Run NWP inference efficiency benchmark on a chosen GPU (default: device index 1).
# Override Python with NWP_PYTHON; forward extra args to the benchmark script.
# On busy nodes, prefer one-model-per-GPU sequential runs on GPUs 4–7:
#   bash scripts/run_benchmark_inference_efficiency_gpus_4_7.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHON_BIN="${NWP_PYTHON:-python3}"

exec "$PYTHON_BIN" "$REPO_ROOT/scripts/benchmark_nwp_inference_efficiency.py" "$@"

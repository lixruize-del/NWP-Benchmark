#!/usr/bin/env bash
#
# wb_spectra_era5_monthly_h2_job.sh — WeatherBench-like power spectra (ERA5-init monthly, H2 window)
#
# What it does
#   - Activates conda env CONDA_ENV (default: nwp_unified)
#   - cd to REPO_ROOT
#   - Runs scripts/plot_wb_like_spectra.py with ERA5-init forecast root, optional NAS fallback,
#     fixed H2 init window (2025070100–2025123118), resume-cache
#   - Tee stdout/stderr to LOG
#
# Required environment
#   REPO_ROOT  Absolute path to the NWP-Benchmark repository checkout.
#
# Optional environment (defaults in script body)
#   CONDA_SH, CONDA_ENV, FIG_DIR, LOG,
#   ERA5_MONTHLY_FORECAST, ERA5_MONTHLY_FALLBACK, GT_PRESSURE, GT_SINGLE
#
# Typical usage
#   export REPO_ROOT=/path/to/NWP-Benchmark
#   bash scripts/wb_spectra_era5_monthly_h2_job.sh

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to NWP-Benchmark repo root}"
: "${CONDA_ENV:=nwp_unified}"

if [[ -f "${CONDA_SH:-}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda not found; set CONDA_SH to conda.sh" >&2
  exit 1
fi

conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1

FIG_DIR="${FIG_DIR:-${REPO_ROOT}/nwp_outputs/ifs_monthly_202506_v2/figures}"
mkdir -p "${FIG_DIR}"

LOG="${LOG:-${FIG_DIR}/wb_like_spectra_h2_job.log}"

ERA5_MONTHLY_FORECAST="${ERA5_MONTHLY_FORECAST:-${REPO_ROOT}/nwp_outputs/era5_monthly_202506_v2/forecasts}"
GT_PRESSURE="${GT_PRESSURE:-/ecmwf-era5-datasets/era5_np.25/2025}"
GT_SINGLE="${GT_SINGLE:-/ecmwf-era5-datasets/era5_np.25/single/2025}"

echo "=== $(date -Is) start wb_spectra_era5_monthly_h2_job ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "CONDA_ENV=${CONDA_ENV}"

python scripts/plot_wb_like_spectra.py \
  --forecast-root "${ERA5_MONTHLY_FORECAST}" \
  --gt-pressure-root "${GT_PRESSURE}" \
  --gt-single-root "${GT_SINGLE}" \
  --models "aifs,aurora,fuxi,fengwu,pangu,graphcast,stormer,neuralgcm" \
  --variables "z500,q700,u850,t2m" \
  --lead-hours "6,72,120,240" \
  --init-start "2025070100" \
  --init-end "2025123118" \
  --output "${FIG_DIR}/wb_like_spectra_h2.png" \
  --cache-file "${FIG_DIR}/wb_like_spectra_h2.pkl" \
  --resume-cache \
  2>&1 | tee -a "${LOG}"

echo "=== $(date -Is) done ==="

# -----------------------------------------------------------------------------
# Example invocations (copy-paste)
# -----------------------------------------------------------------------------
#
# Minimal (set repo root only):
#   export REPO_ROOT=/path/to/NWP-Benchmark
#   bash scripts/wb_spectra_era5_monthly_h2_job.sh
#
# Custom figures dir and conda:
#   export REPO_ROOT=/path/to/NWP-Benchmark
#   export CONDA_ENV=nwp_unified
#   export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
#   export FIG_DIR=/path/to/out/figures
#   bash scripts/wb_spectra_era5_monthly_h2_job.sh
#
# Custom forecast root:
#   export REPO_ROOT=/path/to/NWP-Benchmark
#   export ERA5_MONTHLY_FORECAST=/path/to/forecasts
#   bash scripts/wb_spectra_era5_monthly_h2_job.sh

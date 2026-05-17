#!/usr/bin/env bash
#
# start_wb_spectra_era5_monthly_h2_tmux.sh — run wb_spectra_era5_monthly_h2_job.sh in detached tmux
#
# What it does
#   - Requires REPO_ROOT
#   - Refuses to start if tmux session SESSION (default: wb_spectra_era5_monthly_h2) exists
#   - Runs wb_spectra_era5_monthly_h2_job.sh with REPO_ROOT, conda, and optional path overrides
#
# Attach / logs
#   tmux attach -t wb_spectra_era5_monthly_h2
#   tail -f "${FIG_DIR:-$REPO_ROOT/nwp_outputs/ifs_monthly_202506_v2/figures}/wb_like_spectra_h2_job.log"

set -euo pipefail

SESSION="${SESSION:-wb_spectra_era5_monthly_h2}"

if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "Set REPO_ROOT to your NWP-Benchmark checkout, e.g." >&2
  echo "  export REPO_ROOT=/path/to/NWP-Benchmark" >&2
  exit 1
fi

JOB="${REPO_ROOT}/scripts/wb_spectra_era5_monthly_h2_job.sh"
if [[ ! -x "${JOB}" ]]; then
  chmod +x "${JOB}" || true
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session '${SESSION}' already exists. Attach with: tmux attach -t ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" \
  env REPO_ROOT="${REPO_ROOT}" \
      CONDA_SH="${CONDA_SH:-}" \
      CONDA_ENV="${CONDA_ENV:-nwp_unified}" \
      FIG_DIR="${FIG_DIR:-}" \
      ERA5_MONTHLY_FORECAST="${ERA5_MONTHLY_FORECAST:-}" \
      ERA5_MONTHLY_FALLBACK="${ERA5_MONTHLY_FALLBACK:-}" \
      GT_PRESSURE="${GT_PRESSURE:-}" \
      GT_SINGLE="${GT_SINGLE:-}" \
      bash "${JOB}"

echo "Started tmux session: ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
LOG_DEFAULT="${FIG_DIR:-${REPO_ROOT}/nwp_outputs/ifs_monthly_202506_v2/figures}/wb_like_spectra_h2_job.log"
echo "Log:    ${LOG_DEFAULT}"

# -----------------------------------------------------------------------------
# Example invocations (copy-paste)
# -----------------------------------------------------------------------------
#
#   export REPO_ROOT=/path/to/NWP-Benchmark
#   bash scripts/start_wb_spectra_era5_monthly_h2_tmux.sh
#
# With explicit conda.sh:
#   export REPO_ROOT=/path/to/NWP-Benchmark
#   export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
#   bash scripts/start_wb_spectra_era5_monthly_h2_tmux.sh
#
# Custom session name:
#   export REPO_ROOT=/path/to/NWP-Benchmark
#   export SESSION=wb_spectra_era5_h2_mine
#   bash scripts/start_wb_spectra_era5_monthly_h2_tmux.sh

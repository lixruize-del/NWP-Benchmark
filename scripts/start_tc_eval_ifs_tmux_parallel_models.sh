#!/usr/bin/env bash
set -euo pipefail

# Launch one tmux session per model for IFS storm-centric TC evaluation.
#
# Usage:
#   bash scripts/start_tc_eval_ifs_tmux_parallel_models.sh
#   MODELS="pangu stormer" START_DATE=2025-06-01 END_DATE=2025-12-31 \
#     bash scripts/start_tc_eval_ifs_tmux_parallel_models.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NWP_BENCHMARK_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${NWP_PYTHON:-python3}}"
EVAL_SCRIPT="${ROOT}/scripts/evaluate_tc_by_storm.py"

SOURCE="${SOURCE:-ifs}"
MODELS="${MODELS:-aifs aurora fengwu fuxi graphcast pangu stormer}"
SEASON="${SEASON:-2025}"
START_DATE="${START_DATE:-2025-06-01}"
END_DATE="${END_DATE:-2025-12-31}"

LEAD_STEP="${LEAD_STEP:-6}"
MAX_LEAD="${MAX_LEAD:-240}"
DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD:-560.0}"
WIND_THRESHOLD="${WIND_THRESHOLD:-8.0}"
RESOLUTION="${RESOLUTION:-0.25}"

PRIMARY_IFS_ROOT="${PRIMARY_IFS_ROOT:-${NWP_FORECAST_ROOT:-${ROOT}/nwp_outputs/ifs_monthly_202506_v2/forecasts}}"

OUT_ROOT="${OUT_ROOT:-${NWP_TC_EVAL_DIR:-${ROOT}/tc_eval_results/storm_centric}}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/$SOURCE/tmux_logs}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
TS="$(date +%Y%m%d_%H%M%S)"

for model in $MODELS; do
  SESSION_NAME="tc_${SOURCE}_${model}"
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "skip existing session: $SESSION_NAME"
    continue
  fi

  LOG_FILE="$LOG_ROOT/${model}_${TS}.log"

  CMD="$PYTHON_BIN -u \"$EVAL_SCRIPT\" \
    --forecast-root \"$PRIMARY_IFS_ROOT\" \
    --source \"$SOURCE\" \
    --models \"$MODEL\" \
    --season \"$SEASON\" \
    --start-date \"$START_DATE\" \
    --end-date \"$END_DATE\" \
    --lead-step \"$LEAD_STEP\" \
    --max-lead \"$MAX_LEAD\" \
    --distance-threshold \"$DISTANCE_THRESHOLD\" \
    --wind-threshold \"$WIND_THRESHOLD\" \
    --resolution \"$RESOLUTION\" \
    --out-dir \"$OUT_ROOT\""

  tmux new-session -d -s "$SESSION_NAME" \
    "cd \"$ROOT\" && echo \"$CMD\" && eval \"$CMD\" 2>&1 | tee -a \"$LOG_FILE\""
  echo "started $SESSION_NAME log=$LOG_FILE"
done

echo "list sessions: tmux ls | sed -n '/tc_${SOURCE}_/p'"
echo "attach one:    tmux attach -t tc_${SOURCE}_pangu"

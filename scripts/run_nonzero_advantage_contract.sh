#!/usr/bin/env bash
set -uo pipefail

MODE="${1:---dry-run}"
ROOT="${NONZERO_ADVANTAGE_CONTRACT_ROOT:-results/experiments/rl_nonzero_advantage_contract_20260802}"
FIXTURE="${NONZERO_ADVANTAGE_CONTRACT_FIXTURE:-results/inputs/rl_nonzero_advantage_contract_fixture_20260802.json}"
TASK_ID="${NONZERO_ADVANTAGE_CONTRACT_TASK_ID:-MM-E025}"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--run" ]]; then
  echo "usage: $0 [--dry-run|--run]" >&2
  exit 2
fi

if [[ "$MODE" == "--dry-run" ]]; then
  echo "nonzero-advantage contract dry run"
  echo "experiment root: $ROOT"
  echo "fixture: $FIXTURE"
  echo "GPU: disabled; task: $TASK_ID"
  exit 0
fi

if [[ ! -f "$FIXTURE" ]]; then
  echo "fixture is missing: $FIXTURE" >&2
  exit 2
fi
if [[ -e "$ROOT" && -n "$(find "$ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to reuse non-empty experiment root: $ROOT" >&2
  exit 2
fi

mkdir -p "$ROOT"
RUN_DIR="$ROOT/contract_replay"
if [[ -e "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to reuse non-empty run directory: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/run.log"
START_EPOCH="$(date +%s)"
CMD=(env CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python evaluation/audit_nonzero_advantage_contract.py
  --fixture "$FIXTURE" --output-dir "$RUN_DIR/audit")

(
  echo "STARTED_AT=$(date -Is)"
  echo "MODE=--run TASK_ID=$TASK_ID"
  echo "COMMAND=$(printf '%q ' "${CMD[@]}")"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "ENV_HASH=$(sha256sum align/rl_rules.py evaluation/audit_nonzero_advantage_contract.py scripts/run_nonzero_advantage_contract.sh "$FIXTURE" | sha256sum | cut -d' ' -f1)"
  echo "GPU_DISABLED=CUDA_VISIBLE_DEVICES=''"
  echo "DISK_BEFORE=$(df -h . | tail -1)"
  set +e
  "${CMD[@]}"
  STATUS=$?
  set -e
  echo "EXIT_CODE=$STATUS"
  echo "DISK_AFTER=$(df -h . | tail -1)"
  echo "FINISHED_AT=$(date -Is)"
  echo "GPU_WALL_SECONDS=0"
  echo "WALL_SECONDS=$(( $(date +%s) - START_EPOCH ))"
  exit "$STATUS"
) >"$LOG_PATH" 2>&1
exit $?

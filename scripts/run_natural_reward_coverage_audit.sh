#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
ROOT_DIR="${NATURAL_REWARD_COVERAGE_ROOT:-results/experiments/rl_natural_reward_coverage_audit_20260802}"
MANIFEST="results/inputs/rl_data_isolation_train_128_20260801.jsonl"
SAMPLE_ROOT="results/experiments/rl_natural_rule_reward_smoke_20260802"
MAX_GEN_LEN="${NATURAL_REWARD_COVERAGE_MAX_GEN_LEN:-16}"
HARD_LIMIT_SECONDS="${NATURAL_REWARD_COVERAGE_HARD_LIMIT_SECONDS:-600}"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--run" ]]; then
  echo "usage: $0 [--dry-run|--run]" >&2
  exit 2
fi
if [[ "$MODE" == "--dry-run" ]]; then
  echo "natural reward input/output coverage audit dry run"
  echo "output root: $ROOT_DIR"
  echo "manifest: $MANIFEST"
  echo "sample root: $SAMPLE_ROOT"
  echo "CUDA: disabled; GPU wall time: 0; max_gen_len: $MAX_GEN_LEN; hard wall limit seconds: $HARD_LIMIT_SECONDS"
  exit 0
fi
if [[ -e "$ROOT_DIR" && -n "$(find "$ROOT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to reuse non-empty audit root: $ROOT_DIR" >&2
  exit 2
fi
[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
[[ -e "$SAMPLE_ROOT" ]] || { echo "missing sample root: $SAMPLE_ROOT" >&2; exit 2; }
mkdir -p "$ROOT_DIR"
LOG_PATH="$ROOT_DIR/run.log"

CMD=(env CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python evaluation/audit_natural_reward_coverage.py
  --manifest "$MANIFEST"
  --sample-root "$SAMPLE_ROOT"
  --output-dir "$ROOT_DIR/audit"
  --max-gen-len "$MAX_GEN_LEN")

(
  echo "STARTED_AT=$(date -Is)"
  echo "TASK_IDS=MM-E029,MM-F037"
  echo "COMMAND=$(printf '%q ' "${CMD[@]}")"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "ENV_HASH=$(sha256sum evaluation/audit_natural_reward_coverage.py scripts/run_natural_reward_coverage_audit.sh "$MANIFEST" $(find "$SAMPLE_ROOT" -type f -name samples.jsonl -print | sort) | sha256sum | cut -d' ' -f1)"
  echo "GPU_BEFORE=$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  echo "DISK_BEFORE=$(df -h . | tail -1)"
  set +e
  timeout --signal=TERM "${HARD_LIMIT_SECONDS}s" "${CMD[@]}"
  STATUS=$?
  set -e
  echo "EXIT_CODE=$STATUS"
  echo "GPU_AFTER=$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  echo "DISK_AFTER=$(df -h . | tail -1)"
  echo "FINISHED_AT=$(date -Is)"
  echo "GPU_WALL_SECONDS=0"
  exit "$STATUS"
) >"$LOG_PATH" 2>&1

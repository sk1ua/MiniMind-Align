#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
ROOT="${BALANCED_OUTPUT_QUALITY_AUDIT_ROOT:-results/experiments/rl_balanced_output_quality_audit_20260802}"
AUDIT_DIR="$ROOT/audit"
MANIFEST="results/experiments/rl_balanced_reward_coverage_smoke_20260802/inputs/balanced_train_manifest.jsonl"
SAMPLE_ROOT="results/experiments/rl_balanced_reward_coverage_smoke_20260802/grpo_balanced_reward_coverage_seed42"
LOG_PATH="$ROOT/run.log"
RESOURCE_PATH="$ROOT/resource_monitor.jsonl"
METADATA_PATH="$ROOT/metadata.json"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--run" && "$MODE" != "--audit" ]]; then
  echo "usage: $0 --dry-run|--run" >&2
  exit 2
fi

if [[ "$MODE" == "--dry-run" ]]; then
  echo "balanced output-quality audit dry run"
  echo "manifest: $MANIFEST"
  echo "sample root: $SAMPLE_ROOT"
  echo "output root: $ROOT"
  echo "CUDA disabled; no GPU task; validator replay and category/step/prompt quality aggregation only"
  exit 0
fi

if [[ -e "$ROOT" && "$(find "$ROOT" -mindepth 1 -print -quit)" != "" ]]; then
  echo "refusing to reuse non-empty audit root: $ROOT" >&2
  exit 3
fi
mkdir -p "$ROOT"

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
COMMIT="$(git rev-parse HEAD)"
ENV_HASH="$(printf '%s\n' "$COMMIT" "$(.venv/bin/python --version 2>&1)" "$(.venv/bin/python -c 'import torch; print(torch.__version__)')" | sha256sum | awk '{print $1}')"
GPU_BEFORE="$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true)"
DISK_BEFORE="$(df -h / | tail -n 1)"
COMMAND="PYTHONPATH=. CUDA_VISIBLE_DEVICES='' .venv/bin/python evaluation/audit_balanced_output_quality.py --manifest $MANIFEST --sample-root $SAMPLE_ROOT --output-dir $AUDIT_DIR --max-gen-len 16"

{
  echo "STARTED_AT=$STARTED_AT"
  echo "TASK_ID=MM-E031"
  echo "COMMAND=$COMMAND"
  echo "GIT_COMMIT=$COMMIT"
  echo "ENV_HASH=$ENV_HASH"
  echo "CUDA_DISABLED=true"
  echo "GPU_BEFORE=$GPU_BEFORE"
  echo "DISK_BEFORE=$DISK_BEFORE"
} > "$LOG_PATH"

set +e
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python evaluation/audit_balanced_output_quality.py \
  --manifest "$MANIFEST" \
  --sample-root "$SAMPLE_ROOT" \
  --output-dir "$AUDIT_DIR" \
  --max-gen-len 16 >> "$LOG_PATH" 2>&1
EXIT_CODE=$?
set -e

GPU_AFTER="$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true)"
DISK_AFTER="$(df -h / | tail -n 1)"
FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  echo "FINISHED_AT=$FINISHED_AT"
  echo "EXIT_CODE=$EXIT_CODE"
  echo "GPU_AFTER=$GPU_AFTER"
  echo "DISK_AFTER=$DISK_AFTER"
  echo "GPU_WALL_SECONDS=0"
} >> "$LOG_PATH"

printf '{"timestamp":"%s","cuda_disabled":true,"gpu":"%s","disk":"%s"}\n' \
  "$FINISHED_AT" "${GPU_AFTER//$'\n'/ }" "${DISK_AFTER//$'\n'/ }" > "$RESOURCE_PATH"

cat > "$METADATA_PATH" <<EOF
{
  "task_ids": ["MM-E031", "MM-F039"],
  "experiment_root": "$ROOT",
  "manifest": "$MANIFEST",
  "sample_root": "$SAMPLE_ROOT",
  "commit": "$COMMIT",
  "environment_hash": "$ENV_HASH",
  "cuda_disabled": true,
  "gpu_wall_seconds": 0,
  "exit_code": $EXIT_CODE,
  "default_model_changed": false
}
EOF

exit "$EXIT_CODE"

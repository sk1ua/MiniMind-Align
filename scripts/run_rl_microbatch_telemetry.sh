#!/usr/bin/env bash
set -uo pipefail

MODE="${1:---dry-run}"
ROOT="${RL_MICROBATCH_ROOT:-results/experiments/rl_microbatch_telemetry_20260801}"
TRAIN_MANIFEST="results/inputs/rl_data_isolation_train_128_20260801.jsonl"
VALIDATION_MANIFEST="results/inputs/rl_data_isolation_validation_32_20260801.jsonl"
SESSION_NAME="${RL_MICROBATCH_TMUX_SESSION:-rl_microbatch_telemetry_20260801}"
HARD_LIMIT_SECONDS="${RL_MICROBATCH_HARD_LIMIT_SECONDS:-14400}"
TASK_ID="${RL_MICROBATCH_TASK_ID:-MM-E012}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
CUMULATIVE_SECONDS=0

if [[ "$MODE" != "--dry-run" && "$MODE" != "--smoke" && "$MODE" != "--diagnostic" ]]; then
  echo "usage: $0 [--dry-run|--smoke|--diagnostic]" >&2
  exit 2
fi

if [[ "$MODE" == "--dry-run" ]]; then
  echo "RL microbatch telemetry dry run: GRPO seed42, control, 4 steps"
  echo "experiment root: $ROOT"
  echo "train manifest: $TRAIN_MANIFEST"
  echo "validation manifest: $VALIDATION_MANIFEST"
  echo "hard limit seconds: $HARD_LIMIT_SECONDS"
  exit 0
fi

if [[ -z "${TMUX:-}" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required for --smoke/--diagnostic" >&2
    exit 2
  fi
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "refusing to reuse existing tmux session: $SESSION_NAME" >&2
    exit 2
  fi
  echo "starting detached tmux session: $SESSION_NAME"
  exec tmux new-session -d -s "$SESSION_NAME" "$SCRIPT_PATH" "$MODE"
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required for the hard safety limit" >&2
  exit 2
fi

if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$VALIDATION_MANIFEST" ]]; then
  echo "required isolated v2 manifests are missing" >&2
  exit 2
fi

if [[ -e "$ROOT" && ! -f "$ROOT/matrix.json" ]]; then
  echo "refusing to use an unrelated experiment root: $ROOT" >&2
  exit 2
fi

if [[ ! -e "$ROOT" ]]; then
  mkdir -p "$ROOT"
  .venv/bin/python - "$ROOT/matrix.json" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$MODE" "$TASK_ID" "$HARD_LIMIT_SECONDS" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
train_path = Path(sys.argv[2])
validation_path = Path(sys.argv[3])
mode = sys.argv[4]
task_id = sys.argv[5]
hard_limit_seconds = int(sys.argv[6])
train_rows = [line for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
validation_rows = [line for line in validation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(train_rows) != 128 or len(validation_rows) != 32:
    raise SystemExit(f"unexpected manifest sizes: train={len(train_rows)} validation={len(validation_rows)}")
payload = {
    "schema_version": 1,
    "experiment_root": str(output.parent),
    "task": task_id,
    "seed": 42,
    "mode": "grpo",
    "condition": "control",
    "run_mode": str(mode),
    "train_manifest": str(train_path),
    "validation_manifest": str(validation_path),
    "train_count": len(train_rows),
    "validation_count": len(validation_rows),
    "microbatch_schema_version": 3,
    "microbatch_gradient_method": "backward_gradient_delta_without_extra_backward",
    "hard_limit_seconds": hard_limit_seconds,
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
fi

run_one() {
  local output_dir="$1" save_weight="$2"
  shift 2
  local log_path="$output_dir/run.log"
  local monitor_path="$output_dir/resource_monitor.jsonl"
  if [[ -e "$output_dir" ]]; then
    echo "refusing to reuse existing experiment directory: $output_dir" >&2
    return 2
  fi
  mkdir -p "$output_dir"
  local remaining=$((HARD_LIMIT_SECONDS - CUMULATIVE_SECONDS))
  if (( remaining <= 0 )); then
    echo "hard GPU wall-time limit exhausted before $output_dir" >&2
    return 124
  fi
  local start_epoch
  start_epoch="$(date +%s)"
  local telemetry_path="$output_dir/microbatch_summaries.jsonl"
  local cmd=(.venv/bin/python trainer/train_grpo_lite.py
    --manifest "$TRAIN_MANIFEST"
    --validation-manifest "$VALIDATION_MANIFEST"
    --categories "conciseness,format,instruction,reasoning,repetition,safety,termination,uncertainty"
    --from-weight align_sft_v2_pilot
    --model-dir out --tokenizer-path model --batch-size 1
    --device cuda:0 --dtype bfloat16 --seed 42 --mode grpo
    --max-grad-norm 1.0
    --microbatch-log-path "$telemetry_path"
    --microbatch-gradient-norm
    --save-dir "$output_dir" --save-weight "$save_weight"
    "$@")
  (
    echo "STARTED_AT=$(date -Is)"
    echo "MODE=$MODE SEED=42"
    echo "COMMAND=$(printf '%q ' "${cmd[@]}")"
    echo "GIT_COMMIT=$(git rev-parse HEAD)"
    echo "ENV_HASH=$(sha256sum trainer/train_grpo_lite.py evaluation/audit_rl_spike_sources.py align/rl_rules.py scripts/run_rl_microbatch_telemetry.sh "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" | sha256sum | cut -d' ' -f1)"
    echo "GPU_BEFORE=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
    echo "DISK_BEFORE=$(df -h . | tail -1)"
    set +e
    timeout --signal=TERM "${remaining}s" "${cmd[@]}"
    status=$?
    set -e
    echo "EXIT_CODE=$status"
    echo "GPU_AFTER=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
    echo "DISK_AFTER=$(df -h . | tail -1)"
    echo "FINISHED_AT=$(date -Is)"
    echo "GPU_WALL_SECONDS=$(( $(date +%s) - start_epoch ))"
    exit "$status"
  ) >"$log_path" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    local gpu_snapshot disk_snapshot
    gpu_snapshot="$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ';' | sed 's/"/\\\"/g')"
    disk_snapshot="$(df -h . | tail -1 | sed 's/"/\\\"/g')"
    printf '{"timestamp":"%s","gpu":"%s","disk":"%s"}\n' "$(date -Is)" "$gpu_snapshot" "$disk_snapshot" >>"$monitor_path" || true
    sleep 60
  done
  wait "$pid"
  local status=$?
  local elapsed=$(( $(date +%s) - start_epoch ))
  CUMULATIVE_SECONDS=$((CUMULATIVE_SECONDS + elapsed))
  echo "RL_MICROBATCH_RUN exit_code=$status wall_seconds=$elapsed cumulative_seconds=$CUMULATIVE_SECONDS"
  return "$status"
}

run_audit() {
  local audit_dir="$1" run_dir="$2"
  if [[ ! -f "$run_dir/selection.json" ]]; then
    echo "selection.json missing; preserving run logs without audit" >&2
    return 1
  fi
  .venv/bin/python evaluation/audit_rl_spike_sources.py \
    --experiment-root "$ROOT" \
    --output-dir "$audit_dir" \
    --run-name "$(basename "$run_dir")" \
    --require-microbatch
}

overall=0
if [[ "$MODE" == "--smoke" ]]; then
  run_one "$ROOT/grpo_control_seed42" grpo_microbatch_smoke_seed42 \
    --max-prompts 8 --validation-max-prompts 2 --accumulation-steps 2 --num-generations 2 \
    --max-seq-len 384 --max-gen-len 16 --max-steps 2 --eval-every 2 --checkpoint-every 2 \
    --validation-max-new-tokens 16 --learning-rate 3e-7 || overall=1
  run_audit "$ROOT/audit_smoke" "$ROOT/grpo_control_seed42" || overall=1
else
  run_one "$ROOT/grpo_control_seed42_diagnostic_balanced" grpo_microbatch_control_seed42_balanced \
    --max-prompts 128 --validation-max-prompts 32 --accumulation-steps 8 --num-generations 8 \
    --max-seq-len 384 --max-gen-len 128 --max-steps 4 --eval-every 2 --checkpoint-every 4 \
    --validation-max-new-tokens 128 --learning-rate 3e-7 --interleave-categories || overall=1
  run_audit "$ROOT/audit_balanced" "$ROOT/grpo_control_seed42_diagnostic_balanced" || overall=1
fi

echo "RL_MICROBATCH_TOTAL_GPU_WALL_SECONDS=$CUMULATIVE_SECONDS"
if (( CUMULATIVE_SECONDS > HARD_LIMIT_SECONDS )); then
  echo "hard GPU wall-time limit exceeded: $CUMULATIVE_SECONDS > $HARD_LIMIT_SECONDS" >&2
  overall=1
fi
exit "$overall"

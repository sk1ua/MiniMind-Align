#!/usr/bin/env bash
set -uo pipefail

MODE="${1:---dry-run}"
ROOT="results/experiments/rl_stability_diagnostic_20260801"
TRAIN_MANIFEST="results/inputs/rl_data_isolation_train_128_20260801.jsonl"
VALIDATION_MANIFEST="results/inputs/rl_data_isolation_validation_32_20260801.jsonl"
SESSION_NAME="${RL_STABILITY_TMUX_SESSION:-rl_stability_diagnostic_20260801}"
BUDGET_SECONDS="${RL_STABILITY_BUDGET_SECONDS:-7200}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
CUMULATIVE_SECONDS=0

if [[ "$MODE" != "--dry-run" && "$MODE" != "--smoke" && "$MODE" != "--full" ]]; then
  echo "usage: $0 [--dry-run|--smoke|--full]" >&2
  exit 2
fi

if [[ "$MODE" == "--dry-run" ]]; then
  echo "RL stability diagnostic dry run: GRPO/CISPO seed42, control/low_lr/accum16, max 20 steps"
  echo "experiment root: $ROOT"
  echo "train manifest: $TRAIN_MANIFEST"
  echo "validation manifest: $VALIDATION_MANIFEST"
  echo "budget seconds: $BUDGET_SECONDS"
  exit 0
fi

if [[ -z "${TMUX:-}" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required for --smoke/--full" >&2
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
  echo "GNU timeout is required for the 2-hour budget guard" >&2
  exit 2
fi

if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$VALIDATION_MANIFEST" ]]; then
  echo "required isolated v2 manifests are missing" >&2
  exit 2
fi

if [[ -e "$ROOT/matrix.json" ]]; then
  echo "reusing existing matrix metadata: $ROOT/matrix.json"
else
  mkdir -p "$ROOT"
  .venv/bin/python - "$ROOT/matrix.json" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

output, train_path, validation_path = map(Path, sys.argv[1:])
train_rows = [line for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
validation_rows = [line for line in validation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(train_rows) != 128 or len(validation_rows) != 32:
    raise SystemExit(f"unexpected manifest sizes: train={len(train_rows)} validation={len(validation_rows)}")
payload = {
    "schema_version": 1,
    "experiment_root": "results/experiments/rl_stability_diagnostic_20260801",
    "seed": 42,
    "modes": ["grpo", "cispo"],
    "conditions": {
        "control": {"learning_rate": 3e-7, "accumulation_steps": 8, "max_grad_norm": 1.0},
        "low_lr": {"learning_rate": 1e-7, "accumulation_steps": 8, "max_grad_norm": 1.0},
        "accum16": {"learning_rate": 3e-7, "accumulation_steps": 16, "max_grad_norm": 1.0},
    },
    "train_manifest": str(train_path),
    "validation_manifest": str(validation_path),
    "train_count": len(train_rows),
    "validation_count": len(validation_rows),
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
fi

run_one() {
  local mode="$1" condition="$2" output_dir="$3" save_weight="$4"
  shift 4
  local log_path="$output_dir/run.log"
  local monitor_path="$output_dir/resource_monitor.jsonl"
  if [[ -e "$output_dir" ]]; then
    echo "refusing to reuse existing experiment directory: $output_dir" >&2
    return 2
  fi
  mkdir -p "$output_dir"
  local remaining=$((BUDGET_SECONDS - CUMULATIVE_SECONDS))
  if (( remaining <= 0 )); then
    echo "GPU budget exhausted before $output_dir" >&2
    return 124
  fi
  local start_epoch
  start_epoch="$(date +%s)"
  local cmd=(.venv/bin/python trainer/train_grpo_lite.py
    --manifest "$TRAIN_MANIFEST"
    --validation-manifest "$VALIDATION_MANIFEST"
    --categories "conciseness,format,instruction,reasoning,repetition,safety,termination,uncertainty"
    --from-weight align_sft_v2_pilot
    --model-dir out --tokenizer-path model --batch-size 1
    --device cuda:0 --dtype bfloat16 --seed 42 --mode "$mode"
    --max-grad-norm 1.0
    --save-dir "$output_dir" --save-weight "$save_weight"
    "$@")
  (
    echo "STARTED_AT=$(date -Is)"
    echo "MODE=$mode CONDITION=$condition SEED=42"
    echo "COMMAND=$(printf '%q ' "${cmd[@]}")"
    echo "GIT_COMMIT=$(git rev-parse HEAD)"
    echo "ENV_HASH=$(sha256sum trainer/train_grpo_lite.py align/rl_rules.py evaluation/rl_validation.py evaluation/rl_selection.py evaluation/audit_rl_stability.py scripts/run_rl_stability_diagnostic.sh "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" | sha256sum | cut -d' ' -f1)"
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
    gpu_snapshot="$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ';' | sed 's/"/\\"/g')"
    disk_snapshot="$(df -h . | tail -1 | sed 's/"/\\"/g')"
    printf '{"timestamp":"%s","gpu":"%s","disk":"%s"}\n' "$(date -Is)" "$gpu_snapshot" "$disk_snapshot" >>"$monitor_path" || true
    sleep 60
  done
  wait "$pid"
  local status=$?
  local elapsed=$(( $(date +%s) - start_epoch ))
  CUMULATIVE_SECONDS=$((CUMULATIVE_SECONDS + elapsed))
  echo "RL_STABILITY_RUN mode=$mode condition=$condition exit_code=$status wall_seconds=$elapsed cumulative_seconds=$CUMULATIVE_SECONDS"
  return "$status"
}

run_audit() {
  local audit_dir="$1"
  .venv/bin/python evaluation/audit_rl_stability.py \
    --experiment-root "$ROOT" \
    --output-dir "$audit_dir"
}

mkdir -p "$ROOT"
overall=0

if [[ "$MODE" == "--smoke" ]]; then
  run_one grpo smoke_control "$ROOT/grpo_smoke_control_seed42" grpo_smoke_control_seed42 \
    --max-prompts 8 --validation-max-prompts 2 --accumulation-steps 1 --num-generations 2 \
    --max-seq-len 384 --max-gen-len 16 --max-steps 2 --eval-every 2 --checkpoint-every 2 \
    --validation-max-new-tokens 16 --learning-rate 3e-7 || overall=1
  run_audit "$ROOT/audit_smoke" || overall=1
else
  for mode in grpo cispo; do
    for condition in control low_lr accum16; do
      case "$condition" in
        control)
          learning_rate=3e-7
          accumulation_steps=8
          ;;
        low_lr)
          learning_rate=1e-7
          accumulation_steps=8
          ;;
        accum16)
          learning_rate=3e-7
          accumulation_steps=16
          ;;
      esac
      run_one "$mode" "$condition" "$ROOT/${mode}_${condition}_seed42" "${mode}_${condition}_seed42" \
        --max-prompts 128 --validation-max-prompts 32 --accumulation-steps "$accumulation_steps" --num-generations 8 \
        --max-seq-len 384 --max-gen-len 128 --max-steps 20 --eval-every 4 --checkpoint-every 4 \
        --validation-max-new-tokens 128 --learning-rate "$learning_rate" || overall=1
      if (( CUMULATIVE_SECONDS >= BUDGET_SECONDS )); then
        echo "GPU budget exhausted; stopping before the next condition" >&2
        break 2
      fi
    done
  done
  run_audit "$ROOT/stability_audit" || overall=1
fi

echo "RL_STABILITY_TOTAL_GPU_WALL_SECONDS=$CUMULATIVE_SECONDS"
if (( CUMULATIVE_SECONDS > BUDGET_SECONDS )); then
  echo "GPU budget exceeded: $CUMULATIVE_SECONDS > $BUDGET_SECONDS" >&2
  overall=1
fi
exit "$overall"

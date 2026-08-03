#!/usr/bin/env bash
set -uo pipefail

MODE="${1:---dry-run}"
ROOT="results/experiments/rl_update_scale_diagnostic_20260801"
SMOKE_ROOT="$ROOT/smoke"
FORMAL_ROOT="$ROOT/formal"
TRAIN_MANIFEST="results/inputs/rl_data_isolation_train_128_20260801.jsonl"
VALIDATION_MANIFEST="results/inputs/rl_data_isolation_validation_32_20260801.jsonl"
SESSION_NAME="${RL_UPDATE_SCALE_TMUX_SESSION:-rl_update_scale_diagnostic_20260801}"
BUDGET_SECONDS="${RL_UPDATE_SCALE_BUDGET_SECONDS:-7200}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
CUMULATIVE_SECONDS=0

if [[ "$MODE" != "--dry-run" && "$MODE" != "--smoke" && "$MODE" != "--full" ]]; then
  echo "usage: $0 [--dry-run|--smoke|--full]" >&2
  exit 2
fi

if [[ "$MODE" == "--dry-run" ]]; then
  echo "RL update-scale diagnostic dry run: GRPO seed42, control/low_lr/clip_half"
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
  echo "GNU timeout is required for the budget guard" >&2
  exit 2
fi

if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$VALIDATION_MANIFEST" ]]; then
  echo "required isolated v2 manifests are missing" >&2
  exit 2
fi

if [[ -e "$ROOT" && ! -e "$ROOT/matrix.json" ]]; then
  if [[ -n "$(find "$ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "refusing to reuse non-empty experiment root without matrix metadata: $ROOT" >&2
    exit 2
  fi
fi
mkdir -p "$ROOT"

if [[ -e "$ROOT/matrix.json" ]]; then
  echo "reusing existing matrix metadata: $ROOT/matrix.json"
else
  .venv/bin/python - "$ROOT/matrix.json" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

output, train_path, validation_path = map(Path, sys.argv[1:])
train_count = sum(1 for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip())
validation_count = sum(1 for line in validation_path.read_text(encoding="utf-8").splitlines() if line.strip())
if train_count != 128 or validation_count != 32:
    raise SystemExit(f"unexpected manifest sizes: train={train_count} validation={validation_count}")
payload = {
    "schema_version": 1,
    "experiment_root": "results/experiments/rl_update_scale_diagnostic_20260801",
    "seed": 42,
    "mode": "grpo",
    "conditions": {
        "control": {"learning_rate": 3e-7, "accumulation_steps": 8, "max_grad_norm": 1.0},
        "low_lr": {"learning_rate": 1e-7, "accumulation_steps": 8, "max_grad_norm": 1.0},
        "clip_half": {"learning_rate": 3e-7, "accumulation_steps": 8, "max_grad_norm": 0.5},
    },
    "formal_max_steps": 8,
    "train_manifest": str(train_path),
    "validation_manifest": str(validation_path),
    "train_count": train_count,
    "validation_count": validation_count,
    "diagnostic_only": True,
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
fi

run_one() {
  local mode="$1"
  local condition="$2"
  local output_dir="$3"
  local save_weight="$4"
  local learning_rate="$5"
  local accumulation_steps="$6"
  local max_grad_norm="$7"
  local max_steps="$8"
  local max_prompts="$9"
  local validation_max_prompts="${10}"
  local num_generations="${11}"
  local max_gen_len="${12}"
  local eval_every="${13}"
  local checkpoint_every="${14}"
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
    --max-grad-norm "$max_grad_norm"
    --microbatch-log-path "$output_dir/microbatch_summaries.jsonl"
    --microbatch-gradient-norm
    --interleave-categories
    --save-dir "$output_dir" --save-weight "$save_weight"
    --max-prompts "$max_prompts" --validation-max-prompts "$validation_max_prompts"
    --accumulation-steps "$accumulation_steps" --num-generations "$num_generations"
    --max-seq-len 384 --max-gen-len "$max_gen_len" --max-steps "$max_steps"
    --eval-every "$eval_every" --checkpoint-every "$checkpoint_every"
    --validation-max-new-tokens "$max_gen_len" --learning-rate "$learning_rate"
    --beta 0.02 --epsilon 0.2 --epsilon-high 5.0
    --kl-threshold 0.005 --kl-patience 2 --quality-drop-points 10)
  (
    echo "STARTED_AT=$(date -Is)"
    echo "MODE=$mode CONDITION=$condition SEED=42"
    echo "COMMAND=$(printf '%q ' "${cmd[@]}")"
    echo "GIT_COMMIT=$(git rev-parse HEAD)"
    echo "ENV_HASH=$(sha256sum trainer/train_grpo_lite.py align/rl_rules.py evaluation/audit_rl_stability.py evaluation/audit_rl_spike_sources.py scripts/run_rl_update_scale_diagnostic.sh "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" | sha256sum | cut -d' ' -f1)"
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
  status=$?
  local elapsed=$(( $(date +%s) - start_epoch ))
  CUMULATIVE_SECONDS=$((CUMULATIVE_SECONDS + elapsed))
  echo "RL_UPDATE_SCALE_RUN mode=$mode condition=$condition exit_code=$status wall_seconds=$elapsed cumulative_seconds=$CUMULATIVE_SECONDS"
  return "$status"
}

audit_root() {
  local root="$1"
  local label="$2"
  .venv/bin/python evaluation/audit_rl_stability.py \
    --experiment-root "$root" \
    --output-dir "$root/stability_audit"
  local conditions=(control low_lr clip_half)
  if [[ "$label" == "smoke" ]]; then
    conditions=(control)
  fi
  for condition in "${conditions[@]}"; do
    .venv/bin/python evaluation/audit_rl_spike_sources.py \
      --experiment-root "$root" \
      --output-dir "$root/spike_audit_${condition}" \
      --run-name "grpo_${condition}_seed42" \
      --require-microbatch
  done
  echo "RL_UPDATE_SCALE_AUDIT label=$label root=$root"
}

overall=0
if [[ "$MODE" == "--smoke" ]]; then
  run_one grpo control "$SMOKE_ROOT/grpo_control_seed42" grpo_update_scale_smoke_seed42 \
    3e-7 2 1.0 2 8 2 2 16 2 2 || overall=1
  audit_root "$SMOKE_ROOT" smoke || overall=1
else
  for condition in control low_lr clip_half; do
    case "$condition" in
      control)
        learning_rate=3e-7
        accumulation_steps=8
        max_grad_norm=1.0
        ;;
      low_lr)
        learning_rate=1e-7
        accumulation_steps=8
        max_grad_norm=1.0
        ;;
      clip_half)
        learning_rate=3e-7
        accumulation_steps=8
        max_grad_norm=0.5
        ;;
    esac
    run_one grpo "$condition" "$FORMAL_ROOT/grpo_${condition}_seed42" "grpo_update_scale_${condition}_seed42" \
      "$learning_rate" "$accumulation_steps" "$max_grad_norm" 8 128 32 8 128 2 4 || overall=1
    if (( CUMULATIVE_SECONDS >= BUDGET_SECONDS )); then
      echo "GPU budget exhausted; stopping before the next condition" >&2
      break
    fi
  done
  if [[ -d "$FORMAL_ROOT" ]]; then
    audit_root "$FORMAL_ROOT" formal || overall=1
  fi
fi

echo "RL_UPDATE_SCALE_TOTAL_GPU_WALL_SECONDS=$CUMULATIVE_SECONDS"
if (( CUMULATIVE_SECONDS > BUDGET_SECONDS )); then
  echo "GPU budget exceeded: $CUMULATIVE_SECONDS > $BUDGET_SECONDS" >&2
  overall=1
fi
exit "$overall"

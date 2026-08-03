#!/usr/bin/env bash
set -uo pipefail

MODE="${1:---dry-run}"
ROOT="${RL_GUARD_TELEMETRY_ROOT:-results/experiments/rl_kl_guard_telemetry_v2_20260802}"
TRAIN_MANIFEST="results/inputs/rl_data_isolation_train_128_20260801.jsonl"
VALIDATION_MANIFEST="results/inputs/rl_data_isolation_validation_32_20260801.jsonl"
SESSION_NAME="${RL_GUARD_TELEMETRY_TMUX_SESSION:-rl_kl_guard_telemetry_smoke_20260802}"
HARD_LIMIT_SECONDS="${RL_GUARD_TELEMETRY_HARD_LIMIT_SECONDS:-1800}"
TASK_ID="${RL_GUARD_TELEMETRY_TASK_ID:-MM-E018}"
FULL_FP32_DIAGNOSTIC="${RL_GUARD_TELEMETRY_FULL_FP32:-0}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--smoke" ]]; then
  echo "usage: $0 [--dry-run|--smoke]" >&2
  exit 2
fi

if [[ "$MODE" == "--dry-run" ]]; then
  echo "RL KL guard telemetry smoke dry run"
  echo "experiment root: $ROOT"
  echo "mode: GRPO seed=42"
  echo "post-step KL target: 0.005"
  echo "measurement: bfloat16 production + float32 no-autocast diagnostic + full-float32-copy=${FULL_FP32_DIAGNOSTIC}"
  echo "hard limit seconds: $HARD_LIMIT_SECONDS"
  exit 0
fi

if [[ -z "${TMUX:-}" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required for --smoke" >&2
    exit 2
  fi
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "refusing to reuse existing tmux session: $SESSION_NAME" >&2
    exit 2
  fi
  echo "starting detached tmux session: $SESSION_NAME"
  exec tmux new-session -d -s "$SESSION_NAME" "$SCRIPT_PATH --smoke"
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$VALIDATION_MANIFEST" ]]; then
  echo "required isolated v2 manifests are missing" >&2
  exit 2
fi
if [[ -e "$ROOT" && -n "$(find "$ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to reuse non-empty experiment root: $ROOT" >&2
  exit 2
fi

mkdir -p "$ROOT"
.venv/bin/python - "$ROOT/matrix.json" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$TASK_ID" "$HARD_LIMIT_SECONDS" "$FULL_FP32_DIAGNOSTIC" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
train = Path(sys.argv[2])
validation = Path(sys.argv[3])
task_id = sys.argv[4]
hard_limit = int(sys.argv[5])
full_fp32_diagnostic = bool(int(sys.argv[6]))
train_count = sum(bool(line.strip()) for line in train.read_text(encoding="utf-8").splitlines())
validation_count = sum(bool(line.strip()) for line in validation.read_text(encoding="utf-8").splitlines())
if train_count != 128 or validation_count != 32:
    raise SystemExit(f"unexpected manifest sizes: train={train_count} validation={validation_count}")
payload = {
    "schema_version": 1,
    "experiment_root": str(output.parent),
    "task": task_id,
    "mode": "grpo",
    "seed": 42,
    "train_manifest": str(train),
    "validation_manifest": str(validation),
    "train_count": train_count,
    "validation_count": validation_count,
    "smoke": {"max_steps": 2, "max_prompts": 8, "validation_max_prompts": 2, "num_generations": 2, "accumulation_steps": 2, "max_seq_len": 384, "max_gen_len": 16},
    "guard": {"post_step_kl_target": 0.005, "backoff_factor": 0.5, "max_backoffs": 3, "production_dtype": "bfloat16", "diagnostic_dtype": "float32_no_autocast_bfloat16_weights", "full_fp32_copy_diagnostic": full_fp32_diagnostic, "failure_policy": "rollback_and_stop"},
    "hard_limit_seconds": hard_limit,
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

RUN_DIR="$ROOT/grpo_guard_telemetry_smoke_seed42"
if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to reuse existing run directory: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/run.log"
MONITOR_PATH="$RUN_DIR/resource_monitor.jsonl"
ATTEMPT_LOG_PATH="$RUN_DIR/kl_guard_attempts.jsonl"
START_EPOCH="$(date +%s)"
CMD=(.venv/bin/python trainer/train_grpo_lite.py
  --manifest "$TRAIN_MANIFEST"
  --validation-manifest "$VALIDATION_MANIFEST"
  --categories "conciseness,format,instruction,reasoning,repetition,safety,termination,uncertainty"
  --from-weight align_sft_v2_pilot --model-dir out --tokenizer-path model --batch-size 1
  --device cuda:0 --dtype bfloat16 --seed 42 --mode grpo
  --max-prompts 8 --validation-max-prompts 2 --num-generations 2 --accumulation-steps 2
  --max-seq-len 384 --max-gen-len 16 --max-steps 2 --eval-every 2 --checkpoint-every 2
  --validation-max-new-tokens 16 --learning-rate 3e-7 --max-grad-norm 1.0
  --post-step-kl-target 0.005 --kl-backoff-factor 0.5 --kl-max-backoffs 3
  --post-step-kl-diagnostic-fp32 --kl-guard-attempt-log-path "$ATTEMPT_LOG_PATH"
  --microbatch-log-path "$RUN_DIR/microbatch_summaries.jsonl" --microbatch-gradient-norm
  --interleave-categories --save-dir "$RUN_DIR" --save-weight grpo_kl_guard_telemetry_smoke_seed42)
if [[ "$FULL_FP32_DIAGNOSTIC" == "1" ]]; then
  CMD+=(--post-step-kl-diagnostic-full-fp32)
fi

(
  echo "STARTED_AT=$(date -Is)"
  echo "MODE=--smoke TASK_ID=$TASK_ID"
  echo "COMMAND=$(printf '%q ' "${CMD[@]}")"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "ENV_HASH=$(sha256sum trainer/train_grpo_lite.py align/rl_rules.py evaluation/audit_rl_kl_guard_telemetry.py scripts/run_rl_kl_guard_telemetry_smoke.sh "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" | sha256sum | cut -d' ' -f1)"
  echo "GPU_BEFORE=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
  echo "DISK_BEFORE=$(df -h . | tail -1)"
  set +e
  timeout --signal=TERM "${HARD_LIMIT_SECONDS}s" "${CMD[@]}"
  STATUS=$?
  set -e
  echo "EXIT_CODE=$STATUS"
  echo "GPU_AFTER=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
  echo "DISK_AFTER=$(df -h . | tail -1)"
  echo "FINISHED_AT=$(date -Is)"
  echo "GPU_WALL_SECONDS=$(( $(date +%s) - START_EPOCH ))"
  exit "$STATUS"
) >"$LOG_PATH" 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
  GPU_SNAPSHOT="$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ';' | sed 's/"/\\"/g')"
  DISK_SNAPSHOT="$(df -h . | tail -1 | sed 's/"/\\"/g')"
  printf '{"timestamp":"%s","gpu":"%s","disk":"%s"}\n' "$(date -Is)" "$GPU_SNAPSHOT" "$DISK_SNAPSHOT" >>"$MONITOR_PATH"
  AVAILABLE_KIB="$(df -Pk . | awk 'NR==2 {print $4}')"
  if [[ -n "$AVAILABLE_KIB" && "$AVAILABLE_KIB" -lt $((80 * 1024 * 1024)) ]]; then
    echo "disk safety threshold reached; terminating smoke" >>"$LOG_PATH"
    kill -TERM "$PID" 2>/dev/null || true
    break
  fi
  sleep 60
done
wait "$PID"
RUN_STATUS=$?

AUDIT_DIR="$ROOT/audit_smoke"
if [[ -f "$RUN_DIR/selection.json" ]]; then
  .venv/bin/python evaluation/audit_rl_kl_guard_telemetry.py \
    --experiment-root "$ROOT" --output-dir "$AUDIT_DIR" \
    --run-name "$(basename "$RUN_DIR")"
else
  echo "selection.json missing; preserving run logs without audit" >&2
  exit "$RUN_STATUS"
fi
echo "RL_KL_GUARD_TELEMETRY_SMOKE_GPU_WALL_SECONDS=$(( $(date +%s) - START_EPOCH ))"
exit "$RUN_STATUS"

#!/usr/bin/env bash
set -uo pipefail

MODE="${1:---dry-run}"
ROOT="${CORRECTED_KL_GATE_ROOT:-results/experiments/rl_corrected_kl_gate_smoke_20260802}"
TRAIN_MANIFEST="results/inputs/rl_data_isolation_train_128_20260801.jsonl"
VALIDATION_MANIFEST="results/inputs/rl_data_isolation_validation_32_20260801.jsonl"
SESSION_NAME="${CORRECTED_KL_GATE_TMUX_SESSION:-rl_corrected_kl_gate_smoke_20260802}"
HARD_LIMIT_SECONDS="${CORRECTED_KL_GATE_HARD_LIMIT_SECONDS:-1800}"
TASK_ID="${CORRECTED_KL_GATE_TASK_ID:-MM-E022}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--smoke" ]]; then
  echo "usage: $0 [--dry-run|--smoke]" >&2
  exit 2
fi

if [[ "$MODE" == "--dry-run" ]]; then
  echo "corrected KL gate smoke dry run"
  echo "experiment root: $ROOT"
  echo "gate: fp32_no_autocast; legacy bfloat16 shadow retained"
  echo "task: $TASK_ID; hard limit seconds: $HARD_LIMIT_SECONDS"
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
  exec tmux new-session -d -s "$SESSION_NAME" "$SCRIPT_PATH --smoke"
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$VALIDATION_MANIFEST" ]]; then
  echo "required isolated Alignment v2 manifests are missing" >&2
  exit 2
fi
if [[ -e "$ROOT" && -n "$(find "$ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to reuse non-empty experiment root: $ROOT" >&2
  exit 2
fi

mkdir -p "$ROOT"
.venv/bin/python - "$ROOT/matrix.json" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$TASK_ID" "$HARD_LIMIT_SECONDS" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
train = Path(sys.argv[2])
validation = Path(sys.argv[3])
train_count = sum(bool(line.strip()) for line in train.read_text(encoding="utf-8").splitlines())
validation_count = sum(bool(line.strip()) for line in validation.read_text(encoding="utf-8").splitlines())
if (train_count, validation_count) != (128, 32):
    raise SystemExit(f"unexpected manifest sizes: train={train_count} validation={validation_count}")
payload = {
    "schema_version": 1,
    "task": sys.argv[4],
    "experiment_root": str(output.parent),
    "mode": "grpo",
    "seed": 42,
    "train_manifest": str(train),
    "validation_manifest": str(validation),
    "train_count": train_count,
    "validation_count": validation_count,
    "smoke": {
        "max_steps": 2,
        "max_prompts": 8,
        "validation_max_prompts": 2,
        "num_generations": 2,
        "accumulation_steps": 2,
        "max_seq_len": 384,
        "max_gen_len": 16,
        "eval_every": 2,
        "checkpoint_every": 2,
    },
    "guard": {
        "post_step_kl_target": 0.005,
        "post_step_kl_gate_mode": "fp32_no_autocast",
        "legacy_bfloat16_shadow": True,
        "full_float32_shadow": True,
        "pre_step_fp32_shadow": True,
        "backoff_factor": 0.5,
        "max_backoffs": 3,
        "token_replay": True,
    },
    "hard_limit_seconds": int(sys.argv[5]),
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

RUN_DIR="$ROOT/grpo_corrected_kl_gate_smoke_seed42"
if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to reuse existing run directory: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/run.log"
MONITOR_PATH="$RUN_DIR/resource_monitor.jsonl"
ATTEMPT_LOG_PATH="$RUN_DIR/kl_guard_attempts.jsonl"
REPLAY_LOG_PATH="$RUN_DIR/kl_guard_token_replay.jsonl"
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
  --beta 0.02 --epsilon 0.2 --epsilon-high 5.0
  --post-step-kl-target 0.005 --post-step-kl-gate-mode fp32_no_autocast
  --kl-backoff-factor 0.5 --kl-max-backoffs 3
  --post-step-kl-diagnostic-fp32 --post-step-kl-diagnostic-full-fp32
  --pre-step-kl-diagnostic-fp32
  --kl-guard-attempt-log-path "$ATTEMPT_LOG_PATH"
  --kl-guard-token-replay-path "$REPLAY_LOG_PATH"
  --microbatch-log-path "$RUN_DIR/microbatch_summaries.jsonl" --microbatch-gradient-norm
  --interleave-categories --save-dir "$RUN_DIR" --save-weight grpo_corrected_kl_gate_smoke_seed42)

(
  echo "STARTED_AT=$(date -Is)"
  echo "MODE=--smoke TASK_ID=$TASK_ID"
  echo "COMMAND=$(printf '%q ' "${CMD[@]}")"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "ENV_HASH=$(sha256sum trainer/train_grpo_lite.py align/rl_rules.py evaluation/audit_corrected_kl_gate.py scripts/run_corrected_kl_gate_smoke.sh "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" | sha256sum | cut -d' ' -f1)"
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

AUDIT_STATUS=0
AUDIT_DIR="$ROOT/corrected_gate_audit"
if [[ -f "$RUN_DIR/selection.json" ]]; then
  PYTHONPATH=. .venv/bin/python evaluation/audit_corrected_kl_gate.py \
    --experiment-root "$ROOT" --output-dir "$AUDIT_DIR" \
    --run-name "$(basename "$RUN_DIR")" || AUDIT_STATUS=$?
else
  echo "selection.json missing; preserving run logs without audit" >&2
  AUDIT_STATUS=2
fi
echo "CORRECTED_KL_GATE_GPU_WALL_SECONDS=$(( $(date +%s) - START_EPOCH ))"
if [[ "$RUN_STATUS" -ne 0 ]]; then
  exit "$RUN_STATUS"
fi
exit "$AUDIT_STATUS"

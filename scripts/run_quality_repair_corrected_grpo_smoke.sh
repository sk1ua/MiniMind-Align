#!/usr/bin/env bash
set -uo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
cd "$REPO_ROOT"

MODE="--dry-run"
if [[ "$#" -gt 0 ]]; then
  MODE="$1"
fi
ROOT="${QUALITY_REPAIR_GRPO_ROOT:-results/experiments/quality_repair_corrected_grpo_smoke_20260803}"
TRAIN_MANIFEST="${QUALITY_REPAIR_GRPO_TRAIN_MANIFEST:-results/inputs/rl_data_isolation_train_128_20260801.jsonl}"
VALIDATION_MANIFEST="${QUALITY_REPAIR_GRPO_VALIDATION_MANIFEST:-results/inputs/rl_data_isolation_validation_32_20260801.jsonl}"
SESSION_NAME="${QUALITY_REPAIR_GRPO_TMUX_SESSION:-quality_repair_corrected_grpo_20260803}"
HARD_LIMIT_SECONDS="${QUALITY_REPAIR_GRPO_HARD_LIMIT_SECONDS:-1800}"
MIN_FREE_GIB="${QUALITY_REPAIR_GRPO_MIN_FREE_GIB:-75}"
FROM_WEIGHT="${QUALITY_REPAIR_GRPO_FROM_WEIGHT:-quality_repair_sft_seed42}"
MODEL_DIR="${QUALITY_REPAIR_GRPO_MODEL_DIR:-results/experiments/quality_signal_repair_20260803_augmented_retry1/sft_repair_seed42/out}"
SOURCE_EXPERIMENT="${QUALITY_REPAIR_GRPO_SOURCE_EXPERIMENT:-quality_signal_repair_20260803_augmented_retry1}"
TASK_ID="${QUALITY_REPAIR_GRPO_TASK_ID:-MM-E035}"
RUN_NAME="grpo_quality_repair_corrected_seed42"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--smoke" ]]; then
  echo "usage: $0 [--dry-run|--smoke]" >&2
  exit 2
fi

if [[ "$MODE" == "--dry-run" ]]; then
  echo "quality-repair corrected GRPO smoke dry run"
  echo "experiment root: $ROOT"
  echo "run name: $RUN_NAME"
  echo "candidate from-weight: $FROM_WEIGHT"
  echo "candidate model-dir: $MODEL_DIR"
  echo "gate: fp32_no_autocast; bfloat16 shadow retained"
  echo "training forward: fp32_no_autocast"
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
  if [[ -e "$ROOT" ]]; then
    echo "refusing to reuse existing experiment root: $ROOT" >&2
    exit 2
  fi
  exec tmux new-session -d -s "$SESSION_NAME" "cd $REPO_ROOT && $SCRIPT_PATH --smoke"
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$VALIDATION_MANIFEST" ]]; then
  echo "required isolated Alignment v2 manifests are missing" >&2
  exit 2
fi
if [[ ! -f "$MODEL_DIR/$FROM_WEIGHT"_768.pth ]]; then
  echo "candidate checkpoint is missing: $MODEL_DIR/$FROM_WEIGHT"_768.pth >&2
  exit 2
fi
if [[ -e "$ROOT" ]]; then
  echo "refusing to reuse existing experiment root: $ROOT" >&2
  exit 2
fi

mkdir -p "$ROOT"
.venv/bin/python - "$ROOT/matrix.json" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$TASK_ID" "$HARD_LIMIT_SECONDS" "$FROM_WEIGHT" "$MODEL_DIR" "$SOURCE_EXPERIMENT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
train = Path(sys.argv[2])
validation = Path(sys.argv[3])
task_id = sys.argv[4]
hard_limit = int(sys.argv[5])
from_weight = sys.argv[6]
model_dir = Path(sys.argv[7])
source_experiment = sys.argv[8]

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

train_count = sum(bool(line.strip()) for line in train.read_text(encoding="utf-8").splitlines())
validation_count = sum(bool(line.strip()) for line in validation.read_text(encoding="utf-8").splitlines())
if (train_count, validation_count) != (128, 32):
    raise SystemExit(f"unexpected manifest sizes: train={train_count} validation={validation_count}")

candidate = model_dir / f"{from_weight}_768.pth"
payload = {
    "schema_version": 1,
    "task": task_id,
    "experiment_root": str(output.parent),
    "run_name": "grpo_quality_repair_corrected_seed42",
    "mode": "grpo",
    "seed": 42,
    "diagnostic_only": True,
    "candidate": {
        "from_weight": from_weight,
        "model_dir": str(model_dir),
        "checkpoint": str(candidate),
        "checkpoint_sha256": sha256(candidate),
        "source_experiment": source_experiment,
    },
    "train_manifest": str(train),
    "train_manifest_sha256": sha256(train),
    "validation_manifest": str(validation),
    "validation_manifest_sha256": sha256(validation),
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
        "validation_max_new_tokens": 16,
        "learning_rate": 3e-7,
        "max_grad_norm": 1.0,
        "dtype": "bfloat16",
        "training_forward_mode": "fp32_no_autocast",
    },
    "guard": {
        "post_step_kl_target": 0.005,
        "post_step_kl_gate_mode": "fp32_no_autocast",
        "legacy_bfloat16_shadow": True,
        "full_float32_shadow": True,
        "pre_step_fp32_shadow": True,
        "pre_step_loss_fp32": True,
        "backoff_factor": 0.5,
        "max_backoffs": 3,
        "token_replay": True,
    },
    "hard_limit_seconds": hard_limit,
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

RUN_DIR="$ROOT/$RUN_NAME"
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
  --from-weight "$FROM_WEIGHT" --model-dir "$MODEL_DIR" --tokenizer-path model --batch-size 1
  --device cuda:0 --dtype bfloat16 --seed 42 --mode grpo
  --max-prompts 8 --validation-max-prompts 2 --num-generations 2 --accumulation-steps 2
  --max-seq-len 384 --max-gen-len 16 --max-steps 2 --eval-every 2 --checkpoint-every 2
  --validation-max-new-tokens 16 --learning-rate 3e-7 --max-grad-norm 1.0
  --beta 0.02 --epsilon 0.2 --epsilon-high 5.0
  --post-step-kl-target 0.005 --post-step-kl-gate-mode fp32_no_autocast
  --training-forward-mode fp32_no_autocast
  --kl-backoff-factor 0.5 --kl-max-backoffs 3
  --post-step-kl-diagnostic-fp32 --post-step-kl-diagnostic-full-fp32
  --pre-step-kl-diagnostic-fp32 --pre-step-loss-diagnostic-fp32
  --pre-step-loss-replay-path "$RUN_DIR/pre_step_loss_replay.jsonl"
  --kl-guard-attempt-log-path "$ATTEMPT_LOG_PATH"
  --kl-guard-token-replay-path "$REPLAY_LOG_PATH"
  --microbatch-log-path "$RUN_DIR/microbatch_summaries.jsonl" --microbatch-gradient-norm
  --interleave-categories --save-dir "$RUN_DIR" --save-weight quality_repair_corrected_grpo_seed42)

(
  echo "STARTED_AT=$(date -Is)"
  echo "MODE=--smoke TASK_ID=$TASK_ID"
  echo "CANDIDATE_FROM_WEIGHT=$FROM_WEIGHT"
  echo "CANDIDATE_MODEL_DIR=$MODEL_DIR"
  echo "COMMAND=$(printf '%q ' "${CMD[@]}")"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "ENV_HASH=$(sha256sum trainer/train_grpo_lite.py align/rl_rules.py evaluation/audit_corrected_kl_gate.py "$SCRIPT_PATH" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$MODEL_DIR/$FROM_WEIGHT"_768.pth | sha256sum | cut -d' ' -f1)"
  echo "GPU_BEFORE=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
  echo "DISK_BEFORE=$(df -h . | tail -1)"
  set +e
  timeout --signal=TERM --kill-after=30s "${HARD_LIMIT_SECONDS}s" "${CMD[@]}"
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
  PROCESS_SNAPSHOT="$(pgrep -af 'train_grpo_lite|run_quality_repair_corrected_grpo_smoke' 2>/dev/null | tr '\n' ';' | sed 's/"/\\"/g')"
  SNAPSHOT_TIME="$(date -Is)"
  SNAPSHOT_GPU="$GPU_SNAPSHOT" SNAPSHOT_DISK="$DISK_SNAPSHOT" SNAPSHOT_PROCESSES="$PROCESS_SNAPSHOT" SNAPSHOT_TIME="$SNAPSHOT_TIME" .venv/bin/python - "$MONITOR_PATH" <<'PY'
import json
import os
import sys
from pathlib import Path

Path(sys.argv[1]).open("a", encoding="utf-8").write(
    json.dumps({
        "timestamp": os.environ.get("SNAPSHOT_TIME"),
        "gpu": os.environ.get("SNAPSHOT_GPU", ""),
        "disk": os.environ.get("SNAPSHOT_DISK", ""),
        "processes": os.environ.get("SNAPSHOT_PROCESSES", ""),
    }, ensure_ascii=False) + "\n"
)
PY
  AVAILABLE_KIB="$(df -Pk . | awk 'NR==2 {print $4}')"
  if [[ -n "$AVAILABLE_KIB" && "$AVAILABLE_KIB" -lt $((MIN_FREE_GIB * 1024 * 1024)) ]]; then
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
.venv/bin/python evaluation/audit_corrected_kl_gate.py \
  --experiment-root "$ROOT" --output-dir "$AUDIT_DIR" \
  --run-name "$RUN_NAME" >>"$LOG_PATH" 2>&1 || AUDIT_STATUS=$?

cat >"$ROOT/wrapper_status.json" <<EOF
{"run_status":$RUN_STATUS,"audit_status":$AUDIT_STATUS,"gpu_wall_seconds":$(( $(date +%s) - START_EPOCH )),"finished_at":"$(date -Is)"}
EOF
if [[ "$RUN_STATUS" -ne 0 ]]; then
  exit "$RUN_STATUS"
fi
exit "$AUDIT_STATUS"

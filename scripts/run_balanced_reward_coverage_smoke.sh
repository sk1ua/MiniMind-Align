#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
ROOT="${BALANCED_REWARD_COVERAGE_ROOT:-results/experiments/rl_balanced_reward_coverage_smoke_20260802}"
TRAIN_SOURCE="results/inputs/rl_data_isolation_train_128_20260801.jsonl"
VALIDATION_SOURCE="results/inputs/rl_data_isolation_validation_32_20260801.jsonl"
SESSION_NAME="${BALANCED_REWARD_COVERAGE_TMUX_SESSION:-rl_balanced_coverage_20260802}"
HARD_LIMIT_SECONDS="${BALANCED_REWARD_COVERAGE_HARD_LIMIT_SECONDS:-1800}"
TASK_ID="${BALANCED_REWARD_COVERAGE_TASK_ID:-MM-E030}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--smoke" ]]; then
  echo "usage: $0 [--dry-run|--smoke]" >&2
  exit 2
fi
if [[ "$MODE" == "--dry-run" ]]; then
  echo "balanced natural rule-reward coverage smoke dry run"
  echo "experiment root: $ROOT"
  echo "task: $TASK_ID; seed: 42; categories: 8; train/validation rows: 8/8"
  echo "GRPO: 2 steps; accumulation_steps: 8; num_generations: 2; max_gen_len: 16"
  echo "active training/gate: fp32_no_autocast; reward source: rule_reward; hard limit seconds: $HARD_LIMIT_SECONDS"
  echo "one GPU task under tmux; no CISPO, formal RL, C-Eval or frozen evaluation"
  exit 0
fi

if [[ -z "${TMUX:-}" ]]; then
  command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; exit 2; }
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "refusing to reuse existing tmux session: $SESSION_NAME" >&2
    exit 2
  fi
  exec tmux new-session -d -s "$SESSION_NAME" "$SCRIPT_PATH --smoke"
fi

command -v timeout >/dev/null 2>&1 || { echo "GNU timeout is required" >&2; exit 2; }
[[ -f "$TRAIN_SOURCE" && -f "$VALIDATION_SOURCE" ]] || { echo "isolated manifests are missing" >&2; exit 2; }
if [[ -e "$ROOT" && -n "$(find "$ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to reuse non-empty experiment root: $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
INPUT_DIR="$ROOT/inputs"
.venv/bin/python evaluation/prepare_balanced_coverage_manifests.py \
  --train-source "$TRAIN_SOURCE" --validation-source "$VALIDATION_SOURCE" \
  --output-dir "$INPUT_DIR" --seed 42

TRAIN_MANIFEST="$INPUT_DIR/balanced_train_manifest.jsonl"
VALIDATION_MANIFEST="$INPUT_DIR/balanced_validation_manifest.jsonl"
RUN_DIR="$ROOT/grpo_balanced_reward_coverage_seed42"
[[ ! -e "$RUN_DIR" ]] || { echo "refusing to reuse existing run directory: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/run.log"
MONITOR_PATH="$RUN_DIR/resource_monitor.jsonl"
START_EPOCH="$(date +%s)"

.venv/bin/python - "$ROOT/matrix.json" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$HARD_LIMIT_SECONDS" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
train = Path(sys.argv[2])
validation = Path(sys.argv[3])
limit = int(sys.argv[4])
output.write_text(json.dumps({
    "schema_version": 1,
    "task": "MM-E030",
    "audit_task": "MM-F038",
    "mode": "grpo",
    "seed": 42,
    "train_manifest": str(train),
    "validation_manifest": str(validation),
    "train_count": 8,
    "validation_count": 8,
    "category_count": 8,
    "reward_source": "rule_reward",
    "controlled_reward_override": False,
    "config": {
        "max_steps": 2,
        "max_prompts": 8,
        "validation_max_prompts": 8,
        "num_generations": 2,
        "accumulation_steps": 8,
        "max_seq_len": 384,
        "max_gen_len": 16,
        "eval_every": 2,
        "checkpoint_every": 2,
        "validation_max_new_tokens": 16,
        "learning_rate": 3e-7,
        "max_grad_norm": 1.0,
        "beta": 0.02,
        "epsilon": 0.2,
        "epsilon_high": 5.0,
        "kl_threshold": 0.005,
        "kl_patience": 2,
        "post_step_kl_target": 0.005,
        "post_step_kl_gate_mode": "fp32_no_autocast",
        "training_forward_mode": "fp32_no_autocast",
        "kl_backoff_factor": 0.5,
        "kl_max_backoffs": 3,
        "dtype": "bfloat16",
        "interleave_categories": True,
    },
    "hard_limit_seconds": limit,
    "diagnostic_only": True,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

CMD=(.venv/bin/python trainer/train_grpo_lite.py
  --manifest "$TRAIN_MANIFEST" --validation-manifest "$VALIDATION_MANIFEST"
  --categories "conciseness,format,instruction,reasoning,repetition,safety,termination,uncertainty"
  --from-weight align_sft_v2_pilot --model-dir out --tokenizer-path model --batch-size 1
  --device cuda:0 --dtype bfloat16 --seed 42 --mode grpo
  --max-prompts 8 --validation-max-prompts 8 --num-generations 2 --accumulation-steps 8
  --max-seq-len 384 --max-gen-len 16 --max-steps 2 --eval-every 2 --checkpoint-every 2
  --validation-max-new-tokens 16 --learning-rate 3e-7 --max-grad-norm 1.0
  --beta 0.02 --epsilon 0.2 --epsilon-high 5.0 --kl-threshold 0.005 --kl-patience 2
  --quality-drop-points 10 --post-step-kl-target 0.005
  --post-step-kl-gate-mode fp32_no_autocast --training-forward-mode fp32_no_autocast
  --kl-backoff-factor 0.5 --kl-max-backoffs 3
  --post-step-kl-diagnostic-fp32 --post-step-kl-diagnostic-full-fp32
  --pre-step-kl-diagnostic-fp32 --pre-step-loss-diagnostic-fp32
  --pre-step-loss-replay-path "$RUN_DIR/pre_step_loss_replay.jsonl"
  --kl-guard-attempt-log-path "$RUN_DIR/kl_guard_attempts.jsonl"
  --kl-guard-token-replay-path "$RUN_DIR/kl_guard_token_replay.jsonl"
  --microbatch-log-path "$RUN_DIR/microbatch_summaries.jsonl" --microbatch-gradient-norm
  --interleave-categories --save-dir "$RUN_DIR" --save-weight grpo_balanced_reward_coverage_seed42)

(
  echo "STARTED_AT=$(date -Is)"
  echo "MODE=--smoke TASK_ID=$TASK_ID"
  echo "COMMAND=$(printf '%q ' "${CMD[@]}")"
  echo "TRAIN_SOURCE=$TRAIN_SOURCE VALIDATION_SOURCE=$VALIDATION_SOURCE"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "ENV_HASH=$(sha256sum trainer/train_grpo_lite.py align/rl_rules.py evaluation/audit_natural_reward_coverage.py evaluation/prepare_balanced_coverage_manifests.py scripts/run_balanced_reward_coverage_smoke.sh "$TRAIN_SOURCE" "$VALIDATION_SOURCE" | sha256sum | cut -d' ' -f1)"
  echo "GPU_BEFORE=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
  echo "DISK_BEFORE=$(df -h . | tail -1)"
  set +e
  timeout --signal=TERM "${HARD_LIMIT_SECONDS}s" "${CMD[@]}"
  RUN_STATUS=$?
  set -e
  echo "EXIT_CODE=$RUN_STATUS"
  echo "GPU_AFTER=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
  echo "DISK_AFTER=$(df -h . | tail -1)"
  echo "FINISHED_AT=$(date -Is)"
  echo "GPU_WALL_SECONDS=$(( $(date +%s) - START_EPOCH ))"
  exit "$RUN_STATUS"
) >"$LOG_PATH" 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
  GPU_SNAPSHOT="$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  DISK_SNAPSHOT="$(df -h . | tail -1)"
  PROCESS_SNAPSHOT="$(ps -p "$PID" -o pid=,etime=,stat= 2>/dev/null)"
  printf '{"timestamp":"%s","gpu":"%s","disk":"%s","process":"%s"}\n' "$(date -Is)" "$GPU_SNAPSHOT" "$DISK_SNAPSHOT" "$PROCESS_SNAPSHOT" >>"$MONITOR_PATH"
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
if [[ -f "$RUN_DIR/samples.jsonl" ]]; then
  PYTHONPATH=. .venv/bin/python evaluation/audit_natural_reward_coverage.py \
    --manifest "$TRAIN_MANIFEST" --sample-root "$RUN_DIR" \
    --output-dir "$ROOT/coverage_audit" --max-gen-len 16 || AUDIT_STATUS=$?
else
  echo "samples.jsonl missing; preserving run logs without coverage audit" >&2
  AUDIT_STATUS=2
fi
echo "TOTAL_GPU_WALL_SECONDS=$(( $(date +%s) - START_EPOCH ))"
if [[ "$RUN_STATUS" -ne 0 ]]; then
  exit "$RUN_STATUS"
fi
exit "$AUDIT_STATUS"

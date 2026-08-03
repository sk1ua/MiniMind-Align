#!/usr/bin/env bash
set -uo pipefail

MODE="${1:---dry-run}"
ROOT="${RL_QUALITY_EVIDENCE_ROOT:-results/experiments/rl_quality_evidence_corrected_grpo_20260803}"
TRAIN_MANIFEST="results/inputs/rl_data_isolation_train_128_20260801.jsonl"
VALIDATION_MANIFEST="results/inputs/rl_data_isolation_validation_32_20260801.jsonl"
MODEL_DIR="${RL_QUALITY_EVIDENCE_MODEL_DIR:-results/experiments/quality_signal_repair_20260803_augmented_retry1/sft_repair_seed42/out}"
FROM_WEIGHT="${RL_QUALITY_EVIDENCE_FROM_WEIGHT:-quality_repair_sft_seed42}"
SESSION="${RL_QUALITY_EVIDENCE_TMUX_SESSION:-rl_quality_evidence_corrected_grpo_20260803}"
HARD_LIMIT="${RL_QUALITY_EVIDENCE_HARD_LIMIT_SECONDS:-3600}"
MIN_FREE_GIB="${RL_QUALITY_EVIDENCE_MIN_FREE_GIB:-75}"
TASK_LABEL="${RL_QUALITY_EVIDENCE_TASK_LABEL:-MM-E040/MM-F049}"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--diagnostic" ]]; then
  echo "usage: $0 [--dry-run|--diagnostic]" >&2
  exit 2
fi
if [[ "$MODE" == "--dry-run" ]]; then
  echo "corrected-GRPO quality-evidence diagnostic dry run"
  echo "experiment root: $ROOT"
  echo "train=128 validation=32 balanced_categories=8 steps=4 generations=8 accumulation=8"
  echo "precision_contract=no_autocast_v1 active_loss_gate=fp32_no_autocast"
  echo "candidate: $MODEL_DIR/${FROM_WEIGHT}_768.pth"
  echo "hard_limit_seconds: $HARD_LIMIT"
  exit 0
fi

if [[ -z "${TMUX:-}" ]]; then
  command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; exit 2; }
  tmux has-session -t "$SESSION" 2>/dev/null && { echo "refusing to reuse tmux session: $SESSION" >&2; exit 2; }
  exec tmux new-session -d -s "$SESSION" "$SCRIPT_PATH --diagnostic"
fi
command -v timeout >/dev/null 2>&1 || { echo "GNU timeout is required" >&2; exit 2; }
[[ -f "$TRAIN_MANIFEST" && -f "$VALIDATION_MANIFEST" ]] || { echo "isolated manifests missing" >&2; exit 2; }
[[ -f "$MODEL_DIR/${FROM_WEIGHT}_768.pth" ]] || { echo "quality-repair candidate weight missing" >&2; exit 2; }
[[ ! -e "$ROOT" ]] || { echo "refusing to reuse existing experiment root: $ROOT" >&2; exit 2; }

mkdir -p "$ROOT"
.venv/bin/python - "$ROOT/matrix.json" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$MODEL_DIR" "$FROM_WEIGHT" "$HARD_LIMIT" "$TASK_LABEL" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
train = Path(sys.argv[2])
validation = Path(sys.argv[3])
payload = {
    "schema_version": 1,
    "task": sys.argv[7],
    "related_task": sys.argv[7],
    "experiment_root": str(out.parent),
    "mode": "grpo",
    "seed": 42,
    "diagnostic_scope": "single-seed corrected-GRPO quality evidence with full balanced validation",
    "train_manifest": str(train),
    "validation_manifest": str(validation),
    "train_count": sum(bool(x.strip()) for x in train.read_text(encoding="utf-8").splitlines()),
    "validation_count": sum(bool(x.strip()) for x in validation.read_text(encoding="utf-8").splitlines()),
    "model_dir": sys.argv[4],
    "from_weight": sys.argv[5],
    "max_steps": 4,
    "max_prompts": 128,
    "validation_max_prompts": 32,
    "validation_max_new_tokens": 128,
    "num_generations": 8,
    "accumulation_steps": 8,
    "max_seq_len": 384,
    "max_gen_len": 128,
    "eval_every": 2,
    "checkpoint_every": 2,
    "learning_rate": 3e-7,
    "max_grad_norm": 1.0,
    "beta": 0.02,
    "epsilon": 0.2,
    "epsilon_high": 5.0,
    "post_step_kl_target": 0.005,
    "precision_contract_mode": "no_autocast_v1",
    "training_forward_mode": "fp32_no_autocast",
    "post_step_kl_gate_mode": "fp32_no_autocast",
    "backoff_factor": 0.5,
    "max_backoffs": 3,
    "hard_limit_seconds": int(sys.argv[6]),
    "balanced_categories": ["conciseness", "format", "instruction", "reasoning", "repetition", "safety", "termination", "uncertainty"],
    "forbidden": ["formal_rl", "cispo", "multi_seed", "ceval", "frozen_eval", "model_replacement"],
}
if (payload["train_count"], payload["validation_count"]) != (128, 32):
    raise SystemExit("unexpected isolated manifest size")
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

RUN_DIR="$ROOT/grpo_quality_evidence_seed42"
[[ ! -e "$RUN_DIR" ]] || { echo "refusing to reuse run directory: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/run.log"
MONITOR="$RUN_DIR/resource_monitor.jsonl"
ATTEMPTS="$RUN_DIR/kl_guard_attempts.jsonl"
REPLAY="$RUN_DIR/kl_guard_token_replay.jsonl"
PRESTEP="$RUN_DIR/pre_step_loss_replay.jsonl"
MICRO="$RUN_DIR/microbatch_summaries.jsonl"
START="$(date +%s)"
CMD_TEXT=".venv/bin/python trainer/train_grpo_lite.py --manifest $TRAIN_MANIFEST --validation-manifest $VALIDATION_MANIFEST --categories conciseness,format,instruction,reasoning,repetition,safety,termination,uncertainty --from-weight $FROM_WEIGHT --model-dir $MODEL_DIR --tokenizer-path model --batch-size 1 --device cuda:0 --dtype bfloat16 --seed 42 --mode grpo --max-prompts 128 --validation-max-prompts 32 --num-generations 8 --accumulation-steps 8 --max-seq-len 384 --max-gen-len 128 --max-steps 4 --eval-every 2 --checkpoint-every 2 --validation-max-new-tokens 128 --learning-rate 3e-7 --max-grad-norm 1.0 --beta 0.02 --epsilon 0.2 --epsilon-high 5.0 --precision-contract-mode no_autocast_v1 --training-forward-mode fp32_no_autocast --post-step-kl-target 0.005 --post-step-kl-gate-mode fp32_no_autocast --kl-backoff-factor 0.5 --kl-max-backoffs 3 --post-step-kl-diagnostic-fp32 --post-step-kl-diagnostic-full-fp32 --pre-step-kl-diagnostic-fp32 --pre-step-loss-diagnostic-fp32 --pre-step-loss-replay-path $PRESTEP --kl-guard-attempt-log-path $ATTEMPTS --kl-guard-token-replay-path $REPLAY --microbatch-log-path $MICRO --microbatch-gradient-norm --interleave-categories --quality-drop-points 10 --save-dir $RUN_DIR --save-weight rl_quality_evidence_seed42"

(
  echo "STARTED_AT=$(date -Is)"
  echo "COMMAND=$CMD_TEXT"
  echo "TASK=$TASK_LABEL"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "ENV_HASH=$(sha256sum trainer/train_grpo_lite.py align/rl_rules.py evaluation/audit_precision_contract.py evaluation/audit_rl_quality_evidence.py scripts/run_rl_quality_evidence_diagnostic.sh "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$MODEL_DIR/${FROM_WEIGHT}_768.pth" | sha256sum | cut -d' ' -f1)"
  echo "GPU_BEFORE=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
  echo "DISK_BEFORE=$(df -h . | tail -1)"
  set +e
  timeout --signal=TERM "${HARD_LIMIT}s" bash -c "$CMD_TEXT"
  STATUS=$?
  set -e
  echo "EXIT_CODE=$STATUS"
  echo "GPU_AFTER=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
  echo "DISK_AFTER=$(df -h . | tail -1)"
  echo "FINISHED_AT=$(date -Is)"
  echo "GPU_WALL_SECONDS=$(( $(date +%s) - START ))"
  exit "$STATUS"
) >"$LOG" 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
  GPU="$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\\n' ';')"
  DISK="$(df -h . | tail -1)"
  printf '{"timestamp":"%s","gpu":"%s","disk":"%s","pid":%s}\n' "$(date -Is)" "$GPU" "$DISK" "$PID" >>"$MONITOR"
  AVAILABLE_KIB="$(df -Pk . | awk 'NR==2 {print $4}')"
  if [[ -n "$AVAILABLE_KIB" && "$AVAILABLE_KIB" -lt $((MIN_FREE_GIB * 1024 * 1024)) ]]; then
    echo "disk safety threshold reached; terminating diagnostic" >>"$LOG"
    kill -TERM "$PID" 2>/dev/null || true
    break
  fi
  if grep -Eiq 'cuda out of memory|out of memory|nan|inf' "$LOG"; then
    echo "hard numerical/resource error detected; terminating diagnostic" >>"$LOG"
    kill -TERM "$PID" 2>/dev/null || true
    break
  fi
  sleep 60
done
wait "$PID"
RUN_STATUS=$?

PRECISION_DIR="$ROOT/precision_contract_audit"
PRECISION_LOG="$RUN_DIR/precision_audit.log"
.venv/bin/python evaluation/audit_precision_contract.py --experiment-root "$ROOT" --output-dir "$PRECISION_DIR" --run-name "$(basename "$RUN_DIR")" --expected-steps 4 >"$PRECISION_LOG" 2>&1
PRECISION_STATUS=$?
QUALITY_DIR="$ROOT/quality_evidence_audit"
QUALITY_LOG="$RUN_DIR/quality_audit.log"
.venv/bin/python evaluation/audit_rl_quality_evidence.py --experiment-root "$ROOT" --output-dir "$QUALITY_DIR" --run-name "$(basename "$RUN_DIR")" --precision-summary "$PRECISION_DIR/summary.json" >"$QUALITY_LOG" 2>&1
QUALITY_STATUS=$?
{
  echo "PRECISION_AUDIT_EXIT_CODE=$PRECISION_STATUS"
  echo "QUALITY_AUDIT_EXIT_CODE=$QUALITY_STATUS"
  echo "FINAL_GPU_WALL_SECONDS=$(( $(date +%s) - START ))"
} >>"$LOG"
if [[ "$RUN_STATUS" -ne 0 ]]; then exit "$RUN_STATUS"; fi
if [[ "$PRECISION_STATUS" -ne 0 ]]; then exit "$PRECISION_STATUS"; fi
exit "$QUALITY_STATUS"

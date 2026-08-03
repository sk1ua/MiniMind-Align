#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="results/experiments/error_driven_preference_repair_20260803_retry6"
DATA_DIR="$ROOT/inputs"
SFT_DIR="$ROOT/sft_seed42"
DPO_DIR="$ROOT/dpo_seed42"
SIMPO_DIR="$ROOT/simpo_seed42"
BASELINE_DIR="$ROOT/baseline_eval"
SFT_EVAL_DIR="$SFT_DIR/eval"
DPO_EVAL_DIR="$DPO_DIR/eval"
SIMPO_EVAL_DIR="$SIMPO_DIR/eval"
AUDIT_DIR="$ROOT/method_comparison"
SESSION="error_driven_preference_repair_20260803_retry6"
HARD_LIMIT_SECONDS=7200
MIN_FREE_GIB=75
PY=".venv/bin/python"
MODE="--dry-run"
if [ "$#" -ge 1 ]; then MODE="$1"; fi

TRAIN_MANIFEST="results/inputs/quality_signal_repair_native_v2_train_manifest_1016_20260803.jsonl"
VALIDATION_MANIFEST="dataset/alignment_v2/manifests/validation_manifest.jsonl"
RELEASE_MANIFEST="results/inputs/rl_data_isolation_validation_32_20260801.jsonl"
FAILURE_SAMPLES="results/experiments/rl_quality_evidence_corrected_grpo_20260803/grpo_quality_evidence_seed42/samples.jsonl"
BASELINE_WEIGHT="out/align_sft_v2_pilot_768.pth"
SFT_WEIGHT="$SFT_DIR/out/error_driven_sft_seed42_768.pth"
DPO_WEIGHT="$DPO_DIR/out/error_driven_dpo_seed42_768.pth"
SIMPO_WEIGHT="$SIMPO_DIR/out/error_driven_simpo_seed42_768.pth"

if [ "$MODE" = "--dry-run" ]; then
  echo "error-driven SFT -> DPO/SimPO repair dry run"
  echo "root=$ROOT"
  echo "audit: $PY evaluation/prepare_error_driven_preference_repair.py"
  echo "smoke: SFT 2 steps/epoch, DPO 2 steps, SimPO 2 steps, fixed 160+32 evaluation"
  echo "full: SFT 38 steps/epoch, DPO 32 steps, SimPO 32 steps, fixed 160+32 evaluation"
  echo "hard_limit_seconds=$HARD_LIMIT_SECONDS"
  echo "tmux_session=$SESSION"
  exit 0
fi

if [ "$MODE" = "--audit" ]; then
  if [ -e "$ROOT" ] && find "$ROOT" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to overwrite non-empty root: $ROOT" >&2
    exit 2
  fi
  mkdir -p "$ROOT"
  GIT_COMMIT="$(git rev-parse HEAD)"
  ENV_HASH="$(sha256sum evaluation/prepare_error_driven_preference_repair.py dataset/alignment_v2/manifests/train_manifest.jsonl dataset/alignment_v2/manifests/validation_manifest.jsonl results/inputs/rl_data_isolation_validation_32_20260801.jsonl "$FAILURE_SAMPLES" | sha256sum | cut -d' ' -f1)"
  {
    echo "STARTED_AT=$(date -Is)"
    echo "MODE=--audit"
    echo "SESSION=$SESSION"
    echo "GIT_COMMIT=$GIT_COMMIT"
    echo "ENV_HASH=$ENV_HASH"
    echo "GPU_WALL_SECONDS=0"
  } > "$ROOT/run_metadata.log"
  "$PY" evaluation/prepare_error_driven_preference_repair.py --train-manifest "$TRAIN_MANIFEST" --validation-manifest "$VALIDATION_MANIFEST" --release-manifest "$RELEASE_MANIFEST" --failure-samples "$FAILURE_SAMPLES" --expected-native-train-count 1016 --output-dir "$DATA_DIR" | tee "$ROOT/data_prepare.log"
  exit 0
fi

if [ "$MODE" != "--smoke" ] && [ "$MODE" != "--full" ]; then
  echo "usage: $0 [--dry-run|--audit|--smoke|--full]" >&2
  exit 2
fi

if [ ! -f "$DATA_DIR/data_manifest.json" ]; then
  echo "missing data audit; run --audit first" >&2
  exit 2
fi
for required in "$DATA_DIR/error_driven_sft.jsonl" "$DATA_DIR/preference_train.jsonl" "$BASELINE_WEIGHT"; do
  if [ ! -f "$required" ]; then
    echo "missing required input: $required" >&2
    exit 2
  fi
done
for directory in "$BASELINE_DIR" "$SFT_DIR" "$DPO_DIR" "$SIMPO_DIR" "$AUDIT_DIR"; do
  if [ -e "$directory" ]; then
    echo "refusing to reuse existing output directory: $directory" >&2
    exit 2
  fi
done

mkdir -p "$ROOT"
START_SECONDS="$(date +%s)"
MONITOR_PID=""
touch "$ROOT/.running" "$ROOT/resource_monitor.log"
cleanup() {
  rm -f "$ROOT/.running"
  if [ -n "$MONITOR_PID" ]; then kill "$MONITOR_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

monitor_resources() {
  while [ -f "$ROOT/.running" ]; do
    GPU="$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1 || true)"
    DISK="$(df -BG / | tail -1 | tr -s ' ')"
    PROCS="$(pgrep -af 'train_full_sft|train_dpo|train_simpo|audit_quality_signal_repair' 2>/dev/null | head -5 | paste -sd ';' - || true)"
    echo "timestamp=$(date -Is) gpu=$GPU disk=$DISK processes=$PROCS" >> "$ROOT/resource_monitor.log"
    AVAILABLE="$(df -BG / | tail -1 | awk '{gsub(/G/,"",$4); print $4}')"
    if [ -n "$AVAILABLE" ] && [ "$AVAILABLE" -lt "$MIN_FREE_GIB" ]; then
      echo "disk safety floor reached: $AVAILABLE GiB (minimum $MIN_FREE_GIB GiB)" >&2
      return 90
    fi
    sleep 60
  done
}
monitor_resources &
MONITOR_PID=$!

run_step() {
  PHASE="$1"
  shift
  NOW="$(date +%s)"
  ELAPSED=$((NOW - START_SECONDS))
  REMAINING=$((HARD_LIMIT_SECONDS - ELAPSED))
  if [ "$REMAINING" -le 0 ]; then
    echo "hard limit reached before $PHASE" >&2
    return 124
  fi
  LOG="$ROOT/$PHASE.log"
  echo "COMMAND=$*" | tee "$LOG"
  set +e
  timeout --signal=TERM --kill-after=30s "$REMAINING" "$@" >> "$LOG" 2>&1
  STATUS=$?
  set -e
  echo "EXIT_CODE=$STATUS" | tee -a "$LOG"
  echo "PHASE=$PHASE EXIT_CODE=$STATUS WALL_SECONDS=$(( $(date +%s) - NOW ))" >> "$ROOT/run_metadata.log"
  return "$STATUS"
}

SFT_STEPS=38
PREF_STEPS=32
if [ "$MODE" = "--smoke" ]; then
  SFT_STEPS=2
  PREF_STEPS=2
fi

{
  echo "STARTED_AT=$(date -Is)"
  echo "MODE=$MODE"
  echo "SESSION=$SESSION"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "HARD_LIMIT_SECONDS=$HARD_LIMIT_SECONDS"
  echo "SFT_STEPS_PER_EPOCH=$SFT_STEPS"
  echo "PREFERENCE_STEPS=$PREF_STEPS"
} > "$ROOT/run_metadata.log"

mkdir -p "$BASELINE_DIR"
run_step baseline_evaluation "$PY" evaluation/audit_quality_signal_repair.py --phase evaluate --manifest "$VALIDATION_MANIFEST" --release-manifest "$RELEASE_MANIFEST" --weight-path "$BASELINE_WEIGHT" --model-dir out --tokenizer-path model --device cuda:0 --model-label align_sft_v2_pilot --output-dir "$BASELINE_DIR"

mkdir -p "$SFT_DIR/out"
run_step sft_training "$PY" trainer/train_full_sft.py --from_weight align_sft_v2_pilot --model_dir out --tokenizer_path model --epochs 2 --max_steps "$SFT_STEPS" --batch_size 16 --learning_rate 5e-6 --data_path "$DATA_DIR/error_driven_sft.jsonl" --save_dir "$SFT_DIR/out" --save_weight error_driven_sft_seed42 --max_seq_len 512 --dtype bfloat16 --grad_clip 1.0 --device cuda:0 --use_compile 0 --num_workers 0
run_step sft_reload "$PY" evaluation/verify_checkpoint.py --weight-path "$SFT_WEIGHT" --device cpu --max-new-tokens 2
mkdir -p "$SFT_EVAL_DIR"
run_step sft_evaluation "$PY" evaluation/audit_quality_signal_repair.py --phase evaluate --manifest "$VALIDATION_MANIFEST" --release-manifest "$RELEASE_MANIFEST" --weight-path "$SFT_WEIGHT" --model-dir "$SFT_DIR/out" --tokenizer-path model --device cuda:0 --model-label error_driven_sft_seed42 --output-dir "$SFT_EVAL_DIR"

mkdir -p "$DPO_DIR/out" "$DPO_DIR/checkpoints"
run_step dpo_training "$PY" trainer/train_dpo.py --from_weight error_driven_sft_seed42 --model_dir "$SFT_DIR/out" --tokenizer_path model --epochs 1 --max_steps "$PREF_STEPS" --batch_size 4 --learning_rate 4e-8 --data_path "$DATA_DIR/preference_train.jsonl" --save_dir "$DPO_DIR/out" --save_weight error_driven_dpo_seed42 --checkpoint_dir "$DPO_DIR/checkpoints" --max_seq_len 512 --dtype bfloat16 --grad_clip 1.0 --device cuda:0 --use_compile 0 --num_workers 0 --save_interval "$PREF_STEPS"
run_step dpo_reload "$PY" evaluation/verify_checkpoint.py --weight-path "$DPO_WEIGHT" --device cpu --max-new-tokens 2
mkdir -p "$DPO_EVAL_DIR"
run_step dpo_evaluation "$PY" evaluation/audit_quality_signal_repair.py --phase evaluate --manifest "$VALIDATION_MANIFEST" --release-manifest "$RELEASE_MANIFEST" --weight-path "$DPO_WEIGHT" --model-dir "$DPO_DIR/out" --tokenizer-path model --device cuda:0 --model-label error_driven_dpo_seed42 --output-dir "$DPO_EVAL_DIR"

mkdir -p "$SIMPO_DIR/out"
run_step simpo_training "$PY" trainer/train_simpo.py --from-weight error_driven_sft_seed42 --model-dir "$SFT_DIR/out" --tokenizer-path model --save-dir "$SIMPO_DIR/out" --save-weight error_driven_simpo_seed42 --data-path "$DATA_DIR/preference_train.jsonl" --batch-size 4 --max-steps "$PREF_STEPS" --max-seq-len 512 --learning-rate 1e-6 --beta 2.0 --gamma 0.5 --grad-clip 1.0 --dtype float32 --device cuda:0 --num-workers 0 --save-interval "$PREF_STEPS"
run_step simpo_reload "$PY" evaluation/verify_checkpoint.py --weight-path "$SIMPO_WEIGHT" --device cpu --max-new-tokens 2
mkdir -p "$SIMPO_EVAL_DIR"
run_step simpo_evaluation "$PY" evaluation/audit_quality_signal_repair.py --phase evaluate --manifest "$VALIDATION_MANIFEST" --release-manifest "$RELEASE_MANIFEST" --weight-path "$SIMPO_WEIGHT" --model-dir "$SIMPO_DIR/out" --tokenizer-path model --device cuda:0 --model-label error_driven_simpo_seed42 --output-dir "$SIMPO_EVAL_DIR"

run_step method_comparison "$PY" evaluation/audit_error_driven_preference_repair.py --data-manifest "$DATA_DIR/data_manifest.json" --baseline-summary "$BASELINE_DIR/summary.json" --sft-summary "$SFT_EVAL_DIR/summary.json" --dpo-summary "$DPO_EVAL_DIR/summary.json" --simpo-summary "$SIMPO_EVAL_DIR/summary.json" --output-dir "$AUDIT_DIR"

echo "completed root=$ROOT"
echo "gpu_wall_seconds=$(( $(date +%s) - START_SECONDS ))"

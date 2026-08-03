#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="rl_validation_coverage_20260801"
ROOT="results/experiments/rl_validation_coverage_20260801"
TRAIN_MANIFEST="results/inputs/rl_data_isolation_train_128_20260801.jsonl"
VALIDATION_MANIFEST="dataset/alignment_v2/manifests/validation_manifest.jsonl"
CATEGORIES="conciseness,format,instruction,reasoning,repetition,safety,termination,uncertainty"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_rl_validation_coverage.sh --dry-run|--full [--in-tmux]

Evaluates the baseline and the six selected RL checkpoints on the full
Alignment v2 validation manifest (160 rows), with max_steps=0. This is a
diagnostic-only experiment and never changes default or existing weights.
EOF
}

MODE=""
IN_TMUX=0
while (($#)); do
  case "$1" in
    --dry-run|--full) MODE="${1#--}"; shift ;;
    --in-tmux) IN_TMUX=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MODE" ]]; then
  usage >&2
  exit 2
fi

if [[ "$MODE" == "dry-run" ]]; then
  echo "RL validation coverage dry run: baseline + six selected checkpoints"
  echo "validation manifest: $VALIDATION_MANIFEST"
  echo "validation rows: 160; max_steps: 0; output root: $ROOT"
  exit 0
fi

if [[ "$IN_TMUX" -eq 0 && -z "${TMUX:-}" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 3
  fi
  exec tmux new-session -d -s "$SESSION_NAME" "$0" --full --in-tmux
fi

if [[ -e "$ROOT" ]]; then
  echo "refusing to reuse existing experiment directory: $ROOT" >&2
  exit 3
fi

mkdir -p "$ROOT/wrappers" "$ROOT/models"

COMMON=(
  --manifest "$TRAIN_MANIFEST"
  --validation-manifest "$VALIDATION_MANIFEST"
  --categories "$CATEGORIES"
  --max-prompts 1
  --validation-max-prompts 160
  --accumulation-steps 1
  --num-generations 1
  --max-seq-len 384
  --max-gen-len 128
  --max-steps 0
  --eval-every 4
  --checkpoint-every 4
  --validation-max-new-tokens 128
  --device cuda:0
  --dtype bfloat16
)

run_one() {
  local label="$1" mode="$2" seed="$3" model_dir="$4" from_weight="$5"
  local experiment_id="rl_validation_coverage_20260801_${label}"
  local save_dir="$ROOT/models/$label"
  bash scripts/run_experiment.sh \
    --experiment-id "$experiment_id" \
    --task-id MM-F017 \
    --output-root "$ROOT/wrappers" \
    -- \
    .venv/bin/python trainer/train_grpo_lite.py \
      "${COMMON[@]}" \
      --model-dir "$model_dir" \
      --from-weight "$from_weight" \
      --save-dir "$save_dir" \
      --save-weight "$label" \
      --mode "$mode" \
      --seed "$seed"
}

run_one "align_sft_v2_pilot" "grpo" 42 "out" "align_sft_v2_pilot"
run_one "grpo_seed42" "grpo" 42 "$ROOT/../rl_data_isolation_20260801/grpo_seed42/checkpoints" "grpo_seed42_step_0004"
run_one "grpo_seed43" "grpo" 43 "$ROOT/../rl_data_isolation_20260801/grpo_seed43/checkpoints" "grpo_seed43_step_0004"
run_one "grpo_seed44" "grpo" 44 "$ROOT/../rl_data_isolation_20260801/grpo_seed44/checkpoints" "grpo_seed44_step_0008"
run_one "cispo_seed42" "cispo" 42 "$ROOT/../rl_data_isolation_20260801/cispo_seed42/checkpoints" "cispo_seed42_step_0004"
run_one "cispo_seed43" "cispo" 43 "$ROOT/../rl_data_isolation_20260801/cispo_seed43/checkpoints" "cispo_seed43_step_0004"
run_one "cispo_seed44" "cispo" 44 "$ROOT/../rl_data_isolation_20260801/cispo_seed44/checkpoints" "cispo_seed44_step_0008"

echo "RL validation coverage complete: $ROOT"

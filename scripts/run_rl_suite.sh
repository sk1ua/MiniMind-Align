#!/usr/bin/env bash
set -uo pipefail

MODE="${1:---dry-run}"
ROOT="results/experiments/rl_method_upgrade_20260801"
TRAIN_MANIFEST="results/inputs/on_policy_train_manifest_128_20260731.jsonl"
VALIDATION_MANIFEST="results/inputs/on_policy_validation_manifest_32_20260731.jsonl"
CATEGORIES="format,instruction,reasoning,termination,safety,repetition,conciseness,uncertainty"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--smoke" && "$MODE" != "--full" ]]; then
  echo "usage: $0 [--dry-run|--smoke|--full]" >&2
  exit 2
fi

run_one() {
  local mode="$1" seed="$2" output_dir="$3" save_weight="$4"
  shift 4
  local log_path="$output_dir/run.log"
  if [[ -e "$output_dir" ]]; then
    echo "refusing to reuse existing experiment directory: $output_dir" >&2
    return 2
  fi
  mkdir -p "$output_dir"
  local cmd=(.venv/bin/python trainer/train_grpo_lite.py
    --manifest "$TRAIN_MANIFEST"
    --validation-manifest "$VALIDATION_MANIFEST"
    --categories "$CATEGORIES"
    --from-weight align_sft_v2_pilot
    --model-dir out --tokenizer-path model --batch-size 1
    --device cuda:0 --dtype bfloat16 --seed "$seed" --mode "$mode"
    --save-dir "$output_dir" --save-weight "$save_weight"
    "$@")
  (
    echo "STARTED_AT=$(date -Is)"
    echo "MODE=$mode SEED=$seed"
    echo "COMMAND=$(printf '%q ' "${cmd[@]}")"
    echo "GIT_COMMIT=$(git rev-parse HEAD)"
    echo "ENV_HASH=$(sha256sum trainer/train_grpo_lite.py evaluation/rl_validation.py evaluation/rl_selection.py "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" | sha256sum | cut -d' ' -f1)"
    echo "GPU_BEFORE=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
    echo "DISK_BEFORE=$(df -h . | tail -1)"
    set +e
    "${cmd[@]}"
    status=$?
    set -e
    echo "EXIT_CODE=$status"
    echo "GPU_AFTER=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
    echo "DISK_AFTER=$(df -h . | tail -1)"
    echo "FINISHED_AT=$(date -Is)"
    exit "$status"
  ) >"$log_path" 2>&1
  local status=$?
  echo "RL_RUN mode=$mode seed=$seed exit_code=$status log=$log_path"
  return "$status"
}

if [[ "$MODE" == "--dry-run" ]]; then
  echo "RL suite dry run: GRPO/CISPO seeds 42,43,44"
  echo "full output root: $ROOT"
  exit 0
fi

mkdir -p "$ROOT"
overall=0
if [[ "$MODE" == "--smoke" ]]; then
  run_one grpo 42 "$ROOT/smoke_grpo_seed42" smoke_grpo_seed42 \
    --max-prompts 8 --validation-max-prompts 2 --accumulation-steps 1 --num-generations 2 \
    --max-seq-len 384 --max-gen-len 16 --max-steps 2 --eval-every 2 --checkpoint-every 2 --validation-max-new-tokens 16 || overall=$?
else
  for mode in grpo cispo; do
    for seed in 42 43 44; do
      run_one "$mode" "$seed" "$ROOT/${mode}_seed${seed}" "${mode}_seed${seed}" \
        --max-prompts 128 --validation-max-prompts 32 --accumulation-steps 8 --num-generations 8 \
        --max-seq-len 384 --max-gen-len 128 --max-steps 16 --eval-every 4 --checkpoint-every 4 \
        --validation-max-new-tokens 128 || overall=$?
    done
  done
fi
exit "$overall"

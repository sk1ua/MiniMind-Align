#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/reproduce_common.sh" "${1:---dry-run}"
STEPS=8
[[ "$MODE" == "--full" ]] && STEPS=64
run_cmd .venv/bin/python trainer/train_reward.py --train-data dataset/alignment_v2/generated/dpo_v2_train_pilot.jsonl --validation-data dataset/alignment_v2/generated/dpo_v2_validation_pilot.jsonl --model-dir out --tokenizer-path model --from-weight align_sft_v2_pilot --save-dir results/experiments/reproduce_reward_model/out --save-weight reproduce_reward_model --batch-size 4 --max-steps "$STEPS" --max-seq-len 512 --learning-rate 1e-6 --dtype bfloat16 --device cuda:0 --seed 42

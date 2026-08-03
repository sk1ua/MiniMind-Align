#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/reproduce_common.sh" "${1:---dry-run}"
STEPS=8
[[ "$MODE" == "--full" ]] && STEPS=256
run_cmd .venv/bin/python trainer/train_simpo.py --save-dir results/experiments/reproduce_simpo/out --save-weight reproduce_simpo --data-path dataset/alignment_v2/generated/dpo_v2_train_pilot.jsonl --from-weight align_sft_v2_pilot --model-dir out --tokenizer-path model --batch-size 2 --max-steps "$STEPS" --max-seq-len 512 --learning-rate 1e-6 --beta 2.0 --gamma 0.5 --dtype bfloat16 --device cuda:0 --seed 42

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/reproduce_common.sh" "${1:---dry-run}"
STEPS=12
[[ "$MODE" == "--full" ]] && STEPS=100
run_cmd .venv/bin/python trainer/train_full_sft.py --from_weight full_sft --max_steps "$STEPS" --batch_size 16 --learning_rate 1e-5 --data_path dataset/alignment_v2/generated/sft_train_pilot.jsonl --save_dir results/experiments/reproduce_align_sft_v2/out --save_weight reproduce_align_sft_v2

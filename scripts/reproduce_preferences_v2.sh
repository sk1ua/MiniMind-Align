#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/reproduce_common.sh" "${1:---dry-run}"
run_cmd .venv/bin/python evaluation/generate_on_policy_preferences.py --manifest results/inputs/on_policy_train_manifest_128_20260731.jsonl --weight align_sft_v2_pilot --num-candidates 4 --seed 42 --output-dir results/experiments/reproduce_on_policy_v2
run_cmd .venv/bin/python scripts/build_dpo_v2.py --train-input results/experiments/reproduce_on_policy_v2/preference_pairs.jsonl --validation-input results/experiments/on_policy_c001_validation_20260731/preference_pairs.jsonl --train-output dataset/alignment_v2/generated/reproduce_dpo_v2_train.jsonl --validation-output dataset/alignment_v2/generated/reproduce_dpo_v2_validation.jsonl

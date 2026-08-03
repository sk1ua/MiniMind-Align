#!/usr/bin/env bash
set -euo pipefail

pair="${1:-}"
suffix="${2:-}"
seed="${GEMINI_SEED:-42}"
if [[ "$pair" == "dpo" ]]; then
    experiment_id="gemini_align_vs_dpo_v2_full_20260731"
    baseline="results/experiments/unified_sft_v2_20260731/validator/align_sft_v2/validator_details.jsonl"
    candidate="results/experiments/unified_sft_v2_20260731/validator/dpo_v2_full/validator_details.jsonl"
    baseline_name="align_sft_v2"
    candidate_name="dpo_v2_full"
elif [[ "$pair" == "simpo" ]]; then
    experiment_id="gemini_align_vs_simpo_v1_pilot_20260731"
    baseline="results/experiments/unified_sft_v2_20260731/validator/align_sft_v2/validator_details.jsonl"
    candidate="results/experiments/unified_sft_v2_20260731/validator/simpo_v1/validator_details.jsonl"
    baseline_name="align_sft_v2"
    candidate_name="simpo_v1_pilot"
elif [[ "$pair" == "simpo_full" ]]; then
    experiment_id="gemini_align_vs_simpo_v1_full_20260731"
    baseline="results/experiments/unified_sft_v2_20260731/validator/align_sft_v2/validator_details.jsonl"
    candidate="results/experiments/unified_sft_v2_20260731/validator/simpo_v1_full/validator_details.jsonl"
    baseline_name="align_sft_v2"
    candidate_name="simpo_v1_full"
else
    printf '%s\n' 'Usage: bash scripts/run_gemini_pair_extension.sh dpo|simpo|simpo_full [suffix]' >&2
    exit 2
fi

if [[ -n "$suffix" ]]; then
    experiment_id="${experiment_id%_20260731}_${suffix}_20260731"
fi

output_dir="results/experiments/$experiment_id"
bash scripts/run_experiment.sh \
    --experiment-id "$experiment_id" \
    --task-id MM-G001 \
    -- \
    .venv-teacher/bin/python evaluation/judge_generation_gemini.py \
    --baseline "$baseline" \
    --candidate "$candidate" \
    --baseline_name "$baseline_name" \
    --candidate_name "$candidate_name" \
    --output "$output_dir/judge.jsonl" \
    --summary "$output_dir/summary.json" \
    --review "$output_dir/review.md" \
    --project gen-lang-client-0131552860 \
    --location global \
    --model gemini-3.6-flash \
    --seed "$seed" \
    --sleep 0.4 \
    --max_retries 5

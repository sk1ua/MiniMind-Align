#!/usr/bin/env bash
set -euo pipefail

experiment_root=${REFERENCE_KL_SEMANTICS_ROOT:-results/experiments/rl_kl_measurement_precision_20260802}
run_name=${REFERENCE_KL_SEMANTICS_RUN_NAME:-grpo_guard_telemetry_smoke_seed42}
output_dir=${REFERENCE_KL_SEMANTICS_OUTPUT:-results/experiments/rl_reference_kl_semantics_audit_20260802}

if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite non-empty output directory: $output_dir" >&2
  exit 2
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "CUDA_VISIBLE_DEVICES= .venv/bin/python evaluation/audit_reference_kl_semantics.py --experiment-root $experiment_root --output-dir $output_dir --run-name $run_name"
  exit 0
fi

CUDA_VISIBLE_DEVICES="" .venv/bin/python evaluation/audit_reference_kl_semantics.py \
  --experiment-root "$experiment_root" \
  --output-dir "$output_dir" \
  --run-name "$run_name"

#!/usr/bin/env bash
set -uo pipefail

OUT="${CEVAL_OUTPUT_DIR:-results/experiments/ceval_subset_20260801}"
REVISION="${CEVAL_REVISION:-617524a00b307ff6f9933702f724131fe12ca7ce}"
if [[ -e "$OUT" ]]; then
  echo "refusing to reuse existing C-Eval experiment directory: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"

models=(
  "align_sft_v2_pilot=out/align_sft_v2_pilot_768.pth"
  "simpo_v1_pilot=results/experiments/simpo_v1_pilot_20260731/out/simpo_v1_pilot_768.pth"
)
for mode in grpo cispo; do
  for seed in 42 43 44; do
    path="results/experiments/rl_method_upgrade_20260801/${mode}_seed${seed}/${mode}_seed${seed}_768.pth"
    if [[ -f "$path" ]]; then
      models+=("${mode}_seed${seed}=$path")
    else
      echo "OMITTED_MODEL=${mode}_seed${seed} reason=baseline_retained_or_failed path=$path" >> "$OUT/omitted_models.log"
    fi
  done
done

(
  echo "STARTED_AT=$(date -Is)"
  echo "REVISION=$REVISION"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "ENV_HASH=$(sha256sum evaluation/run_ceval_subset.py | sha256sum | cut -d' ' -f1)"
  echo "GPU_BEFORE=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
  echo "DISK_BEFORE=$(df -h . | tail -1)"
  set +e
  cmd=(.venv/bin/python evaluation/run_ceval_subset.py --revision "$REVISION" --subjects high_school_chinese,high_school_mathematics,high_school_physics,computer_network,business_ethics --questions-per-subject 20 --seed 42 --tokenizer-path model --output-dir "$OUT")
  for spec in "${models[@]}"; do cmd+=(--model "$spec"); done
  "${cmd[@]}"
  status=$?
  set -e
  echo "EXIT_CODE=$status"
  echo "GPU_AFTER=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
  echo "DISK_AFTER=$(df -h . | tail -1)"
  echo "FINISHED_AT=$(date -Is)"
  exit "$status"
) >"$OUT/run.log" 2>&1
status=$?
echo "CEVAL_RUN exit_code=$status output=$OUT"
exit "$status"

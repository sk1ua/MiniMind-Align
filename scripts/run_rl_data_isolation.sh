#!/usr/bin/env bash
set -uo pipefail

MODE="${1:---dry-run}"
ROOT="results/experiments/rl_data_isolation_20260801"
TRAIN_SOURCE="dataset/alignment_v2/manifests/train_manifest.jsonl"
VALIDATION_SOURCE="dataset/alignment_v2/manifests/validation_manifest.jsonl"
EXISTING_VALIDATION="results/inputs/on_policy_validation_manifest_32_20260731.jsonl"
TRAIN_MANIFEST="results/inputs/rl_data_isolation_train_128_20260801.jsonl"
VALIDATION_MANIFEST="results/inputs/rl_data_isolation_validation_32_20260801.jsonl"
SELECTION_META="results/inputs/rl_data_isolation_selection_20260801.json"
SESSION_NAME="${RL_DATA_ISOLATION_TMUX_SESSION:-rl_data_isolation_20260801}"
BUDGET_SECONDS="${RL_DATA_ISOLATION_BUDGET_SECONDS:-28800}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
CUMULATIVE_SECONDS=0

if [[ "$MODE" != "--dry-run" && "$MODE" != "--smoke" && "$MODE" != "--full" ]]; then
  echo "usage: $0 [--dry-run|--smoke|--full]" >&2
  exit 2
fi

if [[ "$MODE" == "--dry-run" ]]; then
  echo "RL data-isolation dry run: native v2 manifests, GRPO/CISPO seeds 42,43,44, max 32 steps"
  echo "experiment root: $ROOT"
  echo "budget seconds: $BUDGET_SECONDS"
  exit 0
fi

if [[ -z "${TMUX:-}" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required for --smoke/--full" >&2
    exit 2
  fi
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "refusing to reuse existing tmux session: $SESSION_NAME" >&2
    exit 2
  fi
  echo "starting detached tmux session: $SESSION_NAME"
  exec tmux new-session -d -s "$SESSION_NAME" "$SCRIPT_PATH" "$MODE"
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required for the 8-hour budget guard" >&2
  exit 2
fi

prepare_manifests() {
  local existing=0
  for path in "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$SELECTION_META"; do
    if [[ -e "$path" ]]; then
      existing=$((existing + 1))
    fi
  done
  if (( existing == 0 )); then
    .venv/bin/python scripts/prepare_rl_data_isolation.py \
      --train-input "$TRAIN_SOURCE" \
      --validation-input "$VALIDATION_SOURCE" \
      --existing-validation "$EXISTING_VALIDATION" \
      --train-output "$TRAIN_MANIFEST" \
      --validation-output "$VALIDATION_MANIFEST" \
      --selection-output "$SELECTION_META" \
      --seed 42 --train-per-category 16 --validation-per-category 4 --validation-offset 4
  elif (( existing != 3 )); then
    echo "partial isolation manifest set; refusing to continue" >&2
    return 2
  else
    .venv/bin/python - "$SELECTION_META" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if selection.get("status") != "PASS":
    raise SystemExit("selection metadata is not PASS")
for path, expected in ((Path(sys.argv[2]), 128), (Path(sys.argv[3]), 32)):
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected:
        raise SystemExit(f"{path}: expected {expected} rows, found {len(rows)}")
print("ISOLATION_MANIFESTS_REUSED_OK")
PY
  fi
}

run_one() {
  local mode="$1" seed="$2" output_dir="$3" save_weight="$4"
  shift 4
  local log_path="$output_dir/run.log"
  local monitor_path="$output_dir/resource_monitor.jsonl"
  if [[ -e "$output_dir" ]]; then
    echo "refusing to reuse existing experiment directory: $output_dir" >&2
    return 2
  fi
  mkdir -p "$output_dir"
  local remaining=$((BUDGET_SECONDS - CUMULATIVE_SECONDS))
  if (( remaining <= 0 )); then
    echo "GPU budget exhausted before $output_dir" >&2
    return 124
  fi
  local start_epoch
  start_epoch="$(date +%s)"
  local cmd=(.venv/bin/python trainer/train_grpo_lite.py
    --manifest "$TRAIN_MANIFEST"
    --validation-manifest "$VALIDATION_MANIFEST"
    --categories "conciseness,format,instruction,reasoning,repetition,safety,termination,uncertainty"
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
    echo "ENV_HASH=$(sha256sum trainer/train_grpo_lite.py evaluation/rl_validation.py evaluation/rl_selection.py evaluation/audit_rl_reward_hacking.py scripts/prepare_rl_data_isolation.py "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" | sha256sum | cut -d' ' -f1)"
    echo "GPU_BEFORE=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
    echo "DISK_BEFORE=$(df -h . | tail -1)"
    set +e
    timeout --signal=TERM "${remaining}s" "${cmd[@]}"
    status=$?
    set -e
    echo "EXIT_CODE=$status"
    echo "GPU_AFTER=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
    echo "DISK_AFTER=$(df -h . | tail -1)"
    echo "FINISHED_AT=$(date -Is)"
    echo "GPU_WALL_SECONDS=$(( $(date +%s) - start_epoch ))"
    exit "$status"
  ) >"$log_path" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    gpu_snapshot="$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ';' | sed 's/"/\\"/g')"
    disk_snapshot="$(df -h . | tail -1 | sed 's/"/\\"/g')"
    printf '{"timestamp":"%s","gpu":"%s","disk":"%s"}\n' "$(date -Is)" "$gpu_snapshot" "$disk_snapshot" >>"$monitor_path" || true
    sleep 60
  done
  wait "$pid"
  local status=$?
  local elapsed=$(( $(date +%s) - start_epoch ))
  CUMULATIVE_SECONDS=$((CUMULATIVE_SECONDS + elapsed))
  echo "RL_ISOLATION_RUN mode=$mode seed=$seed exit_code=$status wall_seconds=$elapsed cumulative_seconds=$CUMULATIVE_SECONDS"
  return "$status"
}

run_audit() {
  local audit_dir="$1"
  .venv/bin/python evaluation/audit_rl_reward_hacking.py \
    --experiment-root "$ROOT" \
    --output-dir "$audit_dir"
}

run_frozen_eval() {
  local frozen_root="$ROOT/frozen_eval"
  mkdir -p "$frozen_root"
  while IFS=$'\t' read -r name checkpoint; do
    [[ -z "$name" || -z "$checkpoint" ]] && continue
    local model_dir weight_name output_dir remaining start_epoch
    if [[ "$name" == "align_sft_v2_pilot" ]]; then
      model_dir="out"
      weight_name="align_sft_v2_pilot"
    else
      model_dir="$(dirname "$checkpoint")"
      weight_name="$(basename "$checkpoint" "_768.pth")"
    fi
    output_dir="$frozen_root/$name"
    if [[ -e "$output_dir" ]]; then
      echo "refusing to reuse frozen output: $output_dir" >&2
      return 2
    fi
    mkdir -p "$output_dir"
    remaining=$((BUDGET_SECONDS - CUMULATIVE_SECONDS))
    if (( remaining <= 0 )); then
      echo "GPU budget exhausted before frozen evaluation $name" >&2
      return 124
    fi
    start_epoch="$(date +%s)"
    set +e
    timeout --signal=TERM "${remaining}s" .venv/bin/python evaluation/generate_frozen_test.py \
      --weight "$weight_name" --model-name "$name" --model-dir "$model_dir" \
      --tokenizer-path model --device cuda:0 --output "$output_dir/generation.jsonl" \
      >"$output_dir/generate.log" 2>&1
    local status=$?
    if (( status == 0 )); then
      .venv/bin/python evaluation/score_frozen_test.py \
        --generation "$output_dir/generation.jsonl" \
        --output-dir "$output_dir/score" >>"$output_dir/generate.log" 2>&1
      status=$?
    fi
    set -e
    echo "EXIT_CODE=$status" >>"$output_dir/generate.log"
    echo "GPU_WALL_SECONDS=$(( $(date +%s) - start_epoch ))" >>"$output_dir/generate.log"
    CUMULATIVE_SECONDS=$((CUMULATIVE_SECONDS + $(date +%s) - start_epoch))
    if (( status != 0 )); then
      return "$status"
    fi
  done < <(.venv/bin/python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seen = set()
print("align_sft_v2_pilot\tout/align_sft_v2_pilot_768.pth")
seen.add(str((Path.cwd() / "out/align_sft_v2_pilot_768.pth").resolve()))
selection_paths = sorted(
    path for path in root.glob("*/selection.json")
    if path.parent.name.startswith(("grpo_seed", "cispo_seed"))
)
if not selection_paths:
    selection_paths = sorted(root.glob("*/selection.json"))
for selection_path in selection_paths:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    checkpoint = selection.get("selected_checkpoint")
    if not checkpoint:
        continue
    path = Path(checkpoint)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    key = str(path)
    if key in seen:
        continue
    seen.add(key)
    print(f"{selection_path.parent.name}\t{path}")
PY
)
}

prepare_manifests || exit $?
mkdir -p "$ROOT"
overall=0

if [[ "$MODE" == "--smoke" ]]; then
  run_one grpo 42 "$ROOT/smoke_grpo_seed42" smoke_grpo_seed42 \
    --max-prompts 8 --validation-max-prompts 2 --accumulation-steps 1 --num-generations 2 \
    --max-seq-len 384 --max-gen-len 16 --max-steps 2 --eval-every 2 --checkpoint-every 2 \
    --validation-max-new-tokens 16 || overall=1
  run_audit "$ROOT/audit_smoke" || overall=1
else
  for mode in grpo cispo; do
    for seed in 42 43 44; do
      run_one "$mode" "$seed" "$ROOT/${mode}_seed${seed}" "${mode}_seed${seed}" \
        --max-prompts 128 --validation-max-prompts 32 --accumulation-steps 8 --num-generations 8 \
        --max-seq-len 384 --max-gen-len 128 --max-steps 32 --eval-every 4 --checkpoint-every 4 \
        --validation-max-new-tokens 128 || overall=1
    done
  done
  run_audit "$ROOT/reward_hacking_audit" || overall=1
  if (( overall == 0 )); then
    run_frozen_eval || overall=1
  fi
fi

echo "RL_ISOLATION_TOTAL_GPU_WALL_SECONDS=$CUMULATIVE_SECONDS"
if (( CUMULATIVE_SECONDS > BUDGET_SECONDS )); then
  echo "GPU budget exceeded: $CUMULATIVE_SECONDS > $BUDGET_SECONDS" >&2
  overall=1
fi
exit "$overall"

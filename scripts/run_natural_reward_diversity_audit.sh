#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
ROOT="${NATURAL_REWARD_DIVERSITY_ROOT:-results/experiments/rl_natural_reward_diversity_audit_20260802}"
SAMPLE_ROOTS=(
  "results/experiments/rl_natural_rule_reward_smoke_20260802"
  "results/experiments/rl_data_isolation_reload_fixed_20260801"
  "results/experiments/rl_method_upgrade_20260801"
)
HARD_LIMIT_SECONDS="${NATURAL_REWARD_DIVERSITY_HARD_LIMIT_SECONDS:-600}"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--run" ]]; then
  echo "usage: $0 [--dry-run|--run]" >&2
  exit 2
fi
if [[ "$MODE" == "--dry-run" ]]; then
  echo "natural reward diversity audit dry run"
  echo "output root: $ROOT"
  echo "sample roots: ${SAMPLE_ROOTS[*]}"
  echo "CUDA: disabled; GPU wall time: 0; hard wall limit seconds: $HARD_LIMIT_SECONDS"
  exit 0
fi
if [[ -e "$ROOT" && -n "$(find "$ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to reuse non-empty audit root: $ROOT" >&2
  exit 2
fi
for sample_root in "${SAMPLE_ROOTS[@]}"; do
  [[ -e "$sample_root" ]] || { echo "missing sample root: $sample_root" >&2; exit 2; }
done
mkdir -p "$ROOT"
LOG_PATH="$ROOT/run.log"
START_EPOCH="$(date +%s)"

.venv/bin/python - "$ROOT/matrix.json" "$HARD_LIMIT_SECONDS" "${SAMPLE_ROOTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
limit = int(sys.argv[2])
roots = sys.argv[3:]
output.write_text(json.dumps({
    "schema_version": 1,
    "task": "MM-E028",
    "audit_task": "MM-F036",
    "mode": "offline_reward_diversity_audit",
    "cuda_disabled": True,
    "gpu_wall_seconds": 0,
    "sample_roots": roots,
    "expected_current_label": "e027_natural_rule_reward",
    "hard_limit_seconds": limit,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

CMD=(env CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python evaluation/audit_natural_reward_diversity.py
  --output-dir "$ROOT/audit"
  --expected-current-label e027_natural_rule_reward)
for sample_root in "${SAMPLE_ROOTS[@]}"; do
  CMD+=(--sample-root "$sample_root")
done

(
  echo "STARTED_AT=$(date -Is)"
  echo "TASK_IDS=MM-E028,MM-F036"
  echo "COMMAND=$(printf '%q ' "${CMD[@]}")"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "ENV_HASH=$(sha256sum evaluation/audit_natural_reward_diversity.py scripts/run_natural_reward_diversity_audit.sh $(find "${SAMPLE_ROOTS[@]}" -type f -name samples.jsonl -print | sort) | sha256sum | cut -d' ' -f1)"
  echo "GPU_BEFORE=$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  echo "DISK_BEFORE=$(df -h . | tail -1)"
  set +e
  timeout --signal=TERM "${HARD_LIMIT_SECONDS}s" "${CMD[@]}"
  STATUS=$?
  set -e
  echo "EXIT_CODE=$STATUS"
  echo "GPU_AFTER=$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  echo "DISK_AFTER=$(df -h . | tail -1)"
  echo "FINISHED_AT=$(date -Is)"
  echo "GPU_WALL_SECONDS=0"
  exit "$STATUS"
) >"$LOG_PATH" 2>&1

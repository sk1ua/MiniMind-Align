#!/usr/bin/env bash
set -euo pipefail
experiment_id=""
task_id=""
output_root="results/experiments"
while (($#)); do
  case "$1" in
    --experiment-id) experiment_id="$2"; shift 2 ;;
    --task-id) task_id="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --) shift; break ;;
    -h|--help) printf '%s\n' 'Usage: run_experiment.sh --experiment-id ID --task-id ID [--output-root DIR] -- COMMAND [ARGS...]'; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ -z "$experiment_id" || -z "$task_id" || "$#" -eq 0 ]]; then printf '%s\n' 'experiment id, task id and command are required' >&2; exit 2; fi
exp_dir="$output_root/$experiment_id"
if [[ -e "$exp_dir" ]]; then printf 'Refusing to overwrite existing experiment directory: %s\n' "$exp_dir" >&2; exit 3; fi
mkdir -p "$exp_dir"
start_epoch="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$start_epoch" > "$exp_dir/start_time.txt"
git rev-parse HEAD > "$exp_dir/git_commit.txt"
git status --short --branch > "$exp_dir/git_status.txt"
df -h / > "$exp_dir/disk_start.txt"
free -h > "$exp_dir/memory_start.txt"
nvidia-smi > "$exp_dir/nvidia_smi_start.txt" 2>&1 || true
python_bin="${PYTHON_BIN:-.venv/bin/python}"
"$python_bin" scripts/collect_environment.py --output-dir "$exp_dir/environment"
"$python_bin" - "$exp_dir/command.txt" "$@" <<'PY'
import re
import sys
from pathlib import Path
text = " ".join(sys.argv[2:])
text = re.sub(r"(?i)(AIza[0-9A-Za-z_-]{20,}|sk-[0-9A-Za-z_-]{16,}|gh[pousr]_[0-9A-Za-z_-]{16,})", "<REDACTED>", text)
text = re.sub(r"(?i)(--?(?:api[-_]?key|token|secret|password))\s+\S+", r"\1 <REDACTED>", text)
Path(sys.argv[1]).write_text(text + "\n", encoding="utf-8")
PY
set +e
("$@") > >(tee "$exp_dir/stdout.log") 2> >(tee "$exp_dir/stderr.log" >&2) &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  date -u +%Y-%m-%dT%H:%M:%SZ >> "$exp_dir/gpu_monitor.csv"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader >> "$exp_dir/gpu_monitor.csv" 2>/dev/null || true
  sleep 5
done
wait "$pid"
status=$?
set -e
end_epoch="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$end_epoch" > "$exp_dir/end_time.txt"
printf '%s\n' "$status" > "$exp_dir/exit_code.txt"
df -h / > "$exp_dir/disk_end.txt"
free -h > "$exp_dir/memory_end.txt"
nvidia-smi > "$exp_dir/nvidia_smi_end.txt" 2>&1 || true
if [[ "$status" -ne 0 ]]; then printf 'Experiment failed; logs retained at %s\n' "$exp_dir" >&2; fi
exit "$status"

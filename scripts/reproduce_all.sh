#!/usr/bin/env bash
set -euo pipefail

MODE=--dry-run
STAGE=all
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--smoke|--full) MODE="$1" ;;
    --stage) shift; STAGE="${1:?missing stage}" ;;
    -h|--help) echo "usage: $0 [--dry-run|--smoke|--full] [--stage data|sft|preferences|simpo|reward|rl|evaluation|all]"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

run_stage() {
  local name="$1"; shift
  if [[ "$STAGE" == all || "$STAGE" == "$name" ]]; then
    "$(dirname "$0")/$1" "$MODE"
  fi
}

run_stage data reproduce_data_v2.sh
run_stage sft reproduce_sft_v2.sh
run_stage preferences reproduce_preferences_v2.sh
run_stage simpo reproduce_simpo.sh
run_stage reward reproduce_reward_model.sh
run_stage rl reproduce_rl.sh
run_stage evaluation reproduce_evaluation.sh

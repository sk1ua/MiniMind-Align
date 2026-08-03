#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/reproduce_common.sh" "${1:---dry-run}"
run_cmd .venv/bin/python evaluation/build_unified_report.py --output-dir results/experiments/unified_sft_v2_20260731

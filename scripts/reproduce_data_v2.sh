#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/reproduce_common.sh" "${1:---dry-run}"
BUILD_MODE=smoke
[[ "$MODE" == "--full" ]] && BUILD_MODE=pilot
run_cmd .venv/bin/python dataset/alignment_v2/build_alignment_v2.py --mode "$BUILD_MODE"
run_cmd .venv/bin/python dataset/alignment_v2/audit_alignment_v2.py --mode "$BUILD_MODE"

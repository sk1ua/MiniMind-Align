#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_rl_suite.sh" "$@"

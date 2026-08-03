#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
case "$MODE" in
  --dry-run) DRY_RUN=1 ;;
  --smoke|--full) DRY_RUN=0 ;;
  *) echo "usage: $0 [--dry-run|--smoke|--full]" >&2; exit 2 ;;
esac

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" == 0 ]]; then
    "$@"
  fi
}

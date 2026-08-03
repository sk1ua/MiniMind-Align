#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON=python
exec "$PYTHON" "$ROOT/evaluation/audit_public_release.py" --root "$ROOT"

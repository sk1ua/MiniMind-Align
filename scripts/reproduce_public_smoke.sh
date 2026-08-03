#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON=python
exec "$PYTHON" -m pytest -q "$ROOT/tests/test_public_release.py"

"""Collect reproducibility metadata without printing secrets."""
from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def run_command(command: Sequence[str]) -> str:
    """Run a diagnostic command and return bounded output."""
    try:
        result = subprocess.run(list(command), check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"<unavailable: {exc}>"
    return (result.stdout + result.stderr).strip()[-20000:]


def collect(output_dir: Path) -> None:
    """Write diagnostics to a new output directory."""
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": str(Path.cwd()),
        "environment_names_only": sorted(key for key in os.environ if any(word in key.lower() for word in ("key", "token", "secret", "password"))),
        "nvidia_smi": run_command(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"]),
        "disk": run_command(["df", "-h", "/"]),
        "memory": run_command(["free", "-h"]),
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "git_status": run_command(["git", "status", "--short", "--branch"]),
        "pip_freeze": run_command([sys.executable, "-m", "pip", "freeze"]),
    }
    (output_dir / "environment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    collect(args.output_dir)


if __name__ == "__main__":
    main()

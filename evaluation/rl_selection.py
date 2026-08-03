"""Pure checkpoint quality gates and deterministic selection rules."""

from __future__ import annotations

from typing import Any


def quality_triggered(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    quality_drop_points: float = 10.0,
) -> dict[str, Any]:
    """Return quality-gate status using absolute percentage-point drops."""
    threshold = quality_drop_points / 100.0
    checks: dict[str, Any] = {}
    for name in ("safety", "termination"):
        key = f"{name}_pass_rate"
        current = metrics.get(key)
        reference = baseline.get(key)
        dropped = current is not None and reference is not None and current < reference - threshold
        checks[name] = {
            "baseline_rate": reference,
            "current_rate": current,
            "drop_points": (reference - current) * 100.0 if current is not None and reference is not None else None,
            "triggered": dropped,
        }
    return {"triggered": any(item["triggered"] for item in checks.values()), "checks": checks}


def select_best_checkpoint(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the best eligible checkpoint according to the registered tie-breaks."""
    eligible = [
        record for record in records
        if record.get("checkpoint") and not record.get("kl_triggered") and not record.get("quality_triggered")
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda record: (
            int(record["metrics"].get("validator_pass", 0)),
            int(record["metrics"].get("safety_pass", 0)),
            int(record["metrics"].get("termination_pass", 0)),
            int(record["metrics"].get("natural_end", 0)),
            -float(record["metrics"].get("average_repeat_3gram", 0.0)),
            -int(record.get("step", 0)),
        ),
    )

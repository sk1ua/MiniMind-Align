"""Compare SFT, DPO and SimPO candidates on the fixed v2 quality gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TARGET_CATEGORIES = ("conciseness", "format", "instruction", "reasoning", "repetition")


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return True


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not finite_tree(value):
        raise ValueError(f"invalid/nonfinite JSON: {path}")
    return value


def ensure_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def rate(block: dict, key: str, default: float = 0.0) -> float:
    count = float(block.get("count", 0))
    return float(block.get(key, 0)) / count if count else default


def summarize(label: str, payload: dict, baseline: dict | None = None) -> dict:
    full = payload.get("full", {})
    release = payload.get("release", {})
    categories = full.get("categories", {})
    target_count = sum(int(categories.get(category, {}).get("count", 0)) for category in TARGET_CATEGORIES)
    target_pass = sum(int(categories.get(category, {}).get("validator_pass", 0)) for category in TARGET_CATEGORIES)
    result = {
        "label": label,
        "weight_path": payload.get("weight_path"),
        "checkpoint_reload_ok": bool(payload.get("checkpoint_reload_ok")),
        "full_count": int(full.get("count", 0)),
        "full_validator_pass": int(full.get("validator_pass", 0)),
        "release_count": int(release.get("count", 0)),
        "release_validator_pass": int(release.get("validator_pass", 0)),
        "full_safety_rate": rate(full, "safety_pass"),
        "full_termination_rate": rate(full, "termination_pass"),
        "full_max_length_rate": rate(full, "max_length_hit"),
        "full_natural_end_rate": rate(full, "natural_end"),
        "full_repeat_3gram": float(full.get("average_repeat_3gram", 0.0)),
        "target_pass": target_pass,
        "target_count": target_count,
        "categories": categories,
    }
    if baseline is None:
        result["gate"] = {"baseline": True}
        return result
    gates = {
        "release_gain_at_least_3": result["release_validator_pass"] - baseline["release_validator_pass"] >= 3,
        "target_categories_not_lower": result["target_pass"] >= baseline["target_pass"],
        "full_validation_not_lower": result["full_validator_pass"] >= baseline["full_validator_pass"],
        "safety_drop_within_10_points": result["full_safety_rate"] >= baseline["full_safety_rate"] - 0.10,
        "termination_drop_within_10_points": result["full_termination_rate"] >= baseline["full_termination_rate"] - 0.10,
        "max_length_absolute_and_relative": (
            result["full_max_length_rate"] <= 0.25
            and result["full_max_length_rate"] <= baseline["full_max_length_rate"] + 0.05
        ),
        "natural_end_at_least_50_percent": result["full_natural_end_rate"] >= 0.50,
        "repeat_delta_within_005": result["full_repeat_3gram"] <= baseline["full_repeat_3gram"] + 0.05,
        "checkpoint_reload": result["checkpoint_reload_ok"],
    }
    result["deltas"] = {
        "full_validator_pass": result["full_validator_pass"] - baseline["full_validator_pass"],
        "release_validator_pass": result["release_validator_pass"] - baseline["release_validator_pass"],
        "target_pass": result["target_pass"] - baseline["target_pass"],
        "safety_rate": result["full_safety_rate"] - baseline["full_safety_rate"],
        "termination_rate": result["full_termination_rate"] - baseline["full_termination_rate"],
        "max_length_rate": result["full_max_length_rate"] - baseline["full_max_length_rate"],
        "natural_end_rate": result["full_natural_end_rate"] - baseline["full_natural_end_rate"],
        "repeat_3gram": result["full_repeat_3gram"] - baseline["full_repeat_3gram"],
    }
    result["gate"] = gates
    result["quality_gate_pass"] = all(gates.values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--sft-summary", type=Path, required=True)
    parser.add_argument("--dpo-summary", type=Path, required=True)
    parser.add_argument("--simpo-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ensure_empty(args.output_dir)
    data = read_json(args.data_manifest)
    if data.get("status") != "ERROR_DRIVEN_DATA_READY":
        raise ValueError("data manifest is not ready")
    baseline = summarize("align_sft_v2_pilot", read_json(args.baseline_summary))
    candidates = [
        summarize("error_driven_sft_seed42", read_json(args.sft_summary), baseline),
        summarize("error_driven_dpo_seed42", read_json(args.dpo_summary), baseline),
        summarize("error_driven_simpo_seed42", read_json(args.simpo_summary), baseline),
    ]
    incomplete = [
        candidate["label"]
        for candidate in candidates
        if candidate["full_count"] != 160 or candidate["release_count"] != 32
    ]
    passing = [candidate for candidate in candidates if candidate.get("quality_gate_pass")]
    if incomplete:
        status = "QUALITY_METHOD_TELEMETRY_INCOMPLETE"
    elif passing:
        status = "QUALITY_METHOD_PASS_DIAGNOSTIC_CORRECTED_GRPO_ELIGIBLE"
    else:
        status = "QUALITY_METHODS_NOT_MET_NO_MODEL_CHANGE"
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate["release_validator_pass"],
            candidate["full_validator_pass"],
            candidate["full_safety_rate"],
            candidate["full_termination_rate"],
            -candidate["full_repeat_3gram"],
        ),
        reverse=True,
    )
    result = {
        "schema_version": 1,
        "status": status,
        "data_manifest": str(args.data_manifest),
        "data_status": data.get("status"),
        "baseline": baseline,
        "candidates": candidates,
        "best_method": ranked[0]["label"] if ranked else None,
        "quality_gate_passing_methods": [candidate["label"] for candidate in passing],
        "next_decision": (
            "A separately approved corrected-GRPO smoke may use the best passing candidate."
            if passing
            else "Do not start corrected-GRPO or formal RL; improve or expand supervised/preference data first."
        ),
        "diagnostic_only": True,
        "default_model_changed": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        "# Error-driven SFT/DPO/SimPO comparison\n\n"
        f"- Status: {status}\n"
        f"- Best observed method: {result['best_method']}\n"
        f"- Methods passing all quality gates: {result['quality_gate_passing_methods']}\n"
        f"- Next decision: {result['next_decision']}\n"
        "- No candidate automatically replaces the default model.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

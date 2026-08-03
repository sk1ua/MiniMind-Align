"""Attribute zero validation gain to category failures and reward coverage offline."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = (
    "conciseness",
    "format",
    "instruction",
    "reasoning",
    "repetition",
    "safety",
    "termination",
    "uncertainty",
)
COMPONENTS = (
    "validator_reward",
    "parse_reward",
    "field_reward",
    "item_count_reward",
    "arithmetic_reward",
    "format_reward",
    "termination_reward",
    "repetition_penalty",
)


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return True


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not finite_tree(value):
        raise ValueError(f"invalid or non-finite JSON: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not finite_tree(value):
            raise ValueError(f"invalid or non-finite JSONL row {path}:{line_number}")
        rows.append(value)
    return rows


def ensure_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def metric_value(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(default if value is None else value)


def validation_snapshot(row: dict) -> dict[str, object]:
    details = row.get("details", [])
    by_category: dict[str, dict[str, object]] = {}
    failure_reasons: Counter[str] = Counter()
    for category in CATEGORIES:
        category_rows = [item for item in details if item.get("category") == category]
        failures = [item for item in category_rows if item.get("validator_pass") is not True]
        for item in failures:
            failure_reasons[str(item.get("validator_problem") or "unknown")] += 1
        by_category[category] = {
            "count": len(category_rows),
            "validator_pass": sum(item.get("validator_pass") is True for item in category_rows),
            "validator_fail": len(failures),
            "failure_reasons": dict(Counter(str(item.get("validator_problem") or "unknown") for item in failures)),
            "natural_end": sum(item.get("finished_naturally") is True for item in category_rows),
            "max_length_hit": sum(
                item.get("finished_naturally") is not True
                and int(item.get("generated_tokens", -1)) >= 128
                for item in category_rows
            ),
            "mean_tokens": (
                sum(metric_value(item, "generated_tokens") for item in category_rows) / len(category_rows)
                if category_rows
                else None
            ),
            "mean_repeat_3gram": (
                sum(metric_value(item, "repeat_3gram_ratio") for item in category_rows) / len(category_rows)
                if category_rows
                else None
            ),
        }
    return {
        "step": row.get("step"),
        "validation": row.get("validation"),
        "checkpoint": row.get("checkpoint"),
        "count": len(details),
        "validator_pass": int(row.get("metrics", {}).get("validator_pass", 0) or 0),
        "natural_end": int(row.get("metrics", {}).get("natural_end", 0) or 0),
        "failure_reasons": dict(failure_reasons),
        "categories": by_category,
        "detail_ids": [str(item.get("id")) for item in details],
    }


def compare_validation(baseline: dict, selected: dict) -> dict[str, object]:
    base_items = {str(item.get("id")): item for item in baseline.get("details", [])}
    selected_items = {str(item.get("id")): item for item in selected.get("details", [])}
    shared_ids = sorted(set(base_items) & set(selected_items))
    transitions = Counter()
    changed: list[dict[str, object]] = []
    for item_id in shared_ids:
        before = bool(base_items[item_id].get("validator_pass"))
        after = bool(selected_items[item_id].get("validator_pass"))
        transition = "stable_pass" if before and after else "stable_fail" if not before and not after else "improved" if after else "degraded"
        transitions[transition] += 1
        if transition in {"improved", "degraded"}:
            changed.append({
                "id": item_id,
                "category": selected_items[item_id].get("category", base_items[item_id].get("category")),
                "transition": transition,
                "baseline_failure_reason": base_items[item_id].get("validator_problem"),
                "selected_failure_reason": selected_items[item_id].get("validator_problem"),
            })
    return {
        "baseline_detail_count": len(base_items),
        "selected_detail_count": len(selected_items),
        "shared_detail_count": len(shared_ids),
        "id_sets_identical": set(base_items) == set(selected_items),
        "transitions": dict(transitions),
        "changed_items": changed,
        "validator_gain": sum(item["transition"] == "improved" for item in changed)
        - sum(item["transition"] == "degraded" for item in changed),
    }


def reward_coverage(samples: list[dict]) -> dict[str, object]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    by_step: dict[int, list[dict]] = defaultdict(list)
    sample_keys: set[str] = set()
    linkage_ok = True
    for row in samples:
        category = row.get("category")
        if category:
            by_category[str(category)].append(row)
        if "step" in row:
            by_step[int(row["step"])].append(row)
        key = row.get("sample_key")
        linkage_ok &= bool(key and row.get("prompt_id") and row.get("category")) and key not in sample_keys
        if key:
            sample_keys.add(str(key))

    def summarize(rows: list[dict]) -> dict[str, object]:
        components: dict[str, dict[str, object]] = {}
        for component in COMPONENTS:
            values = [metric_value(row.get("components", {}), component) for row in rows]
            components[component] = {
                "nonzero_count": sum(value != 0.0 for value in values),
                "coverage": (sum(value != 0.0 for value in values) / len(values) if values else 0.0),
                "mean": (sum(values) / len(values) if values else 0.0),
            }
        return {
            "count": len(rows),
            "prompt_count": len({row.get("prompt_id") for row in rows}),
            "validator_pass": sum(metric_value(row.get("components", {}), "validator_reward") > 0 for row in rows),
            "validator_pass_rate": (
                sum(metric_value(row.get("components", {}), "validator_reward") > 0 for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "empty_response": sum(row.get("empty_response") is True for row in rows),
            "max_length_hit": sum(row.get("max_length_hit") is True for row in rows),
            "natural_end": sum(row.get("finished_naturally") is True for row in rows),
            "mean_reward": (sum(metric_value(row, "reward") for row in rows) / len(rows) if rows else 0.0),
            "mean_termination_reward": (
                sum(metric_value(row.get("components", {}), "termination_reward") for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "mean_repetition_penalty": (
                sum(metric_value(row.get("components", {}), "repetition_penalty") for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "components": components,
        }

    return {
        "sample_count": len(samples),
        "unique_sample_keys": len(sample_keys),
        "linkage_ok": linkage_ok and len(sample_keys) == len(samples),
        "category_counts": {category: len(by_category.get(category, [])) for category in CATEGORIES},
        "by_category": {category: summarize(by_category.get(category, [])) for category in CATEGORIES},
        "by_step": {str(step): summarize(rows) for step, rows in sorted(by_step.items())},
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.experiment_root)
    run_dir = root / args.run_name
    output_dir = Path(args.output_dir)
    ensure_empty_output_dir(output_dir)
    required = [
        root / "matrix.json",
        run_dir / "baseline_validation.json",
        run_dir / "validation_history.jsonl",
        run_dir / "samples.jsonl",
        run_dir / "step_summaries.jsonl",
        run_dir / "selection.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        result = {"status": "QUALITY_FAILURE_ATTRIBUTION_INCOMPLETE", "missing": missing, "diagnostic_only": True}
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    try:
        matrix = load_json(root / "matrix.json")
        baseline_payload = load_json(run_dir / "baseline_validation.json")
        validation = load_jsonl(run_dir / "validation_history.jsonl")
        samples = load_jsonl(run_dir / "samples.jsonl")
        steps = load_jsonl(run_dir / "step_summaries.jsonl")
        selection = load_json(run_dir / "selection.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"status": "QUALITY_FAILURE_ATTRIBUTION_INCOMPLETE", "error": str(exc), "diagnostic_only": True}
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    baseline = {"validation": "baseline", **baseline_payload}
    selected_checkpoint = selection.get("selected_checkpoint")
    selected = next(
        (
            row
            for row in validation
            if row.get("checkpoint") == selected_checkpoint
            or row.get("step") == selection.get("selected_step")
        ),
        None,
    )
    if baseline is None or selected is None:
        result = {
            "status": "QUALITY_FAILURE_ATTRIBUTION_INCOMPLETE",
            "missing_evidence": {"baseline": baseline is None, "selected_checkpoint": selected is None},
            "diagnostic_only": True,
        }
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    baseline_snapshot = validation_snapshot(baseline)
    selected_snapshot = validation_snapshot(selected)
    transitions = compare_validation(baseline, selected)
    rewards = reward_coverage(samples)
    step_reward = {
        str(row.get("step")): {
            "train_validator_pass_rate": row.get("train_validator_pass_rate"),
            "train_reward_components": row.get("train_reward_components"),
            "train_repetition_penalty_mean": row.get("train_repetition_penalty_mean"),
            "train_natural_end_rate": row.get("train_natural_end_rate"),
            "train_max_length_hit_rate": row.get("train_max_length_hit_rate"),
            "advantage_nonzero_rate": row.get("advantage_nonzero_rate"),
            "reward_mean": row.get("reward_mean"),
        }
        for row in steps
    }
    category_counts_ok = all(
        baseline_snapshot["categories"][category]["count"] == 4
        and selected_snapshot["categories"][category]["count"] == 4
        for category in CATEGORIES
    )
    complete = (
        baseline_snapshot["count"] == 32
        and selected_snapshot["count"] == 32
        and transitions["id_sets_identical"] is True
        and transitions["shared_detail_count"] == 32
        and category_counts_ok
        and rewards["sample_count"] == 256
        and rewards["linkage_ok"] is True
        and sorted(int(step) for step in step_reward) == [1, 2, 3, 4]
    )
    warnings = [
        "RL_VALIDATOR_GAIN_ZERO_ON_FULL_BALANCED_SLICE" if transitions["validator_gain"] == 0 else "RL_VALIDATOR_DIRECTIONAL_CHANGE_SINGLE_SEED",
        "SFT_SOURCE_AND_RL_SELECTED_OUTPUTS_MATCHED" if transitions["validator_gain"] == 0 else "SFT_SOURCE_AND_RL_SELECTED_OUTPUTS_DIFFERED",
        "BF16_SHADOW_AND_PRESTEP_PRECISION_WARNINGS_RETAINED",
        "SINGLE_SEED_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE",
    ]
    result = {
        "status": "QUALITY_FAILURE_ATTRIBUTION_COMPLETE" if complete else "QUALITY_FAILURE_ATTRIBUTION_INCOMPLETE",
        "diagnostic_only": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "default_model_changed": False,
        "experiment_root": str(root),
        "selected_checkpoint": selected_checkpoint,
        "scope": {
            "validation_count": 32,
            "balanced_categories": CATEGORIES,
            "training_sample_count": len(samples),
            "training_steps": sorted(int(step) for step in step_reward),
        },
        "baseline": baseline_snapshot,
        "selected": selected_snapshot,
        "transitions": transitions,
        "reward_coverage": rewards,
        "step_reward_telemetry": step_reward,
        "warnings": warnings,
        "complete": complete,
        "formal_rl_ready": False,
        "automatic_gpu_start": False,
        "next_decision": "No new GPU training is justified by this single-seed zero-gain attribution alone; any future plan must target the identified category/failure path and retain independent validation.",
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        "\n".join([
            "# RL quality failure attribution audit",
            "",
            f"- status: {result['status']}",
            f"- source validator: {baseline_snapshot['validator_pass']}/32",
            f"- selected validator: {selected_snapshot['validator_pass']}/32",
            f"- validator gain: {transitions['validator_gain']}",
            f"- transitions: {transitions['transitions']}",
            "",
            "This is an offline single-seed attribution diagnostic; it does not authorize model promotion.",
            "",
        ]),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    print(json.dumps(audit(parser.parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()

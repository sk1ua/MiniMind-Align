"""Audit category exposure and group-advantage signal transmission offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
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


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not finite_tree(value):
            raise ValueError(f"invalid/nonfinite JSONL row {path}:{line_number}")
        rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def number(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(default if value is None else value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def group_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean = statistics.fmean(rewards)
    variance = statistics.fmean([(value - mean) ** 2 for value in rewards])
    std = math.sqrt(variance)
    return [(value - mean) / (std + 1e-4) for value in rewards]


def summarize_group(rows: list[dict]) -> dict[str, object]:
    rewards = [number(row.get("reward")) for row in rows]
    advantages = group_advantages(rewards)
    categories = {str(row.get("category")) for row in rows}
    return {
        "step": rows[0].get("step"),
        "prompt_id": rows[0].get("prompt_id"),
        "micro_index": rows[0].get("micro_index"),
        "sample_count": len(rows),
        "category_count": len(categories),
        "categories": sorted(categories),
        "reward_mean": statistics.fmean(rewards) if rewards else 0.0,
        "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "advantage_nonzero_count": sum(abs(value) > 1e-12 for value in advantages),
        "advantage_nonzero_rate": sum(abs(value) > 1e-12 for value in advantages) / len(advantages) if advantages else 0.0,
        "advantage_abs_sum": sum(abs(value) for value in advantages),
        "advantage_abs_mean": statistics.fmean(abs(value) for value in advantages) if advantages else 0.0,
        "advantage_min": min(advantages) if advantages else 0.0,
        "advantage_max": max(advantages) if advantages else 0.0,
        "collapsed": max(rewards) - min(rewards) <= 1e-12 if rewards else True,
    }


def summarize_category(rows: list[dict], groups: list[dict]) -> dict[str, object]:
    rewards = [number(row.get("reward")) for row in rows]
    components = [row.get("components") or {} for row in rows]
    category_groups = [group for group in groups if group["categories"] == [str(rows[0].get("category"))]] if rows else []
    return {
        "sample_count": len(rows),
        "prompt_count": len({str(row.get("prompt_id")) for row in rows}),
        "family_count": len({str(row.get("family", "<not_persisted>")) for row in rows}),
        "validator_pass": sum(number(component.get("validator_reward")) > 0 for component in components),
        "validator_pass_rate": sum(number(component.get("validator_reward")) > 0 for component in components) / len(rows) if rows else 0.0,
        "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
        "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_termination_reward": statistics.fmean(number(component.get("termination_reward")) for component in components) if components else 0.0,
        "mean_repetition_penalty": statistics.fmean(number(component.get("repetition_penalty")) for component in components) if components else 0.0,
        "group_count": len(category_groups),
        "collapsed_group_count": sum(bool(group["collapsed"]) for group in category_groups),
        "advantage_nonzero_rate": (
            sum(group["advantage_nonzero_count"] for group in category_groups)
            / sum(group["sample_count"] for group in category_groups)
            if category_groups else 0.0
        ),
        "advantage_abs_sum": sum(group["advantage_abs_sum"] for group in category_groups),
        "advantage_abs_mean": (
            statistics.fmean(group["advantage_abs_mean"] for group in category_groups)
            if category_groups else 0.0
        ),
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.experiment_root)
    run_dir = root / args.run_name
    output_dir = Path(args.output_dir)
    ensure_empty_output_dir(output_dir)
    sample_path = run_dir / "samples.jsonl"
    steps_path = run_dir / "step_summaries.jsonl"
    manifest_path = Path(args.manifest) if args.manifest else run_dir / "resolved_train_manifest.jsonl"
    required = [sample_path, steps_path, manifest_path]
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        result = {"status": "TELEMETRY_INCOMPLETE", "missing": missing, "diagnostic_only": True}
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    samples = read_jsonl(sample_path)
    steps = read_jsonl(steps_path)
    manifest = read_jsonl(manifest_path)
    manifest_by_id = {str(row.get("id")): row for row in manifest if row.get("id") is not None}
    linkage_errors: list[str] = []
    by_category: dict[str, list[dict]] = defaultdict(list)
    groups_raw: dict[tuple[object, ...], list[dict]] = defaultdict(list)
    sample_keys: set[str] = set()
    for row in samples:
        key = str(row.get("sample_key", ""))
        if not key or key in sample_keys:
            linkage_errors.append(f"duplicate_or_missing_sample_key:{key}")
        sample_keys.add(key)
        prompt_id = str(row.get("prompt_id", ""))
        if prompt_id not in manifest_by_id:
            linkage_errors.append(f"missing_prompt:{prompt_id}")
        manifest_row = manifest_by_id.get(prompt_id)
        if manifest_row and str(row.get("category")) != str(manifest_row.get("category")):
            linkage_errors.append(f"category_mismatch:{key}")
        annotated_row = dict(row)
        annotated_row["family"] = manifest_row.get("family") if manifest_row else "<missing>"
        by_category[str(row.get("category"))].append(annotated_row)
        groups_raw[(row.get("step"), row.get("prompt_id"), row.get("micro_index"))].append(row)

    group_summaries = [summarize_group(rows) for rows in groups_raw.values()]
    mixed_groups = [group for group in group_summaries if group["category_count"] != 1]
    bad_group_sizes = [group for group in group_summaries if group["sample_count"] != args.num_generations]
    category_summary = {
        category: summarize_category(by_category.get(category, []), group_summaries)
        for category in CATEGORIES
    }
    family_summary: dict[str, dict[str, int]] = {}
    for category in CATEGORIES:
        family_summary[category] = dict(sorted(Counter(
            str(manifest_by_id[str(row.get("prompt_id"))].get("family"))
            for row in by_category.get(category, [])
            if str(row.get("prompt_id")) in manifest_by_id
        ).items()))

    step_summary = []
    for row in steps:
        step_summary.append({
            "step": row.get("step"),
            "train_sample_count": row.get("train_sample_count"),
            "unique_prompt_count": row.get("unique_prompt_count"),
            "train_validator_pass_rate": row.get("train_validator_pass_rate"),
            "reward_mean": row.get("reward_mean"),
            "reward_std_mean": row.get("reward_std_mean"),
            "advantage_nonzero_rate": row.get("advantage_nonzero_rate"),
            "train_reward_components": row.get("train_reward_components"),
        })

    category_sample_counts = {category: len(by_category.get(category, [])) for category in CATEGORIES}
    category_group_counts = {
        category: sum(group["categories"] == [category] for group in group_summaries)
        for category in CATEGORIES
    }
    exposure_balanced = len(set(category_sample_counts.values())) == 1 and len(set(category_group_counts.values())) == 1
    mean_abs_advantages = [float(category_summary[category]["advantage_abs_mean"]) for category in CATEGORIES]
    signal_heterogeneous = max(mean_abs_advantages, default=0.0) - min(mean_abs_advantages, default=0.0) > 1e-6
    observed_prompt_ids = {str(row.get("prompt_id")) for row in samples}
    observed_families = {
        str(manifest_by_id[str(row.get("prompt_id"))].get("family"))
        for row in samples
        if str(row.get("prompt_id")) in manifest_by_id
    }
    all_families = {str(row.get("family")) for row in manifest}
    warnings = ["SINGLE_SEED_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE"]
    if exposure_balanced:
        warnings.append("CATEGORY_SAMPLE_EXPOSURE_BALANCED")
    else:
        warnings.append("CATEGORY_SAMPLE_EXPOSURE_UNBALANCED")
    if signal_heterogeneous:
        warnings.append("CATEGORY_ADVANTAGE_SIGNAL_HETEROGENEOUS")
    if len(observed_families) < len(all_families):
        warnings.append("FAMILY_EXPOSURE_PARTIAL")
    if any(number(row.get("train_validator_pass_rate")) == 0.0 for row in step_summary):
        warnings.append("STEP_VALIDATOR_SIGNAL_ABSENT")

    telemetry_incomplete = (
        not samples
        or not steps
        or linkage_errors
        or mixed_groups
        or bad_group_sizes
        or len(group_summaries) == 0
        or any(category_sample_counts[category] == 0 for category in CATEGORIES)
    )
    status = "TELEMETRY_INCOMPLETE" if telemetry_incomplete else (
        "CATEGORY_EXPOSURE_BALANCED_ADVANTAGE_HETEROGENEOUS_DIAGNOSTIC"
        if exposure_balanced and signal_heterogeneous
        else "CATEGORY_EXPOSURE_BALANCED_ADVANTAGE_UNRESOLVED_DIAGNOSTIC"
        if exposure_balanced
        else "CATEGORY_EXPOSURE_UNBALANCED_DIAGNOSTIC"
    )
    input_manifest = [
        {"path": path.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (sample_path, steps_path, manifest_path)
    ]
    result = {
        "status": status,
        "diagnostic_only": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "default_model_changed": False,
        "experiment_root": str(root),
        "run_name": args.run_name,
        "input_manifest": input_manifest,
        "scope": {
            "sample_count": len(samples),
            "step_count": len(steps),
            "group_count": len(group_summaries),
            "num_generations": args.num_generations,
            "prompt_coverage": f"{len(observed_prompt_ids)}/{len(manifest_by_id)}",
            "family_coverage": f"{len(observed_families)}/{len(all_families)}",
        },
        "exposure": {
            "sample_counts": category_sample_counts,
            "group_counts": category_group_counts,
            "balanced": exposure_balanced,
            "family_counts_by_category": family_summary,
        },
        "signal": {
            "by_category": category_summary,
            "by_step": step_summary,
            "group_collapsed_count": sum(bool(group["collapsed"]) for group in group_summaries),
            "group_nonzero_count": sum(not bool(group["collapsed"]) for group in group_summaries),
            "mean_abs_advantage_range": [min(mean_abs_advantages, default=0.0), max(mean_abs_advantages, default=0.0)],
            "heterogeneous": signal_heterogeneous,
        },
        "integrity": {
            "linkage_error_count": len(linkage_errors),
            "mixed_group_count": len(mixed_groups),
            "bad_group_size_count": len(bad_group_sizes),
            "unique_sample_key_count": len(sample_keys),
        },
        "warnings": warnings,
        "execution": {"gpu_started": False, "gpu_wall_seconds": 0, "server_status": "RUNNING"},
        "next_decision": "Balanced sample exposure does not imply balanced learning signal. Use this result only to distinguish exposure from signal heterogeneity; do not change category weights or start formal RL without a separate plan and independent validation.",
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "group_summaries.jsonl").write_text(
        "".join(json.dumps(group, ensure_ascii=False, allow_nan=False) + "\n" for group in group_summaries),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join([
            "# RL category weighting and advantage audit",
            "",
            f"- status: `{status}`",
            f"- samples: {len(samples)}; groups: {len(group_summaries)}; exposure balanced: {exposure_balanced}",
            f"- prompt coverage: {len(observed_prompt_ids)}/{len(manifest_by_id)}; family coverage: {len(observed_families)}/{len(all_families)}",
            f"- collapsed groups: {sum(bool(group['collapsed']) for group in group_summaries)}",
            f"- category mean absolute advantage range: {result['signal']['mean_abs_advantage_range']}",
            "",
            "Offline diagnostic only; no reward, optimizer, model or formal-RL decision is made here.",
            "",
        ]),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--num-generations", type=int, default=8)
    result = audit(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] == "TELEMETRY_INCOMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

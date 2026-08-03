"""Audit natural rule-reward component diversity and within-group collapse offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.audit_corrected_kl_gate import ensure_empty_output_dir

COMPONENT_KEYS = (
    "validator_reward",
    "parse_reward",
    "field_reward",
    "item_count_reward",
    "arithmetic_reward",
    "format_reward",
    "termination_reward",
    "repetition_penalty",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return True


def _float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def source_label(path: Path) -> str:
    text = path.as_posix()
    if "rl_natural_rule_reward_smoke_20260802" in text:
        return "e027_natural_rule_reward"
    if "rl_data_isolation_reload_fixed_20260801" in text:
        return "e010_v2_reload_fixed_legacy_schema"
    if "rl_method_upgrade_20260801" in text:
        return "e009_v1_legacy_schema"
    return "unclassified_natural_reward_artifact"


def collect_sample_files(sample_roots: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for root in sample_roots:
        if root.is_file() and root.name == "samples.jsonl":
            files.add(root)
        elif root.is_dir():
            files.update(root.rglob("samples.jsonl"))
    return sorted(files)


def _group_id(path: Path, row: dict) -> tuple[object, ...]:
    micro = row.get("micro_index")
    return (
        source_label(path),
        path.as_posix(),
        row.get("step"),
        row.get("prompt_id"),
        micro if micro is not None else "legacy_no_micro_index",
    )


def _component_summary(rows: list[dict], component: str) -> dict[str, object]:
    values: list[float] = []
    missing = 0
    nonzero = 0
    for row in rows:
        value = _float((row.get("components") or {}).get(component))
        if value is None:
            missing += 1
            continue
        values.append(value)
        if abs(value) > 1e-12:
            nonzero += 1
    return {
        "observed_count": len(values),
        "missing_count": missing,
        "coverage_rate": len(values) / len(rows) if rows else 0.0,
        "nonzero_count": nonzero,
        "nonzero_rate": nonzero / len(values) if values else 0.0,
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "unique_count": len(set(values)),
    }


def summarize_source(label: str, files: list[Path], rows: list[tuple[Path, dict]]) -> dict[str, object]:
    samples = [row for _, row in rows]
    reward_values = [value for row in samples if (value := _float(row.get("reward"))) is not None]
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    categories = Counter(str(row.get("category", "<missing>")) for row in samples)
    reward_sources = Counter(str(row.get("reward_source", "legacy_missing")) for row in samples)
    for path, row in rows:
        reward = _float(row.get("reward"))
        if reward is not None:
            groups[_group_id(path, row)].append(reward)
    eligible_groups = [values for values in groups.values() if len(values) >= 2]
    collapsed_groups = [values for values in eligible_groups if max(values) - min(values) <= 1e-12]
    spread_groups = [values for values in eligible_groups if max(values) - min(values) > 1e-12]
    nonzero_components = {
        component
        for component in COMPONENT_KEYS
        if any(
            abs(_float((row.get("components") or {}).get(component)) or 0.0) > 1e-12
            for row in samples
        )
    }
    invalid_rows = sum(not finite_tree(row) for row in samples)
    return {
        "label": label,
        "sample_file_count": len(files),
        "sample_files": [path.as_posix() for path in files],
        "sample_count": len(samples),
        "run_count": len({path.parent.name for path, _ in rows}),
        "category_counts": dict(sorted(categories.items())),
        "reward_source_counts": dict(sorted(reward_sources.items())),
        "reward_missing_count": len(samples) - len(reward_values),
        "reward_unique_count": len(set(reward_values)),
        "reward_mean": statistics.fmean(reward_values) if reward_values else None,
        "reward_std_population": statistics.pstdev(reward_values) if len(reward_values) > 1 else 0.0,
        "reward_min": min(reward_values) if reward_values else None,
        "reward_max": max(reward_values) if reward_values else None,
        "component_nonzero_keys": sorted(nonzero_components),
        "component_summary": {
            component: _component_summary(samples, component) for component in COMPONENT_KEYS
        },
        "group_count": len(groups),
        "eligible_group_count": len(eligible_groups),
        "collapsed_group_count": len(collapsed_groups),
        "collapsed_group_rate": len(collapsed_groups) / len(eligible_groups) if eligible_groups else None,
        "nonzero_reward_spread_group_count": len(spread_groups),
        "nonzero_reward_spread_group_rate": len(spread_groups) / len(eligible_groups) if eligible_groups else None,
        "group_size_min": min((len(values) for values in groups.values()), default=0),
        "group_size_max": max((len(values) for values in groups.values()), default=0),
        "natural_end_rate": sum(bool(row.get("finished_naturally")) for row in samples) / len(samples) if samples else 0.0,
        "max_length_hit_rate": sum(bool(row.get("max_length_hit")) for row in samples) / len(samples) if samples else None,
        "empty_response_rate": sum(bool(row.get("empty_response")) for row in samples) / len(samples) if samples else 0.0,
        "validator_nonzero_rate": sum(
            (_float((row.get("components") or {}).get("validator_reward")) or 0.0) > 0.0
            for row in samples
        ) / len(samples) if samples else 0.0,
        "termination_nonzero_rate": sum(
            (_float((row.get("components") or {}).get("termination_reward")) or 0.0) > 0.0
            for row in samples
        ) / len(samples) if samples else 0.0,
        "repetition_penalty_nonzero_rate": sum(
            (_float((row.get("components") or {}).get("repetition_penalty")) or 0.0) > 0.0
            for row in samples
        ) / len(samples) if samples else 0.0,
        "invalid_or_nonfinite_rows": invalid_rows,
    }


def audit_paths(
    sample_roots: list[Path],
    output_dir: Path,
    expected_current_label: str = "e027_natural_rule_reward",
) -> dict[str, object]:
    ensure_empty_output_dir(output_dir)
    files = collect_sample_files(sample_roots)
    input_manifest = []
    rows_by_label: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    files_by_label: dict[str, list[Path]] = defaultdict(list)
    parse_errors: list[dict[str, str]] = []
    for path in files:
        label = source_label(path)
        files_by_label[label].append(path)
        input_manifest.append({
            "path": path.as_posix(),
            "label": label,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append({"path": path.as_posix(), "line": str(line_number), "error": str(exc)})
                continue
            rows_by_label[label].append((path, row))
    summaries = {
        label: summarize_source(label, files_by_label[label], rows)
        for label, rows in sorted(rows_by_label.items())
    }
    current = summaries.get(expected_current_label)
    warnings: list[str] = []
    if current and current["collapsed_group_rate"] == 1.0:
        warnings.append("NATURAL_REWARD_GROUP_COLLAPSE")
    if current and current["component_nonzero_keys"] == ["termination_reward"]:
        warnings.append("NATURAL_REWARD_TERMINATION_ONLY")
    if current and current["nonzero_reward_spread_group_count"] == 0:
        warnings.append("NO_LIVE_NONZERO_GROUP_ADVANTAGE_EVIDENCE")
    if any(label.endswith("legacy_schema") for label in summaries):
        warnings.append("LEGACY_SCHEMA_SOURCES_REPORTED_SEPARATELY")
    if current is None or current["sample_count"] == 0 or parse_errors:
        status = "TELEMETRY_INCOMPLETE"
    elif current["nonzero_reward_spread_group_count"] == 0:
        status = "NATURAL_REWARD_DIVERSITY_AUDIT_COLLAPSE_CONFIRMED"
    else:
        status = "NATURAL_REWARD_DIVERSITY_PRESENT_DIAGNOSTIC"
    result = {
        "status": status,
        "expected_current_label": expected_current_label,
        "sample_roots": [path.as_posix() for path in sample_roots],
        "input_file_count": len(files),
        "input_manifest": input_manifest,
        "parse_errors": parse_errors,
        "source_summaries": summaries,
        "warnings": warnings,
        "diagnostic_only": True,
        "gpu_wall_seconds": 0,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "default_model_changed": False,
        "limitation": "This is an offline coverage and collapse audit. It does not prove reward causality, model quality or an optimizer remedy.",
    }
    (output_dir / "input_manifest.json").write_text(
        json.dumps(input_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Natural reward diversity audit",
        "",
        f"- status: `{status}`",
        f"- input files: {len(files)}",
        f"- current source: `{expected_current_label}`",
        "",
    ]
    for label, summary in summaries.items():
        lines.extend([
            f"## {label}",
            "",
            f"- samples: {summary['sample_count']}; groups: {summary['group_count']}",
            f"- reward mean/min/max/unique: {summary['reward_mean']} / {summary['reward_min']} / {summary['reward_max']} / {summary['reward_unique_count']}",
            f"- eligible groups collapsed: {summary['collapsed_group_count']}/{summary['eligible_group_count']} ({summary['collapsed_group_rate']})",
            f"- groups with nonzero reward spread: {summary['nonzero_reward_spread_group_count']}/{summary['eligible_group_count']}",
            f"- nonzero components: {summary['component_nonzero_keys']}",
            "",
        ])
    lines.extend([
        "Warnings: " + (", ".join(warnings) if warnings else "none"),
        "",
        "Diagnostic only; no model or optimizer decision is made by this audit.",
        "",
    ])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-current-label", default="e027_natural_rule_reward")
    args = parser.parse_args()
    roots = args.sample_root or [
        ROOT / "results/experiments/rl_natural_rule_reward_smoke_20260802",
        ROOT / "results/experiments/rl_data_isolation_reload_fixed_20260801",
        ROOT / "results/experiments/rl_method_upgrade_20260801",
    ]
    result = audit_paths(roots, args.output_dir, args.expected_current_label)
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] == "TELEMETRY_INCOMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

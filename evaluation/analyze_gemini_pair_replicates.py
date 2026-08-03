from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对多轮 Gemini pairwise judgement 做完整性与稳定性分析"
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path}:{line_number} JSON损坏：{error}"
                ) from error
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number} 不是JSON对象")
            rows.append(row)
    return rows


def mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def summarize_rows(
    rows: list[dict],
    baseline_name: str,
    candidate_name: str,
    include_categories: bool = True,
) -> dict:
    winner_counts = Counter(row["winner_model"] for row in rows)
    baseline_scores = [row["baseline_scores"]["overall"] for row in rows]
    candidate_scores = [row["candidate_scores"]["overall"] for row in rows]
    baseline_pass = sum(bool(row["baseline_category_pass"]) for row in rows)
    candidate_pass = sum(bool(row["candidate_category_pass"]) for row in rows)
    decided = winner_counts[baseline_name] + winner_counts[candidate_name]
    categories: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)

    summary = {
        "total": len(rows),
        "winner_counts": dict(winner_counts),
        "baseline_category_pass": baseline_pass,
        "candidate_category_pass": candidate_pass,
        "baseline_avg_overall": mean_or_zero(baseline_scores),
        "candidate_avg_overall": mean_or_zero(candidate_scores),
        "overall_delta_candidate_minus_baseline": (
            mean_or_zero(candidate_scores) - mean_or_zero(baseline_scores)
        ),
        "average_confidence": mean_or_zero(
            [float(row["confidence"]) for row in rows]
        ),
        "candidate_win_rate_all": (
            winner_counts[candidate_name] / len(rows) if rows else 0.0
        ),
        "candidate_win_rate_non_tie": (
            winner_counts[candidate_name] / decided if decided else 0.0
        ),
    }
    if include_categories:
        summary["categories"] = {
            category: summarize_rows(
                category_rows,
                baseline_name,
                candidate_name,
                include_categories=False,
            )
            for category, category_rows in sorted(categories.items())
        }
    return summary


def validate_replicates(
    replicate_rows: list[list[dict]],
    paths: list[Path],
) -> dict:
    id_sets = []
    duplicate_ids = {}
    for path, rows in zip(paths, replicate_rows):
        ids = [row.get("id") for row in rows]
        counts = Counter(ids)
        duplicates = sorted(
            str(sample_id)
            for sample_id, count in counts.items()
            if count != 1
        )
        duplicate_ids[str(path)] = duplicates
        id_sets.append(set(ids))

    same_id_set = all(current == id_sets[0] for current in id_sets[1:])
    row_counts = [len(rows) for rows in replicate_rows]
    return {
        "replicate_count": len(replicate_rows),
        "rows_per_replicate": row_counts,
        "expected_rows_per_replicate": 100,
        "complete_100_per_replicate": all(count == 100 for count in row_counts),
        "unique_ids": all(not values for values in duplicate_ids.values()),
        "duplicate_ids": duplicate_ids,
        "same_id_set_across_replicates": same_id_set,
        "id_intersection_count": len(set.intersection(*id_sets)) if id_sets else 0,
        "id_union_count": len(set.union(*id_sets)) if id_sets else 0,
    }


def winner_agreement(
    first: list[dict],
    second: list[dict],
) -> dict:
    first_by_id = {row["id"]: row for row in first}
    second_by_id = {row["id"]: row for row in second}
    shared_ids = sorted(set(first_by_id) & set(second_by_id))
    matches = sum(
        first_by_id[sample_id]["winner_model"]
        == second_by_id[sample_id]["winner_model"]
        for sample_id in shared_ids
    )
    transitions = Counter(
        (
            first_by_id[sample_id]["winner_model"],
            second_by_id[sample_id]["winner_model"],
        )
        for sample_id in shared_ids
    )
    return {
        "shared_count": len(shared_ids),
        "exact_winner_agreement_count": matches,
        "exact_winner_agreement_rate": matches / len(shared_ids)
        if shared_ids
        else 0.0,
        "winner_transitions": {
            f"{before}->{after}": count
            for (before, after), count in sorted(transitions.items())
        },
    }


def markdown_report(result: dict, paths: list[Path]) -> str:
    checks = result["quality_checks"]
    lines = [
        "# SimPO full Gemini 独立复核",
        "",
        "本报告比较同一冻结测试集、同一生成结果在两个独立随机匿名顺序（seed=42/43）下的 Gemini pairwise judgment。",
        "原始 judge JSONL 仅保留在远端实验目录，本报告只保留汇总统计。",
        "",
        "## 数据质量检查",
        "",
        f"- 输入：{', '.join(str(path) for path in paths)}",
        f"- 每轮样本数：{checks['rows_per_replicate']}",
        f"- 每轮 100 条：{checks['complete_100_per_replicate']}",
        f"- ID 唯一：{checks['unique_ids']}",
        f"- 两轮 ID 集一致：{checks['same_id_set_across_replicates']}",
        f"- 共享 ID：{checks['id_intersection_count']}，并集 ID：{checks['id_union_count']}",
        "",
        "## 逐轮结果",
        "",
        "| seed | tie | align_sft_v2 胜 | simpo_v1_full 胜 | baseline overall | candidate overall | candidate-baseline | 非平局候选胜率 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for replicate in result["replicates"]:
        counts = replicate["winner_counts"]
        lines.append(
            "| {seed} | {tie} | {baseline} | {candidate} | {base:.2f} | {cand:.2f} | {delta:+.2f} | {rate:.3f} |".format(
                seed=replicate["seed"],
                tie=counts.get("tie", 0),
                baseline=counts.get("align_sft_v2", 0),
                candidate=counts.get("simpo_v1_full", 0),
                base=replicate["baseline_avg_overall"],
                cand=replicate["candidate_avg_overall"],
                delta=replicate["overall_delta_candidate_minus_baseline"],
                rate=replicate["candidate_win_rate_non_tie"],
            )
        )

    agreement = result.get("winner_agreement", {})
    lines.extend([
        "",
        "## 合并描述性统计",
        "",
        f"- 评审总数：{result['aggregate']['total']}",
        f"- 胜负计数：{result['aggregate']['winner_counts']}",
        f"- 平均 overall：align_sft_v2={result['aggregate']['baseline_avg_overall']:.3f}，simpo_v1_full={result['aggregate']['candidate_avg_overall']:.3f}",
        f"- 平均 overall 差值（candidate-baseline）：{result['aggregate']['overall_delta_candidate_minus_baseline']:+.3f}",
        f"- 非平局候选胜率：{result['aggregate']['candidate_win_rate_non_tie']:.3f}",
        f"- 两轮逐样本胜者完全一致：{agreement.get('exact_winner_agreement_count', 0)}/{agreement.get('shared_count', 0)}（{agreement.get('exact_winner_agreement_rate', 0.0):.3f}）",
        "",
        "## 解释边界",
        "",
        "两轮均使用同一冻结测试集和同一生成结果，因此它们检验的是评审顺序/评审随机性的稳定性，而不是新的模型泛化测试。",
        "如果两轮都显示 baseline 胜出更多且 candidate overall 更低，可以增强“SimPO full 过优化”的证据；仍不能替代公开 benchmark、人工专家复核或长程 RL 收敛实验。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in args.inputs]
    replicate_rows = [load_jsonl(path) for path in paths]
    quality_checks = validate_replicates(replicate_rows, paths)
    if not quality_checks["complete_100_per_replicate"]:
        raise RuntimeError("至少一轮不是完整100条，拒绝生成稳健性结论")
    if not quality_checks["unique_ids"]:
        raise RuntimeError("发现重复ID，拒绝生成稳健性结论")
    if not quality_checks["same_id_set_across_replicates"]:
        raise RuntimeError("各轮ID集合不一致，拒绝生成稳健性结论")

    replicate_summaries = [
        summarize_rows(rows, args.baseline_name, args.candidate_name)
        for rows in replicate_rows
    ]
    aggregate_rows = [row for rows in replicate_rows for row in rows]
    result = {
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "replicates": [
            {"seed": seed, **summary}
            for seed, summary in zip((42, 43), replicate_summaries)
        ],
        "aggregate": summarize_rows(
            aggregate_rows,
            args.baseline_name,
            args.candidate_name,
        ),
        "quality_checks": quality_checks,
        "winner_agreement": winner_agreement(
            replicate_rows[0],
            replicate_rows[1],
        ) if len(replicate_rows) == 2 else {},
    }
    output_path = Path(args.output)
    markdown_path = Path(args.markdown)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        markdown_report(result, paths),
        encoding="utf-8",
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False))
    print(json.dumps(result["quality_checks"], ensure_ascii=False))


if __name__ == "__main__":
    main()

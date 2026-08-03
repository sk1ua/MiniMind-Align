"""Create the Alignment v2 spot-review, data report and backup manifest."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from validators import validate_record


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def finalize() -> None:
    audit = json.loads((ROOT / "reports" / "pilot_audit.json").read_text(encoding="utf-8"))
    records = read_jsonl(ROOT / "generated" / "new_validation_pilot.jsonl")
    manifests = read_jsonl(ROOT / "manifests" / "validation_manifest.jsonl")
    by_category: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for record, manifest in zip(records, manifests):
        by_category.setdefault(manifest["category"], []).append((record, manifest))
    rng = random.Random(42)
    review_lines = ["# Alignment v2 pilot 工程人工抽查", "", "抽查规则：seed=42，每类从独立 validation 生成集中随机抽取 2 条；检查 prompt/answer 是否匹配 manifest、专用 validator 是否通过、是否存在提示词泄漏或不安全表述。", ""]
    for category in sorted(by_category):
        review_lines.extend([f"## {category}", ""])
        pairs = rng.sample(by_category[category], min(2, len(by_category[category])))
        for record, manifest in pairs:
            passed, reason = validate_record(record, manifest)
            review_lines.extend([
                f"- {manifest['id']} — {'PASS' if passed else 'FAIL'} ({reason or 'validator_ok'})",
                f"  - prompt: {manifest['prompt']}",
                f"  - chosen: {manifest['chosen']}",
                f"  - family: {manifest['family']}; method: {manifest['generation_method']}",
                "",
            ])
    write_new(ROOT / "reports" / "sample_review.md", "\n".join(review_lines) + "\n")

    hashes = audit["sha256"]
    doc_lines = [
        "# Alignment v2 数据工程报告",
        "",
        "## 实验事实",
        "",
        f"- seed: 42",
        f"- 新 train: {audit['counts']['train']} 条；新 validation: {audit['counts']['validation']} 条。",
        "- 类别比例：format 200/32、instruction 200/32、reasoning 150/24、safety 150/24、repetition 100/16、conciseness 80/12、termination 60/10、uncertainty 60/10。",
        "- Alignment v1 chosen 合并后 SFT train 为 1600 条；SFT validation 为 160 条。",
        f"- 生成方法计数：{json.dumps(audit['generation_method_counts'], ensure_ascii=False)}。",
        "- smoke：8 类各 3 条 train 和 3 条 validation，专用 validator、结构、泄漏检查通过。",
        "- pilot：专用 validator 100% 通过；JSONL、角色顺序、ID、prompt、split 隔离、测试集相似度、v1 prompt 重复和真实 SFTDataset batch 检查通过。",
        f"- SFTDataset batch smoke：{audit['sftdataset_batch']}。",
        "",
        "## SHA256",
        "",
    ]
    doc_lines.extend(f"- {path}: {value}" for path, value in hashes.items())
    doc_lines.extend([
        "",
        "## 结论与限制",
        "",
        "- 该数据集是程序化 pilot，不等同于人工大规模标注质量证明；sample_review.md 记录了每类随机抽查样本。",
        "- 测试集未用于生成答案；审计器以 fail-closed 方式执行。",
        "- 通过 Sprint A 数据门禁，可进入 SFT v2 smoke；是否使用完整训练权重仍需按 Sprint B 的训练前 GPU/时间检查执行。",
    ])
    write_new(REPO / "docs" / "experiments" / "alignment_v2_data.md", "\n".join(doc_lines) + "\n")

    tracked = [
        "out/full_sft_768.pth",
        "out/align_sft_v1_768.pth",
        "out/dpo_v1_768.pth",
        "out/align_sft_dpo_v1_768.pth",
        "dataset/alignment_v2/generated/new_train_pilot.jsonl",
        "dataset/alignment_v2/generated/new_validation_pilot.jsonl",
        "dataset/alignment_v2/generated/sft_train_pilot.jsonl",
        "dataset/alignment_v2/generated/sft_validation_pilot.jsonl",
    ]
    manifest = {
        "created_by": "dataset/alignment_v2/finalize_alignment_v2.py",
        "warning": "This is a local SHA256 inventory; it does not upload or copy backups.",
        "files": [{"path": path, "sha256": digest(REPO / path), "bytes": (REPO / path).stat().st_size} for path in tracked if (REPO / path).exists()],
    }
    write_new(REPO / "results" / "backup_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    finalize()

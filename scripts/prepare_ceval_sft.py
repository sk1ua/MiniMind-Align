"""Convert C-Eval exam questions to SFT-format JSONL for MiniMind training.

Downloads ceval/ceval-exam from Hugging Face and converts each question into
a single-turn instruction-following conversation in the format expected by
SFTDataset in dataset/lm_dataset.py.

Output format per line:
    {"conversations": [{"role": "user", "content": "<question>"}, {"role": "assistant", "content": "<answer_letter>. <explanation>"}]}

Usage (from _public_release_local/):
    python scripts/prepare_ceval_sft.py --output dataset/ceval_qa.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUBJECTS = [
    "high_school_chinese",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_chemistry",
    "high_school_biology",
    "computer_network",
    "operating_system",
    "business_ethics",
    "college_economics",
    "logic",
]

DATASET_SUBJECT_ALIASES = {
    "business_ethics": "business_administration",
}

CHOICES = ("A", "B", "C", "D")

PROMPT_TEMPLATE = (
    "{question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}"
)

SPLITS = ["dev", "val"]


def row_to_conversation(row: dict) -> dict:
    """Convert one C-Eval row to a SFT conversation dict."""
    user_content = PROMPT_TEMPLATE.format(
        question=row["question"],
        A=row["A"],
        B=row["B"],
        C=row["C"],
        D=row["D"],
    )
    answer_letter = row["answer"]
    # Use the answer letter and the option text as the assistant reply
    answer_text = row.get(answer_letter, "")
    assistant_content = f"{answer_letter}. {answer_text}"

    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert C-Eval to SFT JSONL")
    parser.add_argument(
        "--output", type=Path, default=Path("dataset/ceval_qa.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--subjects", nargs="+", default=SUBJECTS,
        help="C-Eval subject names to include",
    )
    args = parser.parse_args()

    from datasets import load_dataset  # lazy import

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    subject_counts: dict[str, int] = {}

    with args.output.open("w", encoding="utf-8") as fout:
        for subject in args.subjects:
            dataset_subject = DATASET_SUBJECT_ALIASES.get(subject, subject)
            count = 0
            for split in SPLITS:
                try:
                    ds = load_dataset(
                        "ceval/ceval-exam", dataset_subject,
                        split=split, trust_remote_code=True,
                    )
                except Exception as exc:
                    print(f"  [{subject}/{split}] skip: {exc}")
                    continue

                for row in ds:
                    # Skip rows without a valid answer (some dev splits lack answers)
                    if not row.get("answer") or row["answer"] not in CHOICES:
                        continue
                    conv = row_to_conversation(row)
                    fout.write(json.dumps(conv, ensure_ascii=False) + "\n")
                    count += 1

            subject_counts[subject] = count
            total += count
            print(f"  {subject}: {count} samples")

    print(f"\n✅ 共写入 {total} 条样本 → {args.output}")
    for s, c in subject_counts.items():
        print(f"   {s}: {c}")


if __name__ == "__main__":
    main()

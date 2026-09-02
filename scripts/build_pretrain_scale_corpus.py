"""Build the small pretraining corpus used by the scale-comparison experiment.

Concatenates alignment_v2 SFT train conversations and the C-Eval SFT QA data
into one {"text": ...} JSONL corpus (1,892 documents).

Usage (from _public_release_local/):
    python scripts/build_pretrain_scale_corpus.py --output dataset/pretrain_scale_corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCES = [
    "dataset/alignment_v2/generated/sft_train_pilot.jsonl",
    "dataset/ceval_qa.jsonl",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pretrain scale-corpus JSONL")
    parser.add_argument("--output", type=Path, default=Path("dataset/pretrain_scale_corpus.jsonl"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for src in SOURCES:
            with open(src, encoding="utf-8") as fin:
                for line in fin:
                    obj = json.loads(line)
                    text = "".join(turn["content"] for turn in obj["conversations"]).strip()
                    if len(text) < 20:
                        continue
                    fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    count += 1
    print(f"wrote {count} documents -> {args.output}")


if __name__ == "__main__":
    main()

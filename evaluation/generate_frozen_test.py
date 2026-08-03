"""Generate all frozen Alignment v1 test prompts with identical decoding."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.model_lora import apply_lora, load_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def repeat_ratio(token_ids: list[int], n: int = 3) -> float:
    grams = [tuple(token_ids[index : index + n]) for index in range(max(0, len(token_ids) - n + 1))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def load_model(args: argparse.Namespace, device: torch.device, tokenizer: AutoTokenizer):
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    model = MiniMindForCausalLM(config)
    base_name = args.base_weight if args.adapter_path else args.weight
    state = torch.load(Path(args.model_dir) / f"{base_name}_768.pth", map_location=device)
    model.load_state_dict(state, strict=True)
    if args.adapter_path:
        apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, target_modules=("q_proj", "v_proj"))
        load_lora(model, args.adapter_path)
    return model.half().to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, default=Path("dataset/alignment_v1/splits/prompts_test.jsonl"))
    parser.add_argument("--model-dir", default="out")
    parser.add_argument("--tokenizer-path", default="model")
    parser.add_argument("--base-weight", default="full_sft")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(args, device, tokenizer)
    prompts = [json.loads(line) for line in args.test_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    decode_config = {"do_sample": False, "max_new_tokens": args.max_new_tokens, "repetition_penalty": args.repetition_penalty, "no_repeat_ngram_size": args.no_repeat_ngram_size}
    rows: list[dict] = []
    with torch.inference_mode():
        for index, sample in enumerate(prompts, 1):
            messages = [{"role": "user", "content": sample["prompt"]}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=False)
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True).to(device)
            start = time.perf_counter()
            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            elapsed = time.perf_counter() - start
            prompt_len = inputs["input_ids"].shape[1]
            new_ids = generated[0][prompt_len:].tolist()
            response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            rows.append({
                "id": sample["id"],
                "category": sample["category"],
                "family": sample.get("family"),
                "prompt": sample["prompt"],
                "response": response,
                "generated_tokens": len(new_ids),
                "tokens_per_second": round(len(new_ids) / max(elapsed, 1e-6), 2),
                "repeat_3gram_ratio": round(repeat_ratio(new_ids), 6),
                "finished_naturally": bool(len(new_ids) < args.max_new_tokens),
                "model": args.model_name,
                "decode_config": decode_config,
            })
            if index % 10 == 0 or index == len(prompts):
                print(f"{args.model_name}: {index}/{len(prompts)}")
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"FROZEN_TEST_GENERATION_OK model={args.model_name} count={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()

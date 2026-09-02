"""Parse scale-experiment training logs, plot loss curves, and sample generations.

Run from _public_release_local/:
    python scripts/analyze_scale_experiment.py
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

LOG_DIR = ROOT / "results" / "scale_comparison"
CKPT_DIR = ROOT / "checkpoints"

RUNS = [
    # (label, log file, log_interval, hidden, layers, seq_len)
    ("S (512/4层, ~19M)",     "log_pretrain_s.txt",       20, 512, 4, 256),
    ("M (768/8层, ~64M)",     "log_pretrain_m.txt",       20, 768, 8, 256),
    ("L (1024/12层, ~150M)",  "log_pretrain_l.txt",       20, 1024, 12, 256),
    ("M-seq512 (768/8层)",    "log_pretrain_m_seq512.txt", 40, 768, 8, 512),
]

LINE_RE = re.compile(r"Epoch:\[(\d+)/\d+\]\((\d+)/(\d+)\), loss: ([\d.]+)")


def moving_avg(xs, ys, window=5):
    out_x, out_y = [], []
    for i in range(len(ys)):
        lo = max(0, i - window + 1)
        seg = ys[lo:i + 1]
        out_y.append(sum(seg) / len(seg))
        out_x.append(xs[i])
    return out_x, out_y


def parse_logs():
    curves = {}
    for label, log_file, interval, *_ in RUNS:
        steps, losses = [], []
        lines = (LOG_DIR / log_file).read_text(encoding="utf-8").splitlines()
        for line in lines:
            m = LINE_RE.search(line)
            if m:
                epoch, inner_step, per_epoch, loss = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
                steps.append((epoch - 1) * per_epoch + inner_step)
                losses.append(loss)
        sx, sy = moving_avg(steps, losses, window=5)
        curves[label] = (sx, sy)
        print(f"{label}: {len(steps)} log points, first={losses[0]:.3f}, last5avg={sum(losses[-5:])/5:.3f}")
    return curves


def plot(curves):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, (xs, ys) in curves.items():
        ax.plot(xs, ys, label=label, linewidth=1.6)
    ax.set_xlabel("训练步数（steps）")
    ax.set_ylabel("训练 Loss（5 点滑动平均）")
    ax.set_title("不同模型规模/上下文长度的预训练 Loss 收敛曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = LOG_DIR / "loss_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


@torch.no_grad()
def sample_generations():
    tokenizer = AutoTokenizer.from_pretrained("model")
    prompts = ["中国的首都是", "机器学习是", "学习深度学习需要"]
    rows = []
    for label, _, _, hidden, layers, _seq in RUNS:
        cfg = MiniMindConfig(hidden_size=hidden, num_hidden_layers=layers)
        model = MiniMindForCausalLM(cfg)
        if "seq512" in label:
            name = "pretrain_m_seq512_768.pth"
        else:
            name = {
                "S": "pretrain_s_512.pth",
                "M": "pretrain_m_768.pth",
                "L": "pretrain_l_1024.pth",
            }[label.split()[0]]
        ckp = CKPT_DIR / name
        state = torch.load(str(ckp), map_location="cpu")
        model.load_state_dict(state, strict=True)
        model = model.float().eval()
        gens = []
        for p in prompts:
            ids = tokenizer(tokenizer.bos_token + p, return_tensors="pt").input_ids
            out = model.generate(ids, max_new_tokens=40, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            gens.append(text.strip().replace("\n", " ")[:60])
        rows.append((label, gens))
        print(f"\n=== {label} ===")
        for p, g in zip(prompts, gens):
            print(f"  {p} → {g[:60]}")
        del model

    md = ["| 模型 | " + " | ".join(prompts) + " |",
          "|---|" + "---|" * len(prompts)]
    for label, gens in rows:
        md.append("| " + label + " | " + " | ".join(g.replace("|", "｜") for g in gens) + " |")
    (LOG_DIR / "generation_samples.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nsaved {LOG_DIR / 'generation_samples.md'}")


if __name__ == "__main__":
    curves = parse_logs()
    plot(curves)
    sample_generations()

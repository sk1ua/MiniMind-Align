"""Build the Sprint F evidence table and static QA charts.

The script reads only archived JSON summaries and experiment wrapper logs.  It
does not rerun training or infer missing metrics.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_ROOT = ROOT / "results/experiments/unified_sft_v2_20260731/validator"
OUTPUT_ROOT = ROOT / "results/experiments/unified_sft_v2_20260731"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def duration_seconds(experiment_id: str) -> float | None:
    start = ROOT / "results/experiments" / experiment_id / "start_time.txt"
    end = ROOT / "results/experiments" / experiment_id / "end_time.txt"
    if not start.exists() or not end.exists():
        return None
    try:
        first = datetime.fromisoformat(start.read_text(encoding="utf-8").strip().replace("Z", "+00:00"))
        last = datetime.fromisoformat(end.read_text(encoding="utf-8").strip().replace("Z", "+00:00"))
        return (last - first).total_seconds()
    except ValueError:
        return None


def load_rl_logs(experiment_id: str) -> list[dict]:
    path = ROOT / "results/experiments" / experiment_id / "stdout.log"
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in row and "reward_mean" in row:
            rows.append(row)
    return rows


def build_metrics() -> dict:
    validator_names = {
        "full_sft": "full_sft",
        "align_sft_v1": "align_sft_v1",
        "align_sft_v2": "align_sft_v2",
        "lora_align_sft_v2": "lora_align_sft_v2",
        "dpo_v2_full": "dpo_v2_full",
        "simpo_v1_pilot": "simpo_v1",
        "simpo_v1_full": "simpo_v1_full",
    }
    frozen = {}
    for label, directory in validator_names.items():
        frozen[label] = read_json(VALIDATOR_ROOT / directory / "validator_summary.json")

    loss_files = {
        "full_sft": "full_sft_v2val_eval_20260731/validation_loss.json",
        "align_sft_v1": "align_sft_v1_v2val_eval_20260731/validation_loss.json",
        "align_sft_v2": "align_sft_v2_eval_20260731/validation_loss.json",
        "lora_align_sft_v2": "lora_align_sft_v2_eval_20260731/validation_loss.json",
    }
    validation_loss = {label: read_json(ROOT / "results/experiments" / path) for label, path in loss_files.items()}
    rl_logs = {
        "grpo": load_rl_logs("grpo_lite_pilot_20260731"),
        "cispo": load_rl_logs("cispo_lite_pilot_20260731"),
    }
    gemini_summary_files = {
        "dpo_v2_full": "gemini_align_vs_dpo_v2_full_retry_20260731/summary.json",
        "simpo_v1_pilot": "gemini_align_vs_simpo_v1_pilot_20260731/summary.json",
        "simpo_v1_full": "gemini_align_vs_simpo_v1_full_20260731/summary.json",
    }
    gemini_pairwise = {}
    for label, relative_path in gemini_summary_files.items():
        summary_path = ROOT / "results/experiments" / relative_path
        if summary_path.exists():
            gemini_pairwise[label] = read_json(summary_path)
    durations = {
        experiment: duration_seconds(experiment)
        for experiment in [
            "align_sft_v2_pilot_20260731",
            "dpo_v2_full_retry_20260731",
            "simpo_v1_pilot_20260731",
            "simpo_v1_full_20260731",
            "reward_model_v1_pilot_20260731",
            "grpo_lite_pilot_20260731",
            "cispo_lite_pilot_20260731",
        ]
    }
    return {
        "metadata": {
            "project": "MiniMind-Align",
            "frozen_test_count": 100,
            "frozen_decode": {"seed": 42, "do_sample": False, "max_new_tokens": 160, "repetition_penalty": 1.15, "no_repeat_ngram_size": 3},
            "source_note": "All values are read from archived experiment summaries; missing values remain missing.",
        },
        "frozen_validator": frozen,
        "validation_loss": validation_loss,
        "rl_lite": rl_logs,
        "gemini_pairwise": gemini_pairwise,
        "wall_time_seconds": durations,
        "billing": {"usd_measured": None, "note": "Local experiment wrapper has no Cloud Billing export. Report wall time and GPU context, not fabricated dollars."},
    }


def render_charts(metrics: dict, output_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    palette = {"blue": "#245B8A", "gold": "#C58A2A", "orange": "#C65D2E", "olive": "#687A3A", "pink": "#A54E67", "ink": "#25313B", "grid": "#D9DEE3"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "normal", "axes.edgecolor": palette["ink"], "axes.labelcolor": palette["ink"], "xtick.color": palette["ink"], "ytick.color": palette["ink"]})
    paths: dict[str, str] = {}

    labels = list(metrics["frozen_validator"])
    pass_rate = [metrics["frozen_validator"][label]["validator_pass"] for label in labels]
    natural_rate = [metrics["frozen_validator"][label]["natural_end"] for label in labels]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, pass_rate, width, label="Validator pass / 100", color=palette["blue"])
    ax.bar(x + width / 2, natural_rate, width, label="Natural end / 100", color=palette["gold"])
    ax.set_title("Frozen test alignment outcomes")
    ax.set_ylabel("Count out of 100")
    ax.set_ylim(0, 100)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.grid(axis="y", color=palette["grid"], linewidth=0.8)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = output_dir / "frozen_validator_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["frozen_validator_comparison"] = str(path)

    category_models = ["align_sft_v2", "dpo_v2_full", "simpo_v1_pilot"]
    category_labels = list(metrics["frozen_validator"]["align_sft_v2"]["categories"])
    fig, ax = plt.subplots(figsize=(10, 6))
    height = 0.23
    y = np.arange(len(category_labels))
    colors = [palette["blue"], palette["orange"], palette["olive"]]
    for index, (model, color) in enumerate(zip(category_models, colors)):
        source = metrics["frozen_validator"][model]
        values = [100 * source["categories"][category]["validator_pass"] / source["categories"][category]["count"] for category in category_labels]
        ax.barh(y + (index - 1) * height, values, height, label=model, color=color)
    ax.set_title("Frozen test validator pass by category")
    ax.set_xlabel("Pass rate (%)")
    ax.set_xlim(0, 100)
    ax.set_yticks(y, category_labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color=palette["grid"], linewidth=0.8)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    path = output_dir / "category_validator_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["category_validator_comparison"] = str(path)

    gemini_models = list(metrics["gemini_pairwise"])
    if gemini_models:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = np.arange(len(gemini_models))
        baseline_wins = [metrics["gemini_pairwise"][label]["winner_counts"].get("align_sft_v2", 0) for label in gemini_models]
        candidate_wins = [metrics["gemini_pairwise"][label]["winner_counts"].get(label, 0) for label in gemini_models]
        ties = [metrics["gemini_pairwise"][label]["winner_counts"].get("tie", 0) for label in gemini_models]
        ax.bar(x, baseline_wins, label="align_sft_v2 wins", color=palette["blue"])
        ax.bar(x, candidate_wins, bottom=baseline_wins, label="candidate wins", color=palette["orange"])
        bottoms = [left + right for left, right in zip(baseline_wins, candidate_wins)]
        ax.bar(x, ties, bottom=bottoms, label="tie", color=palette["gold"])
        ax.set_title("Gemini overall pairwise outcomes (100 samples per comparison)")
        ax.set_ylabel("Count out of 100")
        ax.set_ylim(0, 100)
        ax.set_xticks(x, gemini_models)
        ax.grid(axis="y", color=palette["grid"], linewidth=0.8)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        path = output_dir / "gemini_overall_comparison.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths["gemini_overall_comparison"] = str(path)

    loss_labels = list(metrics["validation_loss"])
    losses = [metrics["validation_loss"][label]["loss"] for label in loss_labels]
    ppls = [metrics["validation_loss"][label]["perplexity"] for label in loss_labels]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(loss_labels, losses, color=palette["blue"])
    axes[0].set_title("Validation loss")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_ylim(0, max(losses) * 1.15)
    axes[1].bar(loss_labels, ppls, color=palette["pink"])
    axes[1].set_title("Validation perplexity")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_ylim(0, max(ppls) * 1.15)
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", color=palette["grid"], linewidth=0.8)
    fig.suptitle("Alignment tax proxy on the independent v2 validation set", y=1.02)
    fig.tight_layout()
    path = output_dir / "validation_loss_ppl.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["validation_loss_ppl"] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for mode, color, style in [("grpo", palette["blue"], "-"), ("cispo", palette["orange"], "--")]:
        rows = metrics["rl_lite"][mode]
        axes[0].plot([row["step"] for row in rows], [row["reward_mean"] for row in rows], marker="o", color=color, linestyle=style, label=mode)
        axes[1].plot([row["step"] for row in rows], [row["kl_mean"] for row in rows], marker="o", color=color, linestyle=style, label=mode)
    axes[0].set_title("Rule reward mean (4-step pilot)")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Reward")
    axes[0].axhline(0, color=palette["ink"], linewidth=0.8)
    axes[1].set_title("Reference KL mean (4-step pilot)")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("KL estimate")
    for axis in axes:
        axis.grid(color=palette["grid"], linewidth=0.8)
        axis.legend(frameon=False)
    fig.suptitle("GRPO/CISPO lite stability signals", y=1.02)
    fig.tight_layout()
    path = output_dir / "rl_reward_kl.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["rl_reward_kl"] = str(path)

    duration_items = [(key, value) for key, value in metrics["wall_time_seconds"].items() if value is not None]
    fig, ax = plt.subplots(figsize=(9, 5))
    duration_items.sort(key=lambda item: item[1])
    ax.barh([item[0] for item in duration_items], [item[1] for item in duration_items], color=palette["olive"])
    ax.set_title("Experiment wall time recorded by the wrapper")
    ax.set_xlabel("Seconds; not a billing amount")
    ax.grid(axis="x", color=palette["grid"], linewidth=0.8)
    fig.tight_layout()
    path = output_dir / "experiment_wall_time.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["experiment_wall_time"] = str(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    metrics = build_metrics()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = args.output_dir / "plots"
    chart_paths = render_charts(metrics, chart_dir)
    metrics["charts"] = chart_paths
    (args.output_dir / "unified_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "chart_map.json").write_text(json.dumps({
        "frozen_validator_comparison": {"question": "Which frozen models pass the same rule set?", "family": "comparison", "source": "validator_summary.json", "caveat": "100 internal test prompts"},
        "category_validator_comparison": {"question": "Where do alignment methods differ by category?", "family": "comparison", "source": "validator_summary.json", "caveat": "category denominators vary"},
        "validation_loss_ppl": {"question": "What is the validation-loss alignment-tax proxy?", "family": "comparison", "source": "validation_loss.json", "caveat": "v2 validation only"},
        "rl_reward_kl": {"question": "Do the tiny RL pilots show reward/KL instability?", "family": "ordered progression", "source": "stdout.log", "caveat": "4 optimizer steps; not convergence"},
        "experiment_wall_time": {"question": "How much wrapper-recorded wall time did new experiments use?", "family": "comparison", "source": "start_time.txt/end_time.txt", "caveat": "not Cloud Billing USD"},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "metrics": str(args.output_dir / "unified_metrics.json"), "charts": chart_paths}, ensure_ascii=False))


if __name__ == "__main__":
    main()

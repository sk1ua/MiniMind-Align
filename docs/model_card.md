# MiniMind-Align model and experiment card

## Intended use

This repository is intended for alignment research, experiment auditing, and educational reproduction. The released snapshot is not a production model distribution.

## Training and evaluation

The study compares a baseline MiniMind alignment model with error-driven SFT, DPO, SimPO, and a corrected-GRPO diagnostic. Evaluation uses independent validation and a fixed 32-row release slice with validator replay and checkpoint reload checks.

## Findings

Supervised and preference-based repair improved the measured quality slice. The single-seed corrected-GRPO diagnostic did not add validation passes over the SFT source. No default model replacement was performed.

## Limitations

See [docs/limitations.md](limitations.md). The public repository excludes raw data, weights, and complete generation traces.

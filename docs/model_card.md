# MiniMind-Align model and experiment card

This is the model card for the MiniMind-Align research snapshot, not the upstream MiniMind project. The base implementation is derived from upstream commit `307fd76`; attribution and license details are recorded in [upstream.md](upstream.md).

## Intended use

This repository is intended for alignment research, experiment auditing, and educational reproduction. The released snapshot is not a production model distribution.

## Training and evaluation

The study compares a baseline MiniMind alignment model with error-driven SFT, DPO, SimPO, and a corrected-GRPO diagnostic. Evaluation uses independent validation and a fixed 32-row release slice with validator replay and checkpoint reload checks.

## Findings

Supervised and preference-based repair improved the measured quality slice. The single-seed corrected-GRPO diagnostic did not add validation passes over the SFT source. No default model replacement was performed.

## Limitations

See [docs/limitations.md](limitations.md). The public repository excludes raw data, weights, and complete generation traces.

# Upstream MiniMind attribution

MiniMind-Align is an independent alignment research project built on top of the MiniMind model and training code.

## Upstream source

- Repository: [jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- Source snapshot: commit `307fd76`
- License: Apache-2.0, retained from the upstream project

The upstream project provides the base model implementation, tokenizer/model utilities, and general training infrastructure. This repository does not claim ownership of those upstream components.

## MiniMind-Align contribution

The project-specific contribution is the auditable alignment workflow around that base:

- Alignment v2 data contracts and validator/reward replay;
- error-driven SFT, DPO, SimPO, and corrected-GRPO experiment entry points;
- validation-slice design, checkpoint and precision-contract audits;
- RL telemetry, failure analysis, reproducibility checks, and compact public results.

The root README and experiment reports describe MiniMind-Align research. They are not copies of the upstream project README. Model weights, raw training data, cloud configuration, and complete generation traces are intentionally excluded from this public snapshot.

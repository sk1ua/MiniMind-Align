# Public release limitations

- The public snapshot contains code, tests, compact metrics, and artifact hashes.
- Model weights are excluded from Git history and are distributed separately as curated v0.1.0 Release assets; native Alignment v2 records, full generation logs, and cloud credentials remain excluded.
- Corrected-GRPO is a single-seed diagnostic on a 32-row release slice; it is not evidence of RL model improvement.
- Error-driven SFT, DPO, and SimPO candidates remain diagnostic artifacts and do not replace the default model.
- Full reproduction of training requires separately obtained, licensed data and model weights.

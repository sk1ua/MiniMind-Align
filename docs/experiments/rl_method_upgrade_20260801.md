# MM-E009：RL 三 seed validation checkpoint selection

## Protocol

- Train manifest: `results/inputs/on_policy_train_manifest_128_20260731.jsonl`，128 prompts。
- Validation manifest: `results/inputs/on_policy_validation_manifest_32_20260731.jsonl`，32 prompts，训练不读取验证 prompt。
- GRPO/CISPO each use seeds `42, 43, 44`、`num_generations=8`、`accumulation_steps=8`、最多 16 optimizer steps。
- Validation/checkpoint interval is 4 steps. KL gate is `>0.005` for two consecutive steps; safety/termination gate is a drop of more than 10 percentage points from the baseline.
- Checkpoint selection: validator pass, then safety/termination, natural end, lower repeat-3gram. If no checkpoint passes, baseline is retained.

The requested train manifest is a legacy Alignment v1 manifest with empty metadata. The original file was not changed; each run stores `resolved_train_manifest.jsonl` with conservative metadata derived from the legacy prompt/chosen constraints. Selection and quality gates use the native-metadata validation manifest.

## Results

| method/seed | baseline validator | selected step | selected validator | safety | termination | natural end | repeat-3gram |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO 42 | 9/32 | 4 | 9/32 | 3/4 | 1/4 | 32/32 | 0.032463 |
| GRPO 43 | 9/32 | 4 | 9/32 | 3/4 | 1/4 | 32/32 | 0.032463 |
| GRPO 44 | 9/32 | 16 | 10/32 | 3/4 | 1/4 | 32/32 | 0.032897 |
| CISPO 42 | 9/32 | 16 | 10/32 | 3/4 | 1/4 | 32/32 | 0.032690 |
| CISPO 43 | 9/32 | 16 | 10/32 | 3/4 | 1/4 | 32/32 | 0.032690 |
| CISPO 44 | 9/32 | 12 | 10/32 | 3/4 | 1/4 | 32/32 | 0.032897 |

GRPO mean is `9.33/32` (population std `0.47`), a `+0.33` pass delta. CISPO mean is `10.00/32` (std `0.00`), a `+1.00` pass delta. Both are below the registered `+3 pass` improvement threshold; safety and termination do not drop, so this is an inconclusive/negative result, not an improvement claim. Existing default weights remain unchanged.

All six runs exited 0, produced step summaries, validation history, checkpoints, selection metadata and non-overwriting selected weights under `results/experiments/rl_method_upgrade_20260801/`.

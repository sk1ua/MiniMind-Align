# MiniMind-Align release weights

The model files are distributed as GitHub Release assets rather than Git
history. The release contains independently reloadable 768-dimension state
dicts from the public research snapshot.

Release page: <https://github.com/sk1ua/MiniMind-Align/releases/tag/v0.1.0>

| Asset | Role | Evidence scope |
| --- | --- | --- |
| `minimind-align-baseline-768.pth` | `align_sft_v2_pilot` reference baseline | 48/160 full validation, 13/32 release slice |
| `minimind-align-error-driven-sft-seed42-768.pth` | Best diagnostic quality-repair candidate | 79/160, 19/32 |
| `minimind-align-error-driven-dpo-seed42-768.pth` | Preference-repair diagnostic candidate | 79/160, 19/32 |
| `minimind-align-error-driven-simpo-seed42-768.pth` | Preference-repair diagnostic candidate | 76/160, 19/32 |
| `minimind-align-corrected-grpo-seed42-768.pth` | Single-seed corrected-GRPO diagnostic | 19/32 release slice; no incremental gain over SFT |

All assets are loaded with strict state-dict matching against the MiniMind
768-hidden-size, 8-layer configuration. `SHA256SUMS.txt` is included in the
release and must be checked before loading a weight.

These files are research artifacts, not a default-model replacement. The
release does not include raw Alignment v2 data, optimizer state, intermediate
checkpoints, or full generation logs. See [the model card](model_card.md),
[limitations](limitations.md), and [upstream attribution](upstream.md).

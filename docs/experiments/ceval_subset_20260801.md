# MM-F015：C-Eval representative subset

The evaluator ran `ceval/ceval-exam` at revision `617524a00b307ff6f9933702f724131fe12ca7ce`, seed 42, greedy decoding, 20 questions per requested subject, for 100 questions total. It records the manifest, source metadata, raw predictions, summaries and SHA256 hashes in `results/experiments/ceval_subset_20260801_v3/`.

The pinned revision has fewer than 20 labelled `val` rows for four requested subjects, so the evaluator uses `val` and supplements from `dev` only when necessary. The requested `business_ethics` label has no matching config in this revision; it is explicitly mapped to the available `business_administration` config. These mappings are recorded in `source.json` and mean this is directional evidence, not an official full C-Eval score.

| model family | total | accuracy | invalid answers |
| --- | ---: | ---: | ---: |
| align_sft_v2_pilot | 12/100 | 12% | 44 |
| simpo_v1_pilot | 12/100 | 12% | 44 |
| GRPO seed 42/43/44 | 12/100 each | 12% each | 44 each |
| CISPO seed 42/43/44 | 12/100 each | 12% each | 44 each |

Every per-subject result is retained in `summary.json`; the result does not distinguish the models on this subset. Gemini was not used as a scorer.

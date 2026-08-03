# MiniMind-Align 路线图

| Sprint | 内容 | 门禁 |
| --- | --- | --- |
| A | Alignment v2 数据工程 | smoke/pilot 审计全通过，现有 SFTDataset 可加载 |
| B | Alignment SFT v2 与 LoRA | smoke 无 NaN，权重可回载，统一冻结测试集评测 |
| C | On-policy hard negative、DPO v2、SimPO | 候选、validator、偏好数据和 loss 数值测试可追溯 |
| D | 轻量级 Reward Model | response 末 token、pairwise accuracy、长度偏置可测 |
| E | GRPO/CISPO | 程序可验证任务 smoke 通过并记录 KL/reward/坍缩 |
| F | 统一评测、消融、Demo、交付 | 事实/失败/未完成状态完整，复现默认不启动昂贵训练 |

## 现场事实

- L4 24GB，根盘约 120GB 可用，GPU 基线检查时空闲。
- Alignment v1 实际规模：SFT train 600、validation 100、DPO train 600、validation 100、测试 prompt 100。
- align_sft_dpo_v1 Gemini 盲评为 99/100，缺失 av1_test_conciseness_0007；冻结报告按 99 条有效样本，不补造判断。
- Alignment v2 smoke（24/24）和 pilot（新 train 1000、新 validation 160）审计通过，SFTDataset 实际加载 batch=(2,512)，数据门禁允许进入 Sprint B。

## 当前交付状态（2026-07-31）

- Sprint C：on-policy 128/32 prompts、4 candidates/prompt、DPO v2 与 SimPO 已完成；SimPO 64-step pilot 保留为规则 validator leader，256-step full 标记为长度/重复过度优化对照。
- Sprint D：Reward Model 64-step pilot 的 validation pair accuracy 为 20/32，Gemini C003 non-tie 一致率为 12/21；长度差相关性为 -0.5057，已在报告中标注短回答偏置风险。
- Sprint E：GRPO/CISPO lite 各完成 4-step、16 prompts、4 generations 的规则奖励 pilot；step 3 出现组内 reward std=0，结论限定为不稳定且不可比较的 tiny pilot。
- Sprint F：统一报告、六张静态图、成本报告、失败分析、复现脚本、Streamlit Demo 和面试/简历材料已归档；另完成 DPO/SimPO 各 100 条完整 Gemini 冻结集对照。默认复现脚本不启动昂贵训练。

## 当前交付状态（2026-08-01）

- MM-E009：RL 三 seed validation checkpoint selection 已完成。GRPO/CISPO 均未达到 +3 pass 改进门禁，默认权重不变；结果归档于 `docs/experiments/rl_method_upgrade_20260801.md`。
- MM-F015：C-Eval 5×20 代表子集已完成。固定 revision 的实际 schema 使用了显式 val/dev 补足，并将 `business_ethics` 映射为 `business_administration`；结果只作方向性证据。
- MM-F016：最终门禁、成本和限制已汇总，服务器保持运行。
- MM-E010：原生 Alignment v2 数据隔离、独立 validation slice、checkpoint 回载选点修复、GRPO/CISPO 六个正式 run 和 reward-hacking 诊断已完成；六个 run 均 KL 早停，回载后的 selected validation 均为 13/32，未达到模型晋级门禁。
- MM-F017：冻结集复核、最终 gate 和归档已完成。回载后的 validation 三 seed 均值均为 13/32，相对 baseline 13/32 为 0，最终 `NOT_MET_NO_MODEL_CHANGE`；默认模型不改变，服务器保持运行。

最终项目状态：PASS_WITH_LIMITATIONS；本轮 RL 晋级门禁：NOT_MET_NO_MODEL_CHANGE。公开子集和长程 RL 已执行但没有产生改进结论；真实美元账单仍无可验证 export。

## 当前交付状态（2026-08-01，MM-E011 / MM-F018）

- 稳定性诊断已完成：GRPO/CISPO 各运行 control、low_lr、accum16，seed=42；训练器 telemetry、审计器和串行 wrapper 已归档。
- 六个正式 run 均因 KL 连续两步超限早停；accum16 比 control 更早触发，low_lr 没有延后触发；所有 selected validation 为 13/32，未出现稳定性改善条件。
- 最终状态为 `NOT_MET_NO_MODEL_CHANGE`；默认模型不改变，正式三 seed 扩展暂停。若继续，先定位 KL 尖峰和大梯度来源，再保持现有门禁做小型可复现实验。

## 当前交付状态（2026-08-01，MM-F019）

- 已完成六个稳定性 run 的离线 KL 尖峰、梯度裁剪、截断/重复和 reward-to-validation gap 归因；结果只作诊断。
- 六个 run 均出现高 KL 尾部与梯度裁剪共现，训练 reward/validator 峰值没有传导至 selected validation；精确 micro-batch/token 来源因 telemetry 缺失仍未确定。
- 当前门禁继续为 `NOT_MET_NO_MODEL_CHANGE`；暂停正式三 seed 扩展，下一步先补充可回放 micro-batch telemetry。

## 当前交付状态（2026-08-01，MM-E012 / MM-F020）

- 已完成 micro-batch telemetry、GRPO control seed=42 短程诊断和离线 spike-source audit；balanced formal run 覆盖八类、32 个 micro-batch，checkpoint reload/selection 和 JSON 完整性检查通过。
- 审计状态为 `BROAD_SPIKE_DIAGNOSTIC`：KL/梯度/质量异常跨类别分布，`termination` 仅是 top-3 类别交集中的较强信号，精确 micro-batch top-3 交集为 0，因果来源仍未解析。
- 首次 coverage-incomplete formal run 已保留并明确排除；最终 balanced run 的 selected validation 为 13/32，safety/termination 为 4/4，默认模型不改变。
- 当前状态为 `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`。下一阶段优先做独立学习率、累积步数或更新尺度对照；在重新完成三 seed 晋级门禁前，不扩展正式 RL、不替换默认模型。

## 当前交付状态（2026-08-01，MM-E013 / MM-F021）

- 已完成 GRPO seed=42 的 control、low_lr、clip_half 更新尺度对照；每个 formal run 8 steps、64 个 micro-batch、八类覆盖，smoke/formal/audit 均归档。
- 修正后的 stability audit 纳入三条件，结果为 `stability_improved_conditions=[]`；low_lr 和 clip_half 均未满足稳定性改善门槛。
- 三个 spike audit 均为 `SOURCE_LOCALIZED_DIAGNOSTIC`，主要重复类别为 conciseness、format、termination；该结果是聚合诊断，不是因果证明。
- 当前门禁为 `NOT_MET_NO_MODEL_CHANGE`；继续暂停正式三 seed 扩展、C-Eval 和冻结集评测，下一步先审计重复出现的 prompt/category 与 reward components。

## 2026-08-01 update: MM-E014 / MM-F022

The update-scale follow-up localized recurring prompt/category signals without adding GPU time. The current status is `RECURRING_PROMPT_DIAGNOSTIC`; model promotion remains `NOT_MET_NO_MODEL_CHANGE`.

Next work is a read-only audit of the 26 recurring prompt identifiers and the five recurring categories, with special attention to termination/format/conciseness templates, truncation and natural-end behavior, and why parse/field reward components are zero in the persisted slice. Formal CISPO, three-seed expansion, C-Eval, and frozen-set reruns remain paused until this input audit produces a testable intervention.

## 2026-08-01 update: MM-E015 / MM-F023

- Generation-end telemetry was corrected to use EOS position and `max_gen_len`; old quality fields remain readable but are labeled `LEGACY_QUALITY_TELEMETRY_UNTRUSTED`.
- The v2 offline audit passed source chosen validation `128/128` and replay consistency `1536/1536`; metadata missing was `0`, with no new GPU time.
- Cross-condition recurrence is now named explicitly: 18 prompt groups and 4 category groups. It does not mean within-run duplicate sampling and does not establish causality.
- Current status remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`. The next permitted experiment is a corrected GRPO control smoke; formal CISPO, three-seed expansion, C-Eval, frozen evaluation, reward changes, and default-model replacement remain paused.

## Corrected control smoke result

- The isolated GRPO seed-42 smoke completed in 13 seconds with 2/2 steps and checkpoint reload success.
- Termination telemetry was internally consistent: EOS `4`, max-token `4`, unknown `0`; no model-improvement conclusion is drawn from the 2-prompt validation.
- Continue to hold the default model and formal RL expansion. The next decision must be based on a separately scoped corrected run, not on legacy quality telemetry.

## 2026-08-02 update: MM-E016 / MM-F024

- A corrected balanced GRPO control diagnostic completed for 4 steps with 32 micro-batches and 256 samples across all eight Alignment v2 categories.
- EOS telemetry remained valid (250 EOS, 6 max-token, 0 unknown), checkpoint reload passed, and all persisted JSON/JSONL artifacts were finite.
- The spike audit remained `BROAD_SPIKE_DIAGNOSTIC`: KL/gradient anomalies were broad rather than isolated to a single category; validation stayed 13/32 and safety/termination stayed 4/4.
- Current state remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`. Do not expand CISPO or seeds, rerun C-Eval/frozen evaluation, or change reward/default weights without a new explicit plan.

## 2026-08-02 update: MM-E017 / MM-F025

- The optional KL trust-region guard was implemented with policy/AdamW rollback and bounded learning-rate backoff. The legacy behavior remains unchanged when the guard target is omitted.
- GRPO seed 42 smoke and formal diagnostics both exhausted the four guard attempts at the first update. The formal post-step KL mean was `0.01638` against a `0.005` target; no optimizer step was accepted and no checkpoint was written.
- Audit status is `GUARD_UNRESOLVED_BASELINE_RETAINED`; baseline validation stayed `13/32`, safety/termination stayed `4/4`, and the default model was unchanged. The project remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`.
- Pause further RL expansion. Any next intervention must be separately approved and narrowly alter update parameterization or the KL measurement design; do not infer model improvement from guard behavior.

## 2026-08-02 update: MM-E018 / MM-F026

- The per-attempt KL/dtype/parameter-delta/digest telemetry implementation was added without changing reward, the production bfloat16 KL gate, checkpoint selection, or the default model.
- The single GRPO seed-42 telemetry smoke was started once and failed before the first guard attempt record because the initial AdamW state digest could not hash a scalar tensor. GPU wall time was 13 seconds; no optimizer step or checkpoint was accepted.
- The scalar digest path was fixed in commit `38e9219997b62574f9f09513917c25e1e9ca4e4b`, and 64 static/regression tests passed. Per the protocol, the failed smoke was not automatically rerun.
- The offline audit is `TELEMETRY_INCOMPLETE`, so this round cannot distinguish real update sensitivity from bfloat16 measurement sensitivity. The default state remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`; a new smoke requires an explicit decision.

## 2026-08-02 corrected MM-E018 / MM-F026 retry decision

- The explicitly authorized corrected smoke completed in a new retry root and recorded all four backoff attempts, same-rollout bfloat16/float32 KL, parameter deltas, and exact policy/AdamW rollback digests.
- The audit is `BF16_MEASUREMENT_SENSITIVE`: bfloat16 post-step KL means were approximately `0.267–0.275`, while float32 means were below `0.00023`; every attempt had opposite gate conclusions.
- This resolves the immediate attribution question as a measurement-dtype issue for this guard diagnostic, not as evidence requiring a new optimizer/update-scale setting. Before further RL, independently validate/correct KL measurement semantics; do not expand CISPO/seeds, change reward, or replace the default model.
- Final state remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`; server remains `RUNNING` with the GPU idle.

## 2026-08-02 MM-E019 / MM-F027 measurement precision decision

- A detached true-float32 policy-copy diagnostic was added without changing the production bfloat16 gate, reward, optimizer, checkpoint selection, or default model.
- The same-rollout comparison gave identical no-autocast and true-float32 KL on 4/4 attempts, while bfloat16 autocast differed and rejected all 4 attempts. The audit status is `BF16_AUTOCAST_SENSITIVE`.
- The immediate source is localized to the bfloat16 autocast measurement path. Do not select a new optimizer/update-scale intervention yet; first validate a corrected float32/no-autocast KL measurement against an independent reference. Formal RL expansion remains paused.
- Final state remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`; GPU is idle and server remains `RUNNING`.

## 2026-08-02 MM-E020 / MM-F028

- Independent reference-KL semantics audit: REFERENCE_KL_SEMANTICS_CONSISTENT_LIMITED.
- Formula, completion-mask scope, token-weighted aggregation, source-path alignment, persisted gate, and rollback evidence passed.
- Historical E019 telemetry lacks token-level log-probabilities and masks; a corrected no-autocast smoke with replay fields is required before treating the KL path as fully validated.
- Formal RL expansion, optimizer/update-scale changes, reward changes, C-Eval, frozen-set reruns, and default-model replacement remain paused.
- Project state remains DIAGNOSTIC_ONLY_NO_MODEL_CHANGE; server remains RUNNING.

## 2026-08-02 update: MM-E021 / MM-F029

- A single GRPO seed-42 smoke added token-level replay for all four KL-guard backoff attempts and all three measurement variants. The final offline audit is `TOKEN_REPLAY_VALIDATED`; exact rollback, sample linkage, mask/reference alignment, and aggregate consistency passed.
- The production bfloat16 autocast gate rejected all four attempts (`0.267400–0.275372` mean KL), while no-autocast bfloat16 and detached full-fp32 measurements matched and stayed below `0.000229`. No optimizer step or checkpoint was accepted.
- This confirms a replayable measurement-path discrepancy in the tested bfloat16 autocast path, but does not establish causal optimizer instability. The state remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`; do not alter reward, optimizer/update-scale, checkpoint selection, or the default model.
- The server remains `RUNNING` with the L4 idle. Formal RL/CISPO/seed expansion, C-Eval and frozen-set reruns remain paused pending a separately approved measurement-corrected training design.

## 2026-08-02 update: MM-E022 / MM-F030

- An explicit `fp32_no_autocast` post-step gate was added without changing the legacy default, reward, training loss, pre-step early-stop, checkpoint selection, optimizer/update-scale, or default model.
- The only GRPO seed-42 smoke accepted 2/2 updates at LR multiplier `1.0`; active post-step KL means were `0.000228583` and `0.000092137`, state digests were continuous, 12 replay rows passed, and the step-2 checkpoint reloaded.
- Audit status is `CORRECTED_GATE_ACCEPTED_2_STEPS_DIAGNOSTIC`, but step 2 also triggered `PRESTEP_PRECISION_DIVERGENCE`: legacy pre-step mean `0.00259953` versus FP32/no-autocast `0.00000265427`.
- Next work is a narrow training-loss/pre-step KL precision audit on identical tokens. Do not proceed directly to the 4-step balanced run, CISPO, three seeds, C-Eval, frozen-set evaluation, optimizer/update-scale changes, or model replacement.
- Project state remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`; the server remains `RUNNING` and the L4 is idle.

## 2026-08-02 update: MM-E023 / MM-F031

- Same-token pre-step replay compared the unchanged bfloat16-autocast training loss with an FP32/no-autocast shadow on four micro-batches; shadow gradients were isolated from production `.grad` state.
- The audit is `TRAINING_AUTOCAST_PRECISION_SENSITIVE`. At step 2, bfloat16 loss was `979.38×` the FP32 shadow and gradient norm was `29.41×`; replay, checkpoint reload, and state continuity passed.
- All advantages were zero, so the result is limited to the KL-loss path and cannot establish nonzero-advantage ratio/clipping behavior or model quality.
- Next stage: add an explicit, default-off FP32/no-autocast training-forward/loss mode; first pass a nonzero-advantage contract test, then run at most one 2-step smoke. Do not tune optimizer scale around the precision artifact or proceed to 4-step/CISPO/three-seed expansion yet.
- State remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`; server remains `RUNNING`, L4 idle.

## MM-E026 / MM-F034 decision (2026-08-02)

- Controlled nonzero-advantage GRPO seed-42 active-path smoke passed 2/2 accepted steps with the explicit FP32/no-autocast training-forward and gate; parameter deltas, replay linkage, state continuity and checkpoint reload passed.
- This was a diagnostic reward injection, not a natural-reward experiment. It does not authorize formal RL, CISPO, multi-seed, benchmark, frozen-set, optimizer or default-model changes.
- Preserve `LEGACY_BF16_SHADOW_DISAGREEMENT`, `PRESTEP_PRECISION_DIVERGENCE`, `CONTROLLED_REWARD_NOT_QUALITY_SIGNAL` and `CHECKPOINT_SELECTION_SMOKE_ONLY` as limitations. Project state remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`.
- Before broader training, make a separate decision on a natural-rule-reward corrected smoke with explicit validation; do not treat the controlled result as model improvement.

## MM-E027 / MM-F035 decision (2026-08-02)

- The natural `rule_reward` corrected smoke completed 2/2 accepted FP32/no-autocast gate steps and passed replay, state continuity and checkpoint reload.
- All generated groups received the same reward `0.1`, yielding zero advantage and zero parameter delta. This is a natural reward distribution/coverage diagnostic, not a model result.
- Keep `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`. Before another GPU task, audit reward-component diversity, validator pass distribution and within-group reward collapse offline. Do not expand formal RL, CISPO, multi-seed, C-Eval, frozen evaluation or optimizer/update-scale experiments.
## 2026-08-02 update: MM-E024 / MM-F032

- Added an explicit opt-in FP32/no-autocast training-forward and loss mode; the legacy bfloat16 autocast training path remains the default.
- One GRPO seed-42 two-step smoke completed with active FP32 post-step gate, replay, state continuity, checkpoint reload, and 102 regression tests passing.
- The smoke groups all had zero advantage, so policy parameters did not move. This validates telemetry and active-path selection only; it does not validate nonzero-advantage clipping or model quality.
- Keep the project at `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`. Before any longer or multi-seed RL, require a deterministic nonzero-advantage contract/fixture and a new explicit decision. Do not run formal RL, CISPO, C-Eval, frozen-set evaluation, reward changes, optimizer tuning, or default-model replacement from this result.

## 2026-08-02 update: MM-E025 / MM-F033

- The deterministic nonzero-advantage contract fixture and offline audit passed for both GRPO and CISPO. Ratio clipping was observed, production/diagnostic loss and KL matched, and FP32 plus bfloat16-quantized gradients were finite and nonzero.
- This closes the E024 zero-advantage math/API limitation only. It does not prove that the live trainer produces nonzero advantages, that updates improve validation, or that any optimizer/update-scale change is warranted.
- Keep the project at `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`. The next possible GPU action is one separately approved narrow active-path smoke with controlled nonzero reward input; formal RL, CISPO expansion, multi-seed, C-Eval, frozen-set evaluation, reward/optimizer changes, and default-model replacement remain paused.

## 2026-08-02 update: MM-E028 / MM-F036

- The offline natural reward diversity audit scanned the current E027 smoke and older E010/E009 artifacts without starting GPU work. The current E027 path collapsed all eight samples to termination-only reward `0.1`, with no nonzero group reward spread.
- Historical E010/E009 diversity is reported separately because those artifacts use legacy schemas/data; it is not causal evidence for the current path.
- Keep `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`. Before another GPU smoke, audit prompt/output component coverage and validator/termination variability under the current generation limits. Do not expand formal RL, CISPO, multi-seed, C-Eval, frozen evaluation, or change reward/optimizer/default weights.

## 2026-08-02 update: MM-E029 / MM-F037

- The offline coverage audit confirms the v2 manifest is balanced (8 categories × 16), but the current E027 artifact covers only conciseness (8 samples, 4 prompts). Validator pass is `0/8`, with one failure reason; reward spread is absent.
- EOS and max-token endings both receive termination reward `0.1`, so the current termination component is not sensitive to that distinction. Rule-function probes vary, but they are not model evidence.
- Keep `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`. Do not start formal RL or change optimizer/reward/default weights. If a new smoke is later authorized, require balanced category sampling and explicit validator/outcome coverage before interpreting any update.
## 2026-08-02 MM-E030 / MM-F038 decision

The balanced coverage smoke closes the immediate sampling-observability gap: all eight Alignment v2 categories, families and prompts are represented in the 32 generated samples, and natural reward replay is internally consistent. It does not close the quality gap: current validator pass is 2/32 and balanced validation is 0/8. The FP32/no-autocast active gate accepts two nonzero updates, but legacy bfloat16 shadow disagreement remains a measurement warning.

Keep the project at `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`. The next decision is an offline validator/output-quality audit, not a formal RL expansion. Formal GRPO/CISPO multi-seed runs, C-Eval, frozen evaluation, reward changes, optimizer changes and default-model replacement remain paused.
## 2026-08-02 MM-E031 / MM-F039 decision

The offline output-quality audit confirms that E030's balanced sampling and replay linkage are sound: category/family/prompt coverage is `8/8`, source chosen validation is `8/8`, and persisted reward/component replay is exact. It also shows that the live quality signal is sparse (`2/32` validator pass), with prevalent `max_new_tokens` hits (`20/32`) and natural end only `12/32`.

Keep `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`. This evidence is insufficient for a reward or optimizer change and does not justify formal RL expansion. A future quality/output intervention must be separately specified and audited before any multi-seed training, C-Eval or frozen evaluation.

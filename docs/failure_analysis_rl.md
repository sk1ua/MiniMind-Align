# RL 阶段失败与风险分析

## 已记录的工程失败

- 直接执行 `trainer/train_grpo.py` 不适合作为本阶段入口：它依赖外部 1.8B reward model 和 rollout engine，容易超出轻量 64M 项目的资源边界。因此保留原脚本，新增 rule-reward lite 路线，不伪装成复用了完整外部 RL 基础设施。
- GRPO lite 首次 smoke 在 autograd 处失败：`MiniMind.generate` 使用 `inference_mode`，生成 tensor 不能直接作为反传输入。修复为 `detach().clone()`，失败实验目录和日志保留，retry 通过。

## 训练信号风险

- step 3 的 reward std 为 0：4 个候选组出现同分，属于 group-relative advantage 失去区分度的坍缩信号。
- step 4 KL 从约 `1e-4` 级升到 `0.002536`，同时 reward mean 跳到 `0.7557`；这可能是 rule reward 被快速 hacking 的早期迹象。
- completion 长度从 26.31 降至 8.69，再回到 14.31；不能把长度变短直接解释为质量提升。
- GRPO 与 CISPO 4-step 日志完全一致，当前数据规模无法区分 loss 设计的实际收益。
- 延长对照使用 8 类 prompt、32 条 validation prompt、每组 8 generations 和 8 steps。GRPO/CISPO 的冻结测试 validator pass 都是 51/100，KL 峰值分别约 0.00123/0.00117，仍不能证明算法优劣；GRPO 平均 64.98 tokens，CISPO 63.91 tokens，长度差也不足以构成质量结论。

## 尚未完成

本阶段没有使用外部 1.8B reward model，也没有宣称 GRPO/CISPO 收敛或优于 DPO/SimPO。延长实验仍是 rule-reward lite、短程、固定 validation prompt family；后续若继续扩大，必须增加独立 prompt families、设定 KL early stop，并单独检查 validator reward hacking。

## MM-E009 新增观察

- 计划指定的 128 条训练 manifest 来自旧 Alignment v1，metadata 为空；直接调用 v2 rule validator 会在 conciseness 等类别触发 `KeyError`。失败 smoke 目录保留，正式 run 使用未修改原始文件的 resolved manifest，并在报告中标记这一适配假设。
- 六个 16-step run 均未触发 KL 或 safety/termination early stop。validation checkpoint 选择是可复现的，但 GRPO/CISPO 三 seed 平均提升均低于 +3 pass 门禁，不能写成改进。
- C-Eval 子集暴露了数据 revision 与计划科目不一致：高中文科等配置的 val 行数不足 20，且没有 business_ethics config。正式结果显式记录 val/dev 补足和 business_administration alias，避免隐藏 schema 偏差。

## MM-E010 / MM-F017：初始 data-isolation run 与审计

- 训练集改为原生 Alignment v2 programmatic rows，metadata 非空；validation 改为避开旧前四条的第 5–8 条 slice。ID/family overlap 检查通过，说明本轮没有把旧 v1 metadata 缺失问题带入正式训练。
- 六个正式 run 均正常退出，但都在 KL 连续两步超过 0.005 后早停；selected checkpoint 只到 step 4 或 8。三 seed validation 均值均为 14/32，而 baseline 为 13/32，+1 低于 +3 晋级门禁。
- 审计发现 max-length hit 和 repetition penalty 增加；部分训练 step 的 max-length hit 接近 98.4%。独立 validation 的 safety、termination、natural end 没有下降，且没有 empty-response warning；因此结论是存在 reward-hacking 风险信号，但不是已证实的单一原因。
- 七个冻结集评分均完成：baseline 50/100，CISPO 50/100（各 seed），GRPO 51/100（各 seed）。冻结集只支持有限泛化观察，不改变 checkpoint 选择，也不支持模型晋级。
- 追加的完整 v2 validation 覆盖评测加载了 6 个已保存 checkpoint；baseline 和所有回载 checkpoint 都是 47/160，全量 natural-end 为 160/160，预注册 32 条切片也都是 13/32。
- 六个 `selection.json` 的内存模型指标为 14/32，但独立回载后均为 13/32，说明当前选点指标不能直接视为 checkpoint artifact 的可复现指标。抽查保存张量为 `float16`，保存路径的 `.half()` 是最可能机制，但仍是推断；后续训练必须保存后回载再选点。

这组初始 run 的训练进程内存指标曾显示 14/32，但独立回载后为 13/32；因此只作为历史诊断保留，不能把 14/32 写成 checkpoint artifact 的结果。默认权重保留，后续若继续应优先提高独立 validation 覆盖、减少短程 KL 早停造成的有效训练步数，并单独分析长度/重复与 validator reward 的因果关系。

## 后续修复：checkpoint 回载选点

已修复训练器：checkpoint 保存后先重新加载序列化 artifact，再计算 validation 和 quality gate；
只有带 checkpoint 的回载记录可以进入 selection。1-step smoke 与独立 `max_steps=0` 回载评测
的稳定输出一致，18 项 RL/回归测试通过。该修复只影响未来实验的选点可复现性，不回写或重新解释
E010/F017 的既有结果。

## MM-E010 / MM-F017：checkpoint 回载修复后的正式复跑

- 修复后的实验根目录为 `results/experiments/rl_data_isolation_reload_fixed_20260801/`，运行代码 commit 为 `14076033f25fc7dfa35403f2d7beccb46ae43d5c`；六个正式 run 均 exit code 0，所有 checkpoint 选择均基于保存后回载的 artifact。
- GRPO seed 42/43/44 均在 20 steps、CISPO seed 42/43 在 20 steps、CISPO seed 44 在 18 steps 因 KL 连续两步超过 0.005 早停；六个 selected checkpoint 均为 step 4。
- 六个 selected artifact 的独立 validation 均为 13/32，safety/termination 均为 4/4，natural end 均为 32/32；三 seed 均值相对 baseline 的增量为 0，最终为 `NOT_MET_NO_MODEL_CHANGE`。
- 审计 warning 统一包括 `train_validator_gain_without_validation_gain`、`reward_gain_without_validation_gain`、`max_length_hit_increase`、`repetition_penalty_increase`；没有 empty-response 或 validation safety/termination/natural-end 下降 warning。审计仍为诊断，不改变 checkpoint 门禁。
- 冻结集 baseline 为 50/100；GRPO 为 51/100、51/100、51/100；CISPO 为 50/100、50/100、51/100。冻结集只提供有限泛化证据，不参与选点或晋级。

这次复跑确认：旧结果的不确定性不能归因于“v1 metadata 缺失”后就视为已改进；换用原生 v2 数据并修正回载选点后，validation 仍未达到 +3 pass 门禁。当前最稳妥结论是 reward-hacking 风险信号存在，但尚未证明唯一因果机制；默认模型不改变。

## MM-E011 / MM-F018：稳定性诊断

- 训练器新增了 loss、裁剪前/后梯度范数、ratio 分位数、micro-batch KL 分布和 `pre_optimizer_step` 标记；`align.rl_rules` 只增加诊断接口，现有 reward/loss API 保持兼容。
- GRPO/CISPO 的 control 与 low_lr 都在 step 20 触发 KL early stop；accum16 两种方法都在 step 10 触发，未满足相对 control 延后至少 4 步的定义。六个 selected checkpoint 回载后的 validation 都是 13/32，safety/termination 都是 4/4。
- low_lr 没有延后 KL 触发，且至少一项 max KL 或 max pre-clip gradient 高于 control；accum16 虽然梯度最大值较低，但更早触发且 max KL 更高。因此没有条件可记为稳定性改善。
- 训练 telemetry 显示所有正式 run 都有梯度裁剪活动；部分 step 的 max-length hit 很高，且 KL 分布存在远高于均值的尖峰（最大 reference KL 约 2.18–2.77）。这支持“训练信号不稳定”的诊断，但不证明单一因果机制。
- 审计器 `results/experiments/rl_stability_diagnostic_20260801/stability_audit/summary.json` 完整通过，所有 JSON/JSONL finite；`PASS` 代表审计流程完成，不代表模型或优化条件通过。最终状态为 `NOT_MET_NO_MODEL_CHANGE`，默认模型不变。

本轮后暂停正式三 seed 扩展。若继续，应先用更小规模、可复现的 telemetry 对照定位 KL 尖峰和梯度异常来源，保持现有 KL、quality 和模型晋级门禁不变。

## MM-F019：KL 尖峰与 reward 传导离线归因

- 对六个稳定性正式 run 的 step summaries 和 samples 做离线审计；六个 run 都报告训练 validator/reward 上升但 selected validation 不变，训练 validator 峰值约 0.8594–0.9062，selected validation 仍为 13/32。
- 所有 run 都出现 KL max/P95 尾部集中和梯度裁剪；最高 reference KL 约 2.77。截断与重复惩罚信号共同出现，但没有 empty-response 或 safety/termination 下降。
- 可验证事实支持“高 KL 尾部与裁剪梯度共现”的诊断推断；不能据此声称因果，也不能把诊断写成模型改进。
- 当前训练产物没有保存 per-micro-batch log-prob/gradient 向量，精确定位到 micro-batch/token 仍未完成。权威输出位于 `results/experiments/rl_stability_diagnostic_20260801/spike_source_audit_v2/`。

因此保持 `NOT_MET_NO_MODEL_CHANGE`，暂停正式三 seed 扩展；下一步若继续，先增加可回放 micro-batch telemetry，再运行最小 GPU 对照。

## MM-E012 / MM-F020：micro-batch 尖峰来源诊断

- 训练器新增 `microbatch_summaries.jsonl`，每个 micro-batch 立即 flush；记录 prompt/category、reward components、KL/ratio tail、截断/重复/自然结束和 backward 前后梯度增量。该改动不增加第二次 backward，也不改变 optimizer、KL early-stop 或 checkpoint selection 语义。
- smoke 运行 2 steps、4 个 micro-batch，验证 JSONL 写入、有限值、checkpoint reload 和 selection。由于样本量很小，smoke 的 `SOURCE_LOCALIZED_DIAGNOSTIC` 只表示管线可工作，不表示真实来源已定位。
- 首次 4-step formal run 因 manifest 按类别成块排列，32 个 micro-batch 只覆盖 `conciseness`/`format`。该 run 未删除、未覆盖，并标记为 `coverage_incomplete`；它不进入最终 category attribution。
- 修正为 `interleave_categories=true` 后，balanced formal run 覆盖八类、每类 4 个 micro-batch。step 2/3 出现高 KL tail，四个 step 都发生梯度裁剪；top-3 KL 与 top-3 梯度的类别交集只有 `termination`，但精确 micro-batch key 交集为 0。
- 审计状态为 `BROAD_SPIKE_DIAGNOSTIC`：异常跨类别分布，termination 是较强信号但不是已证实的根因；不能把该结果写成数据或 reward 的因果结论。

本轮 GPU 结果和离线审计均通过完整性检查，selected validation 仍为 13/32，safety/termination 为 4/4，默认模型不变。当前门禁为 `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`；后续优先做优化器/更新尺度对照，而不是扩展 seed 或替换模型。

## MM-E013 / MM-F021：更新尺度对照结果

- 三个 GRPO seed=42 formal run 均完成 8 steps、64 个 micro-batch，八类 prompt 通过交错顺序覆盖；训练、validation、checkpoint reload 和资源日志完整。
- `low_lr` 将 KL P95 从 control 的约 `0.02795` 降至约 `0.02756`，但 KL max 从约 `0.90024` 升至约 `1.59414`；因此不能按预设门槛判定为稳定性改善。
- `clip_half` 将 post-clip 梯度范数限制到约 `0.5`，但 pre-clip 梯度峰值仍约 `2.12`，KL max 约 `0.90066`，没有消除尖峰。
- 三条件 selected validation 均为 `13/32`，safety/termination 均为 `4/4`；没有条件满足“KL 延后至少 4 steps、尾部不高于 control、质量不下降”的完整定义。
- 修正后的 stability audit 已纳入 `clip_half`；此前漏项的 `stability_audit/summary.json` 保留为审计过程记录，最终以 `stability_audit_v2/summary.json` 为准。
- 三个 spike audit 的启发式状态均为 `SOURCE_LOCALIZED_DIAGNOSTIC`，但只能说明 top-K 聚合中若干类别重复出现，不能证明 prompt/category 是因果源。

结论是“当前更新尺度对照没有解决训练稳定性问题”，而不是“某个条件导致模型退化”。保持 `NOT_MET_NO_MODEL_CHANGE`，不扩展 seed、不替换默认模型。

## MM-E014 / MM-F022 prompt and reward-component audit

The follow-up offline audit was designed to distinguish a small set of repeatedly problematic inputs from a broad optimizer failure. It read only the completed update-scale formal telemetry and sample metadata, with no new training.

Coverage was complete: 3 runs, 192 micro-batches in aggregate, 1,536 linked samples, and no missing sample association. The deterministic top-K union rule identified 26 recurring prompt IDs and 5 recurring categories across at least two conditions. `termination` was repeatedly represented in KL and gradient tails; `format` appeared in all three tail types; `conciseness` was concentrated in gradient tails; `reasoning` was concentrated in quality/truncation tails.

This narrows the next audit target but does not resolve causality. The recurrence can reflect prompt scheduling, category composition, or an interaction between generated response shape and the update. Component-level correlations are descriptive only. In particular, validator reward did not explain KL max in this slice, termination reward had negligible KL association, and repetition penalty did not show a positive KL/gradient association. The zero-valued parse and field components should be checked against the reward-input construction before any intervention.

Decision: keep `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`; do not expand to CISPO or three-seed RL. First inspect the recurring prompt records, category templates, truncation conditions, and reward component input coverage, then design one narrowly scoped intervention with a pre-registered diagnostic criterion.

## MM-E015 / MM-F023 telemetry correction and validator replay

The earlier quality-tail signal was partly a measurement problem: `max_length_hit` was inferred from micro-batch padding width. The trainer now records `termination_reason`, `eos_seen`, `finished_naturally`, and `max_length_hit` from the per-sample EOS position and generation budget. The old artifacts remain readable and are labeled `LEGACY_QUALITY_TELEMETRY_UNTRUSTED` rather than being rewritten.

The v2 offline audit found no input or validator inconsistency: all 128 source chosen records passed their category validator, and replay matched the persisted validator component for 1,536/1,536 generated samples. The remaining cross-condition prompt/category signals are therefore diagnostic localization evidence, not proof of reward hacking. Within-run occurrences and cross-condition recurrence are now reported separately, with category/family exposure denominators and reward-component coverage.

Decision: retain `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`. If a corrected control smoke is later run, interpret quality metrics only from the new schema; do not reuse the legacy quality tail to select checkpoints or alter the default model.

## Corrected control smoke follow-up

A single isolated GRPO smoke completed without a training or reload failure. It produced 8 linked samples across 2 steps: 4 EOS terminations, 4 max-token terminations, and no short no-EOS unknowns. This confirms telemetry consistency, not RL stability or generalization. The validation slice was only 2 prompts and remained `0/2`; the result is therefore `PASS_TELEMETRY_NO_MODEL_CHANGE`, not a model-improvement result.

## MM-E016 / MM-F024 corrected balanced GRPO diagnostic

The first corrected balanced follow-up ran one GRPO control condition (seed 42) for 4 steps with 8 interleaved Alignment v2 categories, 32 micro-batches, and 256 generated samples. The new EOS-based fields remained internally coherent: 250 samples ended with EOS, 6 reached `max_new_tokens`, and none were classified as short no-EOS or unknown. The step-4 validation used the reloaded checkpoint.

The audit status was `BROAD_SPIKE_DIAGNOSTIC`. KL tails and gradient clipping remained broad across the balanced category schedule. `termination` was present in the category-level top-3 KL/gradient intersection, but no exact micro-batch key was shared by both top-3 lists. Validation stayed at 13/32 versus the 13/32 baseline, with safety and termination both 4/4. This supports an unresolved optimizer/update-scale problem, not a proven prompt-family or reward-component cause.

The correct conclusion is diagnostic only: no reward change, checkpoint-selection change, or default-model change is justified. Further training should remain paused until a new narrowly scoped intervention is explicitly selected.

## MM-E018 / MM-F026 KL guard telemetry smoke failure

The telemetry extension itself was implemented and passed static checks, but the single authorized smoke exposed a serialization edge case before the first attempt record: AdamW creates a scalar `step` tensor, and the initial digest helper attempted a byte view that requires a non-scalar dimension. The trainer stopped with exit code `1` after 13 seconds, preserving the run log, baseline validation, samples, micro-batch telemetry, resource monitor, and manifest. No policy update or checkpoint was accepted.

The fix flattens tensors before byte hashing and adds a regression that exercises the optimizer digest after an AdamW step. Post-fix static validation passed all 64 tests, but the smoke was intentionally not rerun under the fixed implementation because the protocol says a failed smoke is retained rather than automatically repeated. The audit status is therefore `TELEMETRY_INCOMPLETE`, not `REAL_UPDATE_SENSITIVITY`, `BF16_MEASUREMENT_SENSITIVE`, or `MIXED_UNRESOLVED`.

This failure is an implementation/telemetry defect, not evidence about KL sensitivity or dtype measurement. The project remains diagnostic-only with no model change; any rerun must be separately authorized and must use a new run directory.

## MM-E017 / MM-F025 KL trust-region guard diagnostic

The next narrow intervention was an optional post-update KL guard. It used the existing Alignment v2 isolated train/validation manifests, GRPO seed 42, and the unchanged reward and pre-step KL logic. The guard collected every rollout from the optimizer step, snapshotted policy and AdamW state, and tried learning-rate multipliers `1.0 -> 0.5 -> 0.25 -> 0.125`.

The formal run stopped at step 1. Pre-step KL mean was `0.000104`, while the same-rollout post-update KL was mean `0.016382`, P95 `0.055950`, and max `2.974543`. All four attempts exceeded the `0.005` mean budget. The implementation restored the policy/optimizer snapshot, marked the optimizer step rejected, skipped checkpoint creation, and retained the baseline. Two smoke attempts showed the same pattern; the retry was performed after making the post-update measurement explicitly eval-mode, so dropout/training-mode noise is not a plausible explanation for this result (the model configuration also has zero dropout).

This resolves the operational question of whether the current update can be kept inside the requested KL budget: it cannot under the tested control configuration. It does not identify the deeper causal source of the sensitivity, and it does not justify changing reward, KL thresholds, checkpoint selection, or model weights. The formal audit is `GUARD_UNRESOLVED_BASELINE_RETAINED`, with baseline/selected validation `13/32`, safety `4/4`, termination `4/4`, and zero accepted optimizer steps. The appropriate status is `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`; further RL expansion should remain paused until a separately approved intervention changes the update parameterization or measurement design.

## MM-E018 / MM-F026 corrected retry: dtype attribution

The first telemetry smoke failed before any attempt record because the digest helper mishandled a scalar AdamW step tensor. That implementation defect was fixed and the explicitly authorized retry used a separate root, so neither run was overwritten.

The retry produced four complete attempts. Each attempted update had a nonzero parameter delta, then restored the same policy and optimizer digests with zero post-rollback parameter delta. The production bfloat16 KL means were `0.267400`, `0.275372`, `0.268694`, and `0.267079`; the same-rollout float32 means were `0.0002286`, `0.0000583`, `0.0000148`, and `0.00000373`. Thus all four attempts were dtype-sensitive and all four had opposite gate conclusions. The audit status is `BF16_MEASUREMENT_SENSITIVE`, not `REAL_UPDATE_SENSITIVITY`.

This narrows the next decision: do not change learning rate, accumulation, clipping, reward, or the default model based on the previous bfloat16 guard rejection. First validate the KL computation in a measurement-corrected diagnostic, preserving the current production gate as a separate compatibility comparison. No formal RL expansion is justified by this smoke.

## MM-E019 / MM-F027 true-fp32-copy measurement audit

The next diagnostic separated three effects without changing the production guard: bfloat16 weights under bfloat16 autocast, the same bfloat16 weights with autocast disabled, and a detached float32 policy copy with autocast disabled. The latter two produced identical KL means for all four backoff attempts (`0.00000373–0.00022858`), while the bfloat16-autocast path remained `0.267400–0.275372`.

This is `BF16_AUTOCAST_SENSITIVE`: the previous guard rejection is a measurement-path artifact under the tested setup, not evidence that the actual policy update exceeds the KL budget. Rollback remains exact, and no optimizer step or checkpoint was accepted. The next plan must validate a float32/no-autocast KL measurement against an independent reference before changing optimizer/update-scale. Formal RL expansion, reward changes, and default-model replacement remain paused.

## MM-E020 / MM-F028 reference-KL semantics audit

The offline contract audit independently reproduced the scalar KL formula and checked completion-mask and token-weighted aggregation semantics. Source inspection verified that reference log-probabilities are computed from the frozen reference model, detached into the guard rollout records, and reused by the BF16, no-autocast, and detached-FP32 measurement paths. Persisted attempt gates, backoff multipliers, step summaries, finite values, and rollback digests were internally consistent.

The status is REFERENCE_KL_SEMANTICS_CONSISTENT_LIMITED. This is deliberately weaker than token-level replay because E019 did not persist token-level log-probabilities and masks. The evidence supports correcting or independently replaying the KL measurement path before any optimizer/update-scale change; it does not support changing the default model or claiming model improvement.

## MM-E021 / MM-F029 corrected token-level replay

The corrected smoke removes the main evidence gap in E020 without changing the production gate: every guard attempt now stores generated IDs, completion masks, reference/new token log-probabilities, token KL values, variant labels, sample keys, and digests. Four backoff attempts were recorded, each with a nonzero trial parameter delta followed by exact policy/AdamW rollback. The production bfloat16 gate rejected all four attempts; no-autocast bfloat16 weights and a full-fp32 copy stayed below the `0.005` target and matched each other.

The final replay audit status is `TOKEN_REPLAY_VALIDATED` (24/24 rows, complete three-variant groups, same rollout/mask/reference, aggregate consistency, and rollback verified). A first audit output was retained as a diagnostic artifact after it rejected rows due only to Python float64 versus torch.float32 round-off; the v2 audit uses a documented `5e-7` absolute tolerance and passes the same persisted data. Thus the tested discrepancy is reproducibly tied to the bfloat16 autocast measurement path. It remains a measurement diagnostic, not proof that the optimizer update is causally unstable; formal RL expansion and optimizer/update-scale changes remain paused.

## MM-E022 / MM-F030 corrected gate trainability and remaining pre-step risk

The opt-in FP32/no-autocast post-step gate accepted two consecutive updates at the original learning-rate multiplier, with active means `0.000228583` and `0.000092137`. This falsifies the narrow hypothesis that the tested optimizer update necessarily exceeds the `0.005` post-step budget: under the replay-validated measurement path it does not. The first legacy bfloat16 shadow value (`0.267400`) would have rejected an update that the active FP32 gate accepted, confirming that the legacy post-step failure was a measurement-path artifact in this smoke.

The corrected run does not clear the whole RL precision path. Step 2's pre-step legacy measurement was `0.00259953`, versus `0.00000265427` from the FP32/no-autocast shadow, triggering `PRESTEP_PRECISION_DIVERGENCE`. Because the existing training loss and pre-step KL early-stop still consume the legacy path by design, extending training now could reintroduce precision-driven optimization or stopping behavior even though the post-step gate is corrected.

Checkpoint reload, policy/AdamW state continuity, replay linkage, and finite artifact checks all passed. No rollback occurred because both first attempts were accepted, so rejected-attempt rollback behavior remains covered by E017-E021 tests and artifacts rather than this smoke. The next diagnostic must isolate training-loss and pre-step KL precision on identical tokens before any 4-step run. No optimizer/update-scale, reward, checkpoint selection, or default-model change is justified.

## MM-E023 / MM-F031 training autocast precision attribution

The same-token replay closes the remaining E022 ambiguity for the tested KL-only updates. The production forward computes `new_log_probs` under bfloat16 autocast while `old_log_probs` and `ref_log_probs` come from no-autocast FP32 forwards. With zero group advantages, the GRPO policy term is exactly zero, so the observed loss and gradient arise only from the reference-KL penalty.

At step 1, FP32/no-autocast reproduced the policy/reference equality and yielded zero loss and zero gradient, while the legacy path yielded loss `2.44991e-6` and mean per-micro gradient norm `0.006985`. At step 2, the legacy loss was `5.19907e-5` versus `5.30854e-8` in FP32, and the gradient norm was `0.030846` versus `0.001049`. Replay arithmetic and shadow-gradient isolation passed for all four micro-batches.

The resulting status is `TRAINING_AUTOCAST_PRECISION_SENSITIVE`: in this smoke, the prior optimizer updates were largely reacting to precision-induced KL rather than the FP32 policy/reference difference. This does not establish behavior for nonzero advantages, ratio clipping, longer runs, or model quality. Do not tune learning rate or accumulation to compensate for the artifact. The next narrow fix should make training-forward precision explicit and opt-in, then validate nonzero-advantage math before one 2-step smoke.
## MM-E024 / MM-F032: active FP32 training-forward smoke

The prior E023 diagnostic showed that the unchanged bfloat16-autocast training loss and pre-step KL path was precision-sensitive on identical zero-advantage rollouts. E024 added an explicit opt-in `fp32_no_autocast` training-forward/loss mode while retaining the legacy default. The smoke also kept the active FP32 post-step KL gate and recorded active/legacy/FP32 variants with replay hashes and gradient-isolation checks.

Both steps passed the active FP32 gate at multiplier 1.0, and the active loss/KL/gradient fields matched the FP32 shadow. Replay, sample linkage, state continuity, checkpoint reload, and finite-value checks passed. The apparent no-op is important: all groups had zero advantage, so both policy parameter deltas were exactly zero. The optimizer path was exercised only as an accepted zero-gradient diagnostic; no claim about nonzero-advantage ratio clipping, update sensitivity, or quality is justified.

The offline nonzero-advantage contract test passed for both GRPO and CISPO diagnostic terms. This is a mathematical/API contract, not a model-run result. The correct next gate is a deterministic nonzero-advantage fixture or controlled data path, followed by a separately approved smoke. Do not expand to longer RL, CISPO, multiple seeds, benchmark evaluation, reward changes, optimizer tuning, or default-model replacement.

## MM-E025 / MM-F033 deterministic nonzero-advantage contract

The isolated contract fixture removes the specific E024 evidence gap: all four samples have nonzero group-relative advantages derived from rewards `[1, 0, 1, 0]`. Offline replay confirmed the GRPO clipped objective on 4 of 8 tokens and the CISPO clipped objective on 2 of 8 tokens. Both production and diagnostic loss/KL calculations matched, and both FP32 and bfloat16-quantized gradient norms were finite and nonzero.

This is a necessary implementation contract, not causal evidence about the model trainer. It does not show that real generated groups in a GPU run will receive nonzero advantages, nor does it establish update quality, stability, validation gain, or a reason to alter reward/optimizer settings. The earlier zero-advantage model smoke remains correctly classified as a no-op diagnostic, while this fixture is correctly classified as `NONZERO_ADVANTAGE_CONTRACT_PASS`. The next step, if approved, is one narrow active-path smoke with controlled nonzero reward input and explicit sample/advantage telemetry; formal RL expansion remains paused.

## MM-E026 / MM-F034 controlled active-path result

The controlled pattern `[1.0, 0.0]` supplied nonzero advantages to the live GRPO trainer. Four micro-batches and eight samples carried the controlled reward source; both optimizer steps were accepted by the active FP32/no-autocast KL gate at multiplier `1.0`, with nonzero parameter deltas and a reloaded step-2 checkpoint. This closes the specific live-trainer nonzero-advantage/update-path coverage gap left by E024.

It does not establish that the natural validator reward produces useful advantages, nor does it say anything about validation improvement, safety, termination, or reward correctness. The injected reward is a diagnostic stimulus only. Legacy bfloat16 shadow disagreement and pre-step precision divergence remain unresolved warnings, so no optimizer expansion or default-model change is justified. Status: `CONTROLLED_NONZERO_ADVANTAGE_ACTIVE_PATH_PASS_2_STEPS`, `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`.

## MM-E027 / MM-F035 natural reward result

The live trainer consumed `rule_reward` without the controlled override. All four micro-batches and eight samples recorded the expected natural source, and the active FP32/no-autocast gate accepted both steps at multiplier `1.0`. The corrected audit also verified replay linkage, state continuity, finite telemetry and checkpoint reload.

The observed reward vector was constant at `[0.1, 0.1]` for every two-generation group. It came from the termination component while validator reward was zero, which forced zero group-relative advantages and zero policy parameter deltas. Thus the run is `NATURAL_RULE_REWARD_ZERO_ADVANTAGE_DIAGNOSTIC`: natural reward plumbing is confirmed, but live nonzero-advantage training remains unobserved. The next action is an offline audit of reward component diversity and generated-group collapse, not an optimizer change or longer RL run.

## MM-E028 / MM-F036 natural reward diversity audit

The offline audit keeps current E027 evidence separate from older legacy-schema artifacts. In the current source, all 8 samples have the same `rule_reward=0.1`; all 4 observed groups collapse, validator coverage is zero, termination is the only nonzero component, and no group contains a nonzero reward spread. Therefore there is no live nonzero-advantage evidence to use for an optimizer or update-scale decision.

E010 v2 and E009 v1 artifacts show broader reward diversity and nonzero component coverage, but they are historical sources with different schemas/data and cannot establish that the current natural path will produce useful advantages. Status: `NATURAL_REWARD_DIVERSITY_AUDIT_COLLAPSE_CONFIRMED`, `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`. The next safe action is offline prompt/output and validator/termination coverage analysis; do not launch formal RL or modify reward/default weights from this result.

## MM-E029 / MM-F037 input/output coverage result

The Alignment v2 train manifest is structurally balanced: 128 rows, 8 categories, 16 rows per category, and 128/128 chosen records pass their corresponding validator. The E027 generation artifact is not balanced: it contains only 8 conciseness samples from 4 prompts and 4 families. All 8 generated responses fail `length_or_core_definition`, so the live run never observes a validator pass or within-group reward spread.

The two termination reasons are split evenly (`eos=4`, `max_new_tokens=4`), but the natural termination component is `0.1` for both. This means the current reward path is insensitive to that ending distinction. Deterministic empty/newline probes confirm the functions are not mathematically constant, but probes cannot substitute for model outputs. Status: `CURRENT_GENERATION_SIGNAL_COVERAGE_INSUFFICIENT_DIAGNOSTIC`, `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`. The next decision should concern balanced output coverage and validator observability, not optimizer tuning.
## MM-E030 / MM-F038：平衡覆盖 smoke 与当前结论

E030 使用确定性一行一类的独立 train/validation manifest，覆盖 `conciseness`、`format`、`instruction`、`reasoning`、`repetition`、`safety`、`termination`、`uncertainty` 八类。32 个生成样本实现了 8/8 category、8/8 family、8/8 prompt 覆盖，validator replay 与持久化 reward/component 均无不一致；因此 E029 的“只采到 conciseness”覆盖缺口已被修正。

但当前自然生成质量仍不足以支持训练结论：train validator pass 仅为 2/32，独立 balanced validation baseline 为 0/8。termination reward 已出现 `{0.0, 0.1}`，说明信号可观测，但这只是短 smoke 的信号覆盖，不是泛化或安全提升证据。两步均由 FP32/no-autocast active gate 接受，legacy bfloat16 shadow 仍有明显分歧，因此不能把本轮解释为 optimizer/update-scale 改进。

结论：`CURRENT_GENERATION_SIGNAL_VARIABILITY_OBSERVED_DIAGNOSTIC`，最终状态 `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`。下一步先裁定 validator/output quality 的离线审计；不启动 formal RL、CISPO、多 seed、C-Eval 或冻结集，也不修改 reward、optimizer 或默认模型。
## MM-E031 / MM-F039：balanced output-quality audit

E031/F039 对 E030 的 32 个生成样本做了逐样本 validator replay，并按 category、step、prompt 汇总长度、EOS/max-token、自然结束、空答、重复三元组和 reward component。8/8 category、family、prompt 覆盖完整，persisted reward 和 component replay 都是 0 mismatch，因此当前主要问题不是样本关联或 reward 重算不一致。

质量信号仍不足：validator pass 仅 2/32，失败原因分散；20/32 命中 `max_new_tokens`，自然结束只有 12/32。termination reward 已能观察到 0.0/0.1 的变化，说明信号路径可见，但不能把短 smoke 的输出质量当作泛化或训练收益。该结果不支持修改 reward、增加长度、调整 optimizer 或晋级 checkpoint。

结论为 `OUTPUT_QUALITY_SIGNAL_SPARSE_DIAGNOSTIC`，最终状态仍为 `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`。下一步如继续，先单独裁定质量/output 方案；不启动 formal RL、CISPO、多 seed、C-Eval 或冻结集。

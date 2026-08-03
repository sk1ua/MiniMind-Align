# MiniMind-Align 实验总报告

## 一句话结论

在同一组 100 条冻结测试 prompt 上，Alignment SFT v2 已明显改善自然结束和规则遵循；on-policy DPO v2 带来小幅稳定改进；SimPO 64-step pilot 在本地规则指标上最好，但继续训练到 256 steps 出现明显长度/重复坍缩。轻量 Reward Model 能学习部分 pair 偏好，但存在短回答偏置；GRPO/CISPO 只完成了规则奖励 tiny pilot，不能宣称收敛或算法优劣。

## 统一冻结测试

解码固定为 seed=42、`do_sample=false`、`max_new_tokens=160`、`repetition_penalty=1.15`、`no_repeat_ngram_size=3`，测试集为 `dataset/alignment_v1/splits/prompts_test.jsonl`，100 条。规则结果：

| 模型 | validator pass | natural end | 平均 tokens | repeat-3gram |
|---|---:|---:|---:|---:|
| full_sft | 0/100 | 1/100 | 159.09 | 0.10053 |
| align_sft_v1 | 19/100 | 57/100 | 104.44 | 0.14979 |
| align_sft_v2 | 50/100 | 88/100 | 64.92 | 0.06725 |
| LoRA v2 | 30/100 | 88/100 | 65.54 | 0.09029 |
| DPO v2 full-on-pilot | 52/100 | 93/100 | 61.19 | 0.05049 |
| SimPO v1 pilot | 64/100 | 93/100 | 56.23 | 0.04579 |
| SimPO v1 full | 58/100 | 99/100 | 34.26 | 0.03037 |

SimPO pilot 是当前冻结规则集的 validator leader；SimPO full 虽然更短、更少重复，但 repetition 类 pass 降为 1/13，因此不被静默选为最佳模型。

## 对齐税代理指标

独立 Alignment v2 validation（160 条）上的 loss/PPL：

| 模型 | loss | PPL |
|---|---:|---:|
| full_sft | 3.282972 | 26.6549 |
| align_sft_v1 | 2.644963 | 14.0829 |
| align_sft_v2 | 1.094926 | 2.9890 |
| LoRA v2 | 2.113108 | 8.2739 |

这只是内部 validation 的对齐税代理，不是公开 benchmark。图表位于 `results/experiments/unified_sft_v2_20260731/plots/`。

## Preference / Reward / RL

- on-policy：128 train + 32 validation prompts，每个 4 candidates，保存 640 条候选相关产物；DPO pair audit PASS。
- Gemini C003 smoke：32 条，validator chosen 胜 16、hard rejected 胜 5、tie 11；该 smoke 结果不外推为全量训练排名。
- Gemini C003 全量 on-policy 排名：训练集 128 条，validator chosen 胜 62、tie 48、hard rejected 胜 18，候选平均 overall 1.6328 vs 基线 0.9219，错误 0；验证集 32 条，validator chosen 胜 16、tie 12、hard rejected 胜 4，候选平均 overall 2.0938 vs 基线 1.1563，错误 0。两组结果均使用 Gemini 3.6 Flash、seed=42、独立 experiment id，原始逐条 judgment 和 SHA256 已归档。

### C003 Gemini 全量 on-policy 排名

训练集与验证集均来自已审计的 on-policy pair：同一 prompt 下 validator_chosen 对 hard_rejected，Gemini judgment 不回流训练。训练集 128 条的候选胜率（按全部样本）为 0.4844，验证集为 0.5000；这支持 validator 选择方向，但不能替代更大独立测试集或人工复核。详细命令、路径和 SHA256 见 docs/experiments/gemini_on_policy_full_ranking.md。
- DPO v2 validation pair accuracy：26/32 = 0.8125。
- Reward Model：64 steps，validation pair accuracy 20/32 = 0.625；与 Gemini 的 21 条 non-tie 一致率 12/21 = 0.5714；margin 与长度差 Pearson `-0.5057`。
- GRPO/CISPO lite：16 rule prompts，4 generations，4 optimizer steps；step 3 reward std=0，step 4 KL≈0.002536，属于不稳定 tiny pilot。
- GRPO/CISPO extended：8 类、32 条 validation prompts、8 generations、8 optimizer steps，使用独立输出目录。两者均在未参与 RL 训练的 100 条冻结测试上取得 validator pass 51/100；GRPO 的 natural end / 平均 tokens / repeat-3gram 为 90/64.98/0.06596，CISPO 为 91/63.91/0.05808。该结果只说明延长 run 可完成并暴露短程波动，不能证明收敛或算法优劣。

## Gemini 完整冻结集对照扩展

在相同 100 条冻结测试 prompt、相同 seed=42 和匿名 A/B 评审器下，补做了两组完整 Gemini 对照：

| 对照 | tie | candidate 胜 | align_sft_v2 胜 | candidate avg overall | 错误 |
|---|---:|---:|---:|---:|---:|
| align_sft_v2 vs DPO v2 full | 93 | 6 | 1 | 1.40 | 0 |
| align_sft_v2 vs SimPO pilot | 77 | 16 | 7 | 1.47 | 0 |
| align_sft_v2 vs SimPO full | 47 | 22 | 31 | 1.28 | 0 |

这三组结果说明：DPO 的 validator `50→52` 增量在 Gemini 主观评审中几乎全部表现为 tie；SimPO pilot 的 validator leader 地位得到较弱但可见的主观支持，不过仍只有 16/100 胜出，不能描述为泛化胜利。SimPO pilot 的 safety 类平均 overall 低于基线（1.15 vs 1.31），因此默认选择仍需保留安全性审查。

SimPO full 的 Gemini 结果进一步支持过度优化判断：虽然 termination 类略占优，但总体平均 overall 低于 align_sft_v2，且基线胜出 31/100、SimPO full 胜出 22/100；repetition 与 uncertainty 类也分别低于基线。它保留为负结果对照，不作为默认模型。

为检查评审顺序敏感性，在同一冻结 100 条测试集上使用 `seed=43` 重新匿名排列并独立评审：tie 47、align_sft_v2 胜 32、SimPO full 胜 21，平均 overall 为 1.32 vs 1.21。两轮合计 200 条 judgment 的胜负计数为 tie 94、align_sft_v2 胜 63、SimPO full 胜 43，逐样本胜者完全一致 81/100；这增强了负结果的评审稳定性，但仍属于同一生成集的复核，不等同于新的泛化测试。

逐条评审、summary、人工复核和 SHA256 位于 `results/experiments/gemini_align_vs_dpo_v2_full_retry_20260731/`、`results/experiments/gemini_align_vs_simpo_v1_pilot_20260731/`、`results/experiments/gemini_align_vs_simpo_v1_full_seed43_20260731/`；复核汇总为 `results/experiments/gemini_simpo_full_replicates_20260731/report.md`，汇总图为 `results/experiments/unified_sft_v2_20260731/plots/gemini_overall_comparison.png`。

## 事实、推断和假设

### 实验事实

事实均来自 `results/experiment_registry.jsonl`、独立 experiment wrapper 目录和上述 JSON summary；权重不会覆盖旧文件，所有 C-E 阶段均有独立 Git commit。

### 推断

SFT v2 的收益明显大于 DPO v2 的增量，说明本项目当前瓶颈更接近数据质量/基础能力与格式控制，而不是单纯偏好排序。SimPO full 的 pass 下降与长度极短共同支持“过度优化”推断，但不能单凭 100 条规则集证明泛化坍缩。

### 假设

更大的 on-policy family 覆盖、更多 Gemini/人工校验和带 KL early-stop 的 RL 可能改善结论；这些都未在本轮证实。

### 失败或未完成

初始 DPO/RM/RL 命令错误均保留失败目录；`av1_test_conciseness_0007` 旧 Gemini judgment 仍缺失，不补造；公开 benchmark、长程 RL 收敛均未完成，本轮 on-policy Gemini 全量 ranking 已完成但仍不是独立泛化测试。

## 2026-08-01 追加：MM-E009 / MM-F015 / MM-F016

RL 方法升级完成了 GRPO/CISPO 各 3 seeds、最多 16 steps、每 4 steps validation/checkpoint。三 seed 平均 validator pass 相对 baseline 的增量为 GRPO `+0.33/32`、CISPO `+1.00/32`，低于“平均至少 +3 pass”的改进门禁；两者 safety/termination 未下降，因此结论为不确定/负结果，默认模型不改变。详见 `docs/experiments/rl_method_upgrade_20260801.md`。

公开 C-Eval 代表子集完成 5 个请求科目、100 题、8 个模型的贪心评测。由于固定 revision 的实际配置差异，使用了显式记录的 val/dev 补足和 `business_ethics -> business_administration` alias；所有模型均为 12/100，invalid 44。它只能作为外部方向性证据，不是官方全量 benchmark 分数。详见 `docs/experiments/ceval_subset_20260801.md`。

## 2026-08-01 追加：MM-E010 / MM-F017

本轮将 RL 训练数据替换为原生 Alignment v2 programmatic 数据，并使用独立 validation slice，未修改 `align.rl_rules.rule_reward`、学习率或晋级规则。训练集从 `dataset/alignment_v2/manifests/train_manifest.jsonl` 按 `source=alignment_v2_programmatic_v1`、metadata 非空、seed=42 的确定性规则选择 8 类各 16 条，共 128 条；validation 使用 v2 manifest 各类原始第 5–8 条，共 32 条。train/validation ID 与 family overlap 均为 0，选择 metadata 和 source/output SHA256 保存在 `results/inputs/rl_data_isolation_selection_20260801.json`。

初始 data-isolation run 保留在 `results/experiments/rl_data_isolation_20260801/`：训练进程内存中的 selected 指标为 14/32，但独立回载诊断显示保存后的 artifact 为 13/32。该结果不再用于最终 checkpoint selection，旧目录和结论原样保留，避免把内存指标误当成可复现 artifact 指标。

随后修复 `trainer/train_grpo_lite.py`：每个 checkpoint 写入后立即回载，并以回载模型计算 validation、quality gate 和 selection；中间无 checkpoint 的评测明确标为 `in_memory_policy`。1-step smoke 与独立 `max_steps=0` 回载评测的稳定输出一致，18 项相关测试通过。

### checkpoint 回载修复后的正式复跑

修复后的独立实验根目录为 `results/experiments/rl_data_isolation_reload_fixed_20260801/`，运行代码 commit 为 `14076033f25fc7dfa35403f2d7beccb46ae43d5c`。GRPO/CISPO 各运行 seed 42/43/44，最多 32 steps、每 4 steps validation/checkpoint；六个 run 均 exit code 0。GRPO 三个 run 均完成 20 steps 后因 reference KL 连续两步超过 0.005 早停，CISPO seed 42/43 完成 20 steps、seed 44 完成 18 steps 后同样早停；六个 run 的 selected checkpoint 均为回载后的 step 4。

修正版 selected artifact 的独立 validation 均为 13/32，safety 与 termination 均为 4/4，natural end 均为 32/32。GRPO/CISPO 三 seed 均值均为 13.0、总体标准差均为 0，相对 baseline 13/32 的增量为 0；因此最终 gate 为 `NOT_MET_NO_MODEL_CHANGE`，默认模型和旧实验结果不改变。

reward-hacking audit 只作诊断、不改变 checkpoint 选择：六个 run 均报告训练 reward/validator 上升但独立 validation 不上升，以及 max-length hit 和 repetition penalty 增加 warning；没有 empty-response 或 validation safety/termination/natural-end 下降 warning。部分训练 step 的 max-length hit 最高约 98.4%，这是风险证据，不能解释为质量提升。

正式冻结集 validator 评测覆盖 baseline 和六个唯一 selected checkpoint（每个 100 条）：baseline 50/100；GRPO 三个 seed 均 51/100；CISPO seed 42/43/44 分别为 50/50/51。GRPO 均值为 51.0（标准差 0），CISPO 均值为 50.33（总体标准差约 0.47）。该冻结集只作泛化证据，不参与 checkpoint 选择；本阶段不重跑 C-Eval。

完整 v2 validation 覆盖评测仍显示 baseline 与六个回载 checkpoint 均为 47/160，预注册 32 条切片均为 13/32；它是诊断证据，不参与晋级门禁。所有新 run、audit、frozen generation/score 和资源日志均位于上述独立实验根目录，旧目录未覆盖。详细结果见 `docs/experiments/rl_data_isolation_reload_fixed_20260801.md`。

## 2026-08-01 追加：MM-E011 / MM-F018 RL 稳定性诊断

在不改变 reward、KL 门禁、checkpoint 选择或晋级规则的前提下，补充了 GRPO/CISPO seed=42 的三条件对照：control、low_lr 和 accum16。训练器新增 loss、pre/post-clip 梯度范数、ratio 分位数、micro-batch KL 分布和 `pre_optimizer_step` 测量阶段记录；新增审计器只生成诊断 warning，不改变 checkpoint eligibility。

六个正式 run 均 exit code 0，但全部在 reference KL 连续两步超过 `0.005` 后早停：control 和 low_lr 在 step 20，accum16 在 step 10。保存后回载的 selected checkpoint 均为 step 4；独立 validation 均为 13/32，safety/termination 为 4/4，natural end 为 32/32。accum16 没有达到“触发至少延后 4 步”，low_lr 触发时间没有延后，且两者的 KL 最大值或梯度最大值至少一项高于 control；因此没有稳定性改善条件。

正式 GPU wall time 合计 1125 秒，smoke 为 13 秒，低于 7200 秒硬上限；GPU 最终 0 MiB，服务器保持 `RUNNING`。审计 JSON/JSONL 全部 finite，完整结果、checkpoint、资源日志和审计位于 `results/experiments/rl_stability_diagnostic_20260801/`，专用说明见 `docs/experiments/rl_stability_diagnostic_20260801.md`。本轮不重跑 C-Eval/冻结集，不改变默认模型，最终状态为 `NOT_MET_NO_MODEL_CHANGE`。

## 2026-08-01 追加：MM-F019 RL KL 尖峰与 reward 传导离线归因

在六个稳定性对照 run 上新增 step/sample-level forensic audit，没有启动 GPU。六个 run 的训练 validator 峰值约为 0.8594–0.9062，但保存并回载后的 selected validation 均为 13/32；训练 reward/validator 增益没有传导到独立 validation。六个 run 均出现 KL max/P95 尾部集中和梯度裁剪，最高 reference KL 约 2.77；同时观察到截断和重复惩罚上升，未观察到 empty-response 或 safety/termination 下降。

这支持“高 KL 尾部与裁剪梯度共同出现”的诊断推断，但不能作为因果证明。现有 artifact 没有保存每个 micro-batch 的 log-prob 张量或梯度向量，精确 source attribution 仍未完成。因此当前裁定为 `DIAGNOSTIC_ONLY_WITH_SIGNALS`，不改变 `NOT_MET_NO_MODEL_CHANGE`，继续暂停正式三 seed RL 扩展。详细输出见 `results/experiments/rl_stability_diagnostic_20260801/spike_source_audit_v2/` 和 `docs/experiments/rl_spike_source_forensics_20260801.md`。

## 2026-08-01 追加：MM-E012 / MM-F020 micro-batch 尖峰归因诊断

### 结论

补齐 micro-batch 可回放 telemetry 后，使用 GRPO control、seed=42、4 steps、32 个 micro-batch，并通过 `interleave_categories=true` 覆盖 Alignment v2 隔离训练集的八类 prompt。离线审计结果为 `BROAD_SPIKE_DIAGNOSTIC`：异常没有被可靠地收敛到单一 category 或单一 prompt；`termination` 同时出现在 top-3 KL 与 top-3 梯度的类别交集，但精确 top-3 micro-batch key 交集为 0，不能据此声称 termination 是因果来源。

| 项目 | balanced formal 事实 |
| --- | ---: |
| steps / micro-batches | 4 / 32 |
| categories | 8/8，均有 4 个 micro-batch |
| baseline validation | 13/32 |
| selected checkpoint validation | 13/32 |
| safety / termination | 4/4 / 4/4 |
| KL mean 超过 0.005 | step 3 |
| KL max | 0.90024 |
| KL P95 最大值 | 0.02795 |
| pre-clip 梯度范数 | 1.50–2.05（每步均发生裁剪） |
| GPU wall time | 71 秒 |

step 2 已出现 KL max 0.84002，step 3 的 KL mean 为 0.00705 且 KL max 为 0.90024；step 4 KL mean 回到 0.00428，但 KL P95 仍为 0.02795。训练结束后 checkpoint reload 和 selection 均通过，未触发连续两步 KL early stop。完整审计位于 `results/experiments/rl_microbatch_telemetry_20260801/audit_balanced_v2/`。

### 解释边界与后续裁定

- 第一次 4-step formal run 因原始 manifest 的类别块顺序只覆盖了 `conciseness`/`format`；该目录和日志保留，但标记为 coverage-incomplete，不用于最终来源判定。
- smoke 仅用于验证写入、回载和审计管线；它的少量 prompt 不能作为类别来源结论。
- balanced run 的 telemetry 能区分“局部单类集中”与“跨类广泛异常”，但不包含完整 token-level causal intervention，因此 `BROAD_SPIKE_DIAGNOSTIC` 仍是诊断而非因果证明。
- baseline 与 selected validation 没有变化，模型晋级门禁仍为 `NOT_MET_NO_MODEL_CHANGE`；默认模型、既有权重、旧 manifest、C-Eval 结果均未改变。

因此暂停 CISPO、三 seed 扩展、C-Eval 和冻结集评测。下一轮应设计独立的学习率、累积步数或更新尺度对照，并继续保留当前 KL、质量和模型晋级门禁。

## 2026-08-01 追加：MM-E013 / MM-F021 更新尺度稳定性对照

### 技术摘要

在保留 reward、KL 门禁、checkpoint selection 和默认模型不变的前提下，新增 GRPO seed=42 的三条件短程对照：control、low learning rate 和 `max_grad_norm=0.5`。每个 formal run 为 8 steps、64 个 micro-batch，八类 prompt 交错覆盖；三者 selected validation 都是 13/32，safety/termination 都是 4/4。完整 stability audit（修正后纳入 `clip_half`）没有条件满足稳定性改善定义，最终状态为 `NOT_MET_NO_MODEL_CHANGE`。

| 条件 | 学习率 | max grad norm | max KL mean | max KL P95 | max KL | selected validation | GPU 秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 3e-7 | 1.0 | 0.00705 | 0.02795 | 0.90024 | 13/32 | 119 |
| low_lr | 1e-7 | 1.0 | 0.00696 | 0.02756 | 1.59414 | 13/32 | 119 |
| clip_half | 3e-7 | 0.5 | 0.00705 | 0.02566 | 0.90066 | 13/32 | 118 |

### 结果与解释

- 三个条件都在 8 steps 内完成，没有连续两步 KL early-stop；因此相对 control 没有达到“KL 触发延后至少 4 steps”的稳定性改善条件，且不能把“未触发”解释为已证明稳定。
- `clip_half` 确实把每步 post-clip 梯度范数限制在约 0.5，但 pre-clip 梯度峰值仍约 2.12，KL max 仍约 0.90；它只改变了裁剪后的更新尺度，没有消除尖峰。
- `low_lr` 的 KL P95 略低于 control，但 KL max 升至约 1.59；按照预先定义的“最大值不高于 control”门槛，不能标记为改善。
- 三条件的独立 validation、safety、termination 和 natural end 没有差异，未出现模型质量下降门槛触发，也没有任何条件满足完整稳定性规则。

逐条件 micro-batch audit 均给出 `SOURCE_LOCALIZED_DIAGNOSTIC`，主要类别交集为 `conciseness`、`format` 和 `termination`；这是基于 top-K 聚合的诊断启发式，不是 token-level 因果证明。结果支持“局部 prompt/category 集合反复出现异常”的工作假设，但还不足以决定修改数据或 reward。

本轮 GPU wall time 为 smoke 13 秒加 formal 356 秒，共 369 秒；34 个 JSON、2182 条 JSONL 记录 finite，三个 checkpoint 均可回载。权威输出位于 `results/experiments/rl_update_scale_diagnostic_20260801/formal/stability_audit_v2/` 与对应 `spike_audit_*` 目录。

### 后续裁定

当前证据不支持扩展三 seed 或替换默认模型。下一步优先对重复出现的 `conciseness`/`format`/`termination` prompt 做输入与 reward 组件审计，并保留低学习率和更严格裁剪为诊断对照；任何模型变更仍必须重新通过三 seed 晋级门禁。

## 2026-08-01 MM-E014 / MM-F022: prompt and reward-component audit

The offline audit consumed the three completed update-scale formal runs (`grpo_control_seed42`, `grpo_low_lr_seed42`, and `grpo_clip_half_seed42`). Each run contributed 64 micro-batches and 512 samples; every declared sample key linked successfully. No GPU work was performed.

Using top-K=10 for the union of KL-max, unscaled-gradient, and quality-anomaly tails, 26 prompt identifiers and 5 categories recurred across at least two conditions. The strongest category-level diagnostic signals were:

| category | recurring signal | interpretation |
|---|---|---|
| termination | 15 KL-tail hits and 15 gradient-tail hits | repeated association with update spikes |
| format | KL, gradient, and quality-tail hits | mixed optimization and output-quality signal |
| conciseness | 9 gradient-tail hits and 3 KL-tail hits | mainly gradient-associated in this audit |
| reasoning | 21 quality-tail hits | mainly truncation/natural-end quality signal |
| instruction | 9 KL-tail and 3 quality-tail hits | KL and quality association |

The descriptive component correlations over 192 micro-batch aggregates do not establish causality. Validator reward had near-zero correlation with KL max (`r=0.0202`) and a positive correlation with reward mean (`r=0.9545`); termination reward had near-zero correlation with KL max (`r=0.0309`); repetition penalty had weak negative correlation with KL max (`r=-0.0841`) and gradient norm (`r=-0.0925`). `parse_reward` and `field_reward` were zero in this persisted sample slice, so they provide no variation for correlation. The result is therefore `RECURRING_PROMPT_DIAGNOSTIC`, not a model-improvement claim.

Artifacts are under `results/experiments/rl_prompt_reward_component_audit_20260801/`. The top-K rule is a localization heuristic over persisted aggregate telemetry; it is not token-level replay or causal proof. The default model, checkpoint selection, and formal seed-expansion plan remain unchanged.

## 2026-08-01 MM-E015 / MM-F023: corrected termination telemetry and v2 input/reward audit

### Technical summary

The generation-end telemetry correction is implemented in `trainer/train_grpo_lite.py`. EOS position now determines natural completion; only EOS-free generations reaching `max_gen_len` are marked `max_new_tokens`; shorter EOS-free generations are retained as `no_eos_short_generation`. Padding width no longer changes the classification. The offline v2 audit passed the input/reward consistency gates: source chosen validator `128/128`, replay agreement `1536/1536`, metadata missing `0`, and GPU wall time `0` seconds.

### Key findings

- The audit status is `CROSS_CONDITION_PROMPT_DIAGNOSTIC`: 18 prompt groups and 4 category groups occur in top-K tails across conditions. This is cross-condition recurrence, not within-run repeated sampling.
- The prior runs predate the corrected fields, so their quality evidence is explicitly `LEGACY_QUALITY_TELEMETRY_UNTRUSTED`; old `max_length_hit` is not used for a new truncation/quality-tail claim.
- Validator replay and persisted `validator_reward` agree for all 1,536 samples. No reward-input/validator mismatch was found, and the reward formula, KL gate, checkpoint selection, and default model are unchanged.

### Scope, method, and limitations

The audit reads the three completed update-scale formal runs and `results/inputs/rl_data_isolation_train_128_20260801.jsonl`. It records within-run occurrence counts separately from condition counts, adds category/family exposure denominators, replays the shared validator, and reports reward-component nonzero coverage. The result is descriptive and diagnostic; it does not establish causality or justify model promotion.

### Decision and next step

Keep `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`. The old quality-tail conclusion is downgraded to a telemetry artifact until corrected runs exist. Only after this offline audit should a corrected GRPO control smoke be considered; CISPO, three-seed expansion, C-Eval, frozen-set reruns, and reward changes remain paused.

## Corrected GRPO control smoke: telemetry behavior is internally consistent

One isolated GRPO control smoke was run after the offline audit: seed 42, 2 steps, 8 generated samples, validation on 2 prompts, and `max_gen_len=16`. It exited with code 0 after 13 GPU wall seconds. The checkpoint was reloaded successfully and all JSON/JSONL and checkpoint tensors were finite.

The new termination fields behaved as intended: 4 samples ended with EOS, 4 reached `max_new_tokens`, and 0 were `no_eos_short_generation`; the unknown rate was `0.0`. This validates the corrected telemetry path, but the smoke is too small to support an optimization or quality claim: validation remained `0/2`, and no model gate changed.

The smoke output is isolated at `results/experiments/rl_corrected_telemetry_smoke_20260801/`. The next decision remains diagnostic-only: if a larger corrected run is proposed later, it must be separately justified and must preserve the existing KL/checkpoint/promotion rules.

## 2026-08-02 MM-E016 / MM-F024: corrected balanced GRPO spike diagnostic

After the corrected telemetry smoke passed, one separately scoped GRPO control run was executed with seed 42. It used the isolated Alignment v2 train/validation manifests, interleaved all eight categories, and wrote to `results/experiments/rl_corrected_balanced_diagnostic_20260802/`. No CISPO, additional seeds, C-Eval, frozen-set evaluation, reward change, or default-model replacement was performed.

| metric | result |
| --- | ---: |
| steps / micro-batches / samples | 4 / 32 / 256 |
| category coverage | 8/8; 4 micro-batches per category |
| baseline / selected validation | 13/32 / 13/32 |
| safety / termination | 4/4 / 4/4 |
| termination counts | EOS 250; max_new_tokens 6; unknown 0 |
| maximum KL mean / P95 / max | 0.00705 / 0.02795 / 0.90024 |
| gradient clipping | active at all 4 steps |
| GPU wall time | 80 seconds |

The offline spike audit returned `BROAD_SPIKE_DIAGNOSTIC`. KL and gradient signals were not confined to one category or one prompt; `termination` appeared in the top-3 category intersection, but the exact top-3 micro-batch intersection was zero. This is aggregate diagnostic evidence, not causal proof. Step 4 validation explicitly used a reloaded checkpoint, and all JSON/JSONL artifacts were finite and parseable.

The corrected run therefore does not resolve the optimizer/update-scale issue and does not satisfy any model-promotion condition. The project remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`; formal CISPO, three-seed expansion, C-Eval, and frozen-set reruns remain paused.

## 2026-08-02 MM-E018 / MM-F026: KL guard telemetry smoke

The E018 implementation added per-attempt bfloat16/float32 KL, parameter delta metrics, deterministic policy/AdamW state digests, and rollback verification. Production acceptance remains the existing bfloat16 post-step KL mean gate at `0.005`; float32 is diagnostic-only. The new smoke root is isolated at `results/experiments/rl_kl_guard_telemetry_v2_20260802/` and does not modify the E017/F025 root.

The single GRPO seed-42 smoke started with the specified 2-step/8-prompt/2-generation configuration but stopped before the first optimizer attempt was logged. The initial implementation attempted to byte-view a scalar AdamW `step` tensor while computing the optimizer digest, which raised a runtime error. The run exited with code `1` after `13` GPU wall seconds; its baseline validation and generated sample logs were preserved, no checkpoint was written, and the GPU returned to idle.

The scalar-digest bug was fixed offline by flattening tensors before byte hashing, and a regression test was added. After the fix, `py_compile`, `bash -n`, the new 7 tests, and the full 64-test suite passed. Per the smoke protocol, the failed smoke was not automatically rerun. The offline audit therefore returns `TELEMETRY_INCOMPLETE` because `selection.json`, `step_summaries.jsonl`, and `kl_guard_attempts.jsonl` were never produced. No dtype or update-sensitivity conclusion is claimed.

The final state remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`. No reward, default model, checkpoint-selection rule, C-Eval result, or frozen-set result changed. A future smoke rerun requires a separate explicit decision.

## 2026-08-02 MM-E017 / MM-F025: KL trust-region guard diagnostic

### Scope and implementation

The optional post-update KL guard was added to `trainer/train_grpo_lite.py` and kept disabled by default. When enabled, it snapshots the policy and AdamW state, measures the reference KL over all rollouts in the optimizer step, and retries learning-rate multipliers `1.0`, `0.5`, `0.25`, and `0.125`. Only the full-step post-update KL mean is a hard acceptance gate at `0.005`; P95 and maximum are recorded diagnostically. An unresolved guard restores the pre-step policy/optimizer state, writes the step telemetry, skips checkpoint creation, and stops the run.

The isolated implementation and audit are under `results/experiments/rl_kl_guard_diagnostic_20260802/`. No reward formula, pre-step KL early-stop, checkpoint selection, or default weight was changed.

### Results

| run | requested / completed steps | pre-step KL mean | post-step KL mean / P95 / max | attempts / backoffs | optimizer step | checkpoint |
|---|---:|---:|---:|---:|---|---|
| smoke, first | 2 / 1 | 0.000122 | 0.26708 / 1.30190 / 5.21149 | 4 / 3 | rejected | none |
| smoke, retry | 2 / 1 | 0.000122 | 0.26708 / 1.30190 / 5.21149 | 4 / 3 | rejected | none |
| formal | 4 / 1 | 0.000104 | 0.01638 / 0.05595 / 2.97454 | 4 / 3 | rejected | none |

The formal audit status is `GUARD_UNRESOLVED_BASELINE_RETAINED`: baseline validation remained `13/32`, selected validation remained `13/32`, safety and termination remained `4/4`, and no accepted optimizer step or new checkpoint existed. The guard therefore did not produce a candidate model. The existing unguarded corrected control is included only as a descriptive reference (`max` KL mean `0.00705`, P95 `0.02795`, max `0.90024`), not as a promotion comparison.

Two smoke attempts and the formal run consumed `49` GPU wall seconds against the `3600` second hard limit. All `29` JSON/JSONL artifacts and `248` parsed records were finite, sample linkage was complete, and the server remained `RUNNING` with the L4 at `0 MiB` and approximately `92 GiB` disk available.

### Decision

The guard correctly detected that the current update cannot satisfy the `0.005` post-step mean-KL budget even after three backoffs. This is an update-scale diagnostic, not evidence of model improvement. The final state remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`; the baseline is retained and formal CISPO, three-seed expansion, C-Eval, frozen-set evaluation, and default-model replacement remain paused.

## 2026-08-02 MM-E018 / MM-F026 corrected retry and dtype attribution

The corrected retry used a new isolated root, the fixed digest implementation, GRPO seed 42, and the original two-step smoke configuration. It completed one optimizer-step attempt before the guard stopped the run: four attempts were recorded at LR multipliers `1.0`, `0.5`, `0.25`, and `0.125`; all were rejected by the production bfloat16 mean-KL gate; no checkpoint was written.

| metric | result |
|---|---:|
| GPU wall time | 30 seconds |
| bfloat16 post-step KL mean | 0.267079 |
| float32 post-step KL mean | 0.00000373 |
| dtype-sensitive attempts | 4/4 |
| gate disagreement | 4/4: bfloat16 reject, float32 accept |
| parameter delta L2, first attempt | 0.002126 |
| rollback verification | 4/4 exact digest matches; zero parameter delta after rollback |
| accepted optimizer steps / checkpoints | 0 / 0 |

The audit status is `BF16_MEASUREMENT_SENSITIVE`: the same rollout, completion mask, and reference log-prob produced a large bfloat16/float32 gap on every attempt, while float32 stayed below the `0.005` target. This is evidence that the observed guard rejection is dominated by KL measurement dtype sensitivity in this smoke; it is not evidence of model improvement or a reason to select a new optimizer/update-scale intervention. The default model, reward, gate policy, and checkpoint selection remain unchanged. Before any optimizer/update-scale experiment, the next approved step should independently validate or correct the post-step KL measurement semantics.

## 2026-08-02 MM-E019 / MM-F027: true-fp32-copy measurement precision diagnostic

The follow-up diagnostic added an optional detached float32 policy copy. It did not participate in backpropagation, optimizer state, checkpoint selection, the production bfloat16 gate, reward, or model promotion. The smoke used the same GRPO seed-42 protocol in a new root and recorded three measurement variants on the same rollout and completion mask:

| variant | post-step KL behavior |
|---|---|
| policy bfloat16 + bfloat16 autocast | mean `0.267400–0.275372`, gate reject on 4/4 attempts |
| policy bfloat16 + autocast disabled | mean `0.00000373–0.00022858`, gate pass on 4/4 |
| detached policy float32 + autocast disabled | exactly equal to the preceding variant on 4/4 |

Parameter deltas were nonzero before each attempted update, while all four rollback policy/AdamW digests matched exactly and no optimizer step or checkpoint was retained. The dedicated audit status is `BF16_AUTOCAST_SENSITIVE`. This localizes the discrepancy to the bfloat16 autocast measurement path rather than bfloat16 weight storage or an optimizer/update-scale sensitivity. The next intervention is therefore measurement correction/validation; no new optimizer/update-scale setting is selected, and the default model remains unchanged.

## 2026-08-02 MM-E020 / MM-F028: independent reference-KL semantics audit

An offline audit was added and run against the E019 artifact. It did not load a model or expose CUDA. Independent pure-Python fixtures confirmed the contract 'exp(ref_log_prob - new_log_prob) - (ref_log_prob - new_log_prob) - 1', completion-mask-only selection, and token-weighted aggregation across all rollouts and micro-batches.

The source-path checks passed: reference log-probabilities come from the frozen reference model, the three measurement variants use the same guard rollouts, the production gate still reads the BF16 variant, and all four persisted attempts agree with the recorded gate and rollback fields. BF16 rejected 4/4 attempts, true FP32 passed 4/4, and no-autocast matched true FP32 4/4.

The result is REFERENCE_KL_SEMANTICS_CONSISTENT_LIMITED, not token-level replay: E019 stored aggregate KL summaries rather than token-level log-probabilities and masks. GPU wall time was 0 seconds. No reward, optimizer/update-scale, checkpoint selection, default model, or prior experiment was changed. The next authorized experiment, if any, is a corrected no-autocast KL smoke that adds token-level replay telemetry; formal RL expansion remains paused.

## 2026-08-02 MM-E021 / MM-F029: corrected KL token replay smoke

One GRPO seed-42 smoke used the isolated Alignment v2 train/validation manifests and added token-level replay for every guard attempt. The run completed one training step, exhausted the four backoff multipliers (`1.0`, `0.5`, `0.25`, `0.125`), rejected the production bfloat16 gate on all four attempts, accepted zero optimizer steps, wrote zero checkpoints, and retained the baseline after exact policy/AdamW rollback.

The run persisted 24 replay rows: four attempts × two rollout micro-batches × three variants (`bfloat16_autocast`, `bfloat16_no_autocast`, `full_float32_no_autocast`). The independent v2 audit validated token arithmetic, completion masks, same-rollout/reference alignment, aggregate KL summaries, sample linkage, finite JSON, and rollback evidence. The final audit is `TOKEN_REPLAY_VALIDATED`; an earlier audit directory records a non-destructive false negative caused by an overly strict Python-float tolerance for torch.float32 serialization and is not used for the final decision.

Production bfloat16 post-step KL means were `0.267400`, `0.275372`, `0.268694`, and `0.267079`; no-autocast and detached full-fp32 means were identical and ranged from `0.00000373` to `0.00022858`. This reproduces the measurement-path discrepancy in a replayable artifact. It is diagnostic evidence, not a causal proof and not a reason to change optimizer/update-scale, reward, checkpoint selection, or the default model. Project state remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`.

## 2026-08-02 MM-E022 / MM-F030: corrected KL gate trainability smoke

The trainer now exposes an opt-in `fp32_no_autocast` post-step gate while retaining `legacy_bfloat16_autocast` as the default. Optimizer updates still use the existing bfloat16 training path; reward, loss, pre-step early-stop, checkpoint selection, and the default model are unchanged. The bfloat16 post-step measurement remains a shadow field, and detached full-fp32 plus token replay remain diagnostic-only.

The only authorized GRPO seed-42 smoke completed both optimizer steps without backoff. The active FP32 gate means were `0.000228583` and `0.000092137`, both below the `0.005` target at LR multiplier `1.0`. Step 1 showed the expected shadow disagreement: legacy bfloat16 reported `0.267400` and would have rejected the same update. Both accepted updates had nonzero parameter deltas, policy/AdamW state continuity passed from step 1 to step 2, and the step-2 checkpoint was independently reloaded.

| evidence | result |
|---|---|
| optimizer steps | 2/2 accepted |
| backoffs | 0; accepted multipliers `1.0`, `1.0` |
| active FP32 post-step KL means | `0.000228583`, `0.000092137` |
| legacy bfloat16 post-step KL means | `0.267400`, `0.000936847` |
| token replay | 12 rows; groups/source/attempt linkage complete |
| state continuity | policy and AdamW digests continuous |
| checkpoint | step 2, SHA-256 `43130481841fe5c8601632bc6c557c117db20f65e35f66d6d2c6dcf337681238`, reload passed |
| artifact checks | JSON/JSONL parsed; no non-finite values |
| audit | `CORRECTED_GATE_ACCEPTED_2_STEPS_DIAGNOSTIC` |

The audit also emitted `PRESTEP_PRECISION_DIVERGENCE`: at step 2, the unchanged legacy pre-step path reported mean KL `0.00259953`, while the FP32 no-autocast shadow reported `0.00000265427`. Therefore the post-step trainability question is answered, but the next stage is not a 4-step expansion. Training-loss and pre-step KL precision must be audited first. The two validation prompts only verified plumbing (`0/2`) and support no safety, generalization, or model-quality claim. Final status remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`.

## 2026-08-02 MM-E023 / MM-F031: same-token pre-step precision smoke

An opt-in diagnostic now evaluates the unchanged bfloat16-autocast training loss and an FP32/no-autocast shadow on identical generated tokens, completion masks, old/reference log-probabilities, and advantages. The shadow gradient is computed with `torch.autograd.grad` and verified not to mutate accumulated production gradients. The production loss, optimizer update, pre-step early-stop, reward, post-step gate, checkpoint selection, and default model remain unchanged.

The single GRPO seed-42 smoke completed 2/2 accepted updates and produced four pre-step replay rows plus twelve post-step replay rows. All replay arithmetic, hashes, sample links, state continuity, finite-value checks, and the step-2 checkpoint reload passed.

| step | bfloat16 loss | FP32 shadow loss | bfloat16 KL | FP32 KL | bfloat16 / FP32 gradient norm |
|---:|---:|---:|---:|---:|---:|
| 1 | `2.44991e-6` | `0` | `1.22495e-4` | `0` | `0.006985 / 0` |
| 2 | `5.19907e-5` | `5.30854e-8` | `0.00259953` | `2.65427e-6` | `0.030846 / 0.001049` |

At step 2 the legacy loss was `979.38×` the FP32 shadow and the gradient norm was `29.41×`; both step-2 micro-batches crossed the loss and KL disagreement thresholds, and all four micro-batches crossed the gradient threshold. The audit status is `TRAINING_AUTOCAST_PRECISION_SENSITIVE`.

All sampled groups had zero advantage, so this result cleanly isolates the KL-loss path but does not validate nonzero-advantage policy ratios or clipping. It is diagnostic evidence for correcting the training-forward precision path, not evidence of model improvement. The next intervention should be an opt-in FP32/no-autocast training-forward/loss mode with a nonzero-advantage contract test and at most one new 2-step smoke. No 4-step expansion is authorized by this result.
## 2026-08-02 MM-E024 / MM-F032: opt-in FP32 training-forward smoke

The trainer now exposes `--training-forward-mode` with the unchanged `legacy_bfloat16_autocast` default and an explicit `fp32_no_autocast` mode. The active mode is recorded separately from legacy bfloat16 and FP32 shadow loss/KL/gradient telemetry; reward, pre-step KL early-stop, post-step gate semantics, checkpoint selection, and the default model remain unchanged unless the new flag is explicitly supplied.

The isolated GRPO seed-42 smoke used two steps, eight prompts, two generations, two accumulation steps, the Alignment v2 isolated manifests, and the FP32/no-autocast post-step gate. It completed 2/2 accepted optimizer-step records, produced four pre-step replay rows and twelve post-step replay rows, preserved sample linkage and state continuity, and successfully reloaded the step-2 checkpoint. Active loss/KL/gradient fields matched the FP32 shadow path.

All sampled groups had zero advantage. Consequently policy parameter delta was zero on both steps; this is an accepted no-op diagnostic, not evidence of a real policy update. The nonzero-advantage contract test passed offline, but the smoke does not validate nonzero-advantage ratio clipping or model quality. Final status is `FP32_TRAINING_FORWARD_ACCEPTED_2_STEPS_DIAGNOSTIC` under `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`; no 4-step, CISPO, multi-seed, C-Eval, frozen-set, reward, optimizer, or default-model change is authorized.

## 2026-08-02 MM-E025 / MM-F033: deterministic nonzero-advantage contract

The zero-advantage limitation in MM-E024 was addressed with an isolated, deterministic offline fixture. It uses four samples with rewards `[1, 0, 1, 0]`, two generations per group, nonzero group-relative advantages, and eight masked response tokens. The fixture is stored at `results/inputs/rl_nonzero_advantage_contract_fixture_20260802.json` with SHA-256 `132257c5f3d8db35a858b079029e454f9777a793e5a86a1018bef2117a673a95`.

The contract audit replayed both loss variants without loading a model or exposing CUDA:

| variant | clipped tokens | clipping observed | FP32 gradient norm | bfloat16-quantized gradient norm |
|---|---:|---|---:|---:|
| GRPO | 4/8 | yes | 0.250776 | 0.250750 |
| CISPO | 2/8 | yes | 0.947565 | 0.947653 |

Production and diagnostic loss/KL calculations matched, both gradient paths were finite and nonzero, and the legacy/FP32 active-mode contract remained explicit. The audit status is `NONZERO_ADVANTAGE_CONTRACT_PASS`. This closes the mathematical and API coverage gap left by the zero-advantage smoke; it is not a model-training, causal, validation, or quality result. GPU wall time was zero, the default model and all previous experiment roots were unchanged, and the project remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`.

Before any longer run, approve one narrow deterministic nonzero-advantage smoke that exercises the active trainer path. Do not expand to formal RL, CISPO, multi-seed, C-Eval, frozen-set evaluation, reward/optimizer changes, or default-model selection from this offline result alone.

## 2026-08-02 MM-E026 / MM-F034: controlled nonzero-advantage active-path smoke

The trainer received a default-off `--controlled-reward-pattern` diagnostic hook. With pattern `[1.0, 0.0]`, GRPO seed 42 completed 2/2 optimizer steps under the explicit `fp32_no_autocast` training-forward and post-step gate. All four micro-batches recorded nonzero group-relative advantages, both steps were accepted at learning-rate multiplier `1.0`, parameter deltas were nonzero, and the step-2 checkpoint reloaded with continuous policy/AdamW state digests.

The corrected offline audit status is `CONTROLLED_NONZERO_ADVANTAGE_ACTIVE_PATH_PASS_2_STEPS`. Active post-step KL means were `0.000236967` and `0.0000810903`; GPU wall time was `26` seconds of the `1800` second limit. The first automatic audit remained preserved as `CONTROLLED_NONZERO_ADVANTAGE_PARTIAL_DIAGNOSTIC`; its step-level field assumption was corrected offline and the audit was rerun without another GPU task.

This is an implementation and telemetry diagnostic, not a quality result: the reward pattern was injected and validation was only `0/2` plumbing. Legacy bfloat16 shadow disagreement and pre-step precision divergence remain warnings. The checkpoint is not promoted, the default model and prior experiment roots are unchanged, and the project remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`.

## 2026-08-02 MM-E027 / MM-F035: natural rule-reward corrected smoke

The isolated follow-up removed the controlled reward override while retaining the explicit `fp32_no_autocast` training-forward and post-step gate. GRPO seed 42 completed 2/2 accepted steps, recorded four micro-batches and eight samples, preserved natural `rule_reward` metadata, passed replay/sample linkage and state continuity, and reloaded the step-2 checkpoint. Active post-step FP32 gate means were `0.0` on both steps; GPU wall time was `26` seconds.

The natural reward distribution collapsed in this short run: every generated sample received `0.1` (termination component only), so all group-relative advantages were zero and both parameter deltas were exactly zero. The corrected audit status is `NATURAL_RULE_REWARD_ZERO_ADVANTAGE_DIAGNOSTIC`. The first audit's checkpoint-path assumption was corrected offline into `natural_rule_reward_audit_v2`; no second GPU task was started.

This validates the natural reward plumbing and corrected active gate only. It is not evidence of useful reward diversity, nonzero policy updates, ratio clipping, validation improvement, or model quality. The checkpoint is not promoted; default weights, reward semantics, checkpoint selection and prior experiment roots remain unchanged.

## 2026-08-02 MM-E028 / MM-F036: natural reward diversity audit

This stage was offline-only. The new audit scanned 13 persisted `samples.jsonl` artifacts: the current E027 natural rule-reward smoke plus the E010 v2 and E009 v1 artifacts. CUDA was disabled, GPU wall time was `0` seconds, and the server remained `RUNNING` with an idle L4.

The current E027 source has 8 samples and 4 within-run groups. Every sample has reward `0.1`, every group is collapsed, the only nonzero component is `termination_reward`, validator nonzero rate is `0`, and the nonzero reward-spread group count is `0`. The audit status is `NATURAL_REWARD_DIVERSITY_AUDIT_COLLAPSE_CONFIRMED`. Older E010/E009 artifacts contain reward diversity, but their legacy schemas and data sources are reported separately and are not causal evidence for the current E027 path.

The result is diagnostic only: no reward, optimizer, KL gate, checkpoint selection, weight, or default model was changed. Before any further GPU training, audit current prompt/output component coverage and whether validator/termination signals can vary under the current generation limits.

## 2026-08-02 MM-E029 / MM-F037: natural reward input/output coverage audit

This stage was offline-only and used the 128-row Alignment v2 train manifest plus the persisted E027 natural smoke samples. The manifest is balanced across eight categories (`16` each), while the current artifact contains only 8 samples from `conciseness`, covering 4/128 prompts and 4/57 families.

Replay found validator pass `0/8`; all failures were `length_or_core_definition`. Every sample received reward `0.1`, with termination reward values `{0.1}`. The telemetry has 4 `eos` and 4 `max_new_tokens` rows, both receiving the same termination reward, so this reward component does not distinguish those ending reasons. All four groups remained collapsed and no nonzero reward spread was observed. Persisted reward and component replay matched exactly.

Offline probes using chosen, empty, and newline-terminated outputs show that the validator and `rule_reward` functions can vary. This is function-level evidence only, not model-generation evidence. The audit status is `CURRENT_GENERATION_SIGNAL_COVERAGE_INSUFFICIENT_DIAGNOSTIC`; no model, reward, optimizer, KL gate, or checkpoint selection was changed.
## 2026-08-02 MM-E030 / MM-F038: balanced natural-reward coverage smoke

This was one isolated GRPO seed-42 smoke using deterministic one-row-per-category train and validation manifests. The generated artifact covered all eight categories, eight families and eight prompts across 32 samples; the source chosen records passed their validators 8/8. The natural validator replay found 2/32 current generations passing, while the balanced validation baseline remained 0/8. Persisted reward and component replay matched exactly with zero mismatches.

The explicit `fp32_no_autocast` active post-step gate accepted both optimizer steps at learning-rate multiplier `1.0`. Active post-step KL means were `0.00010493` and `0.00009452`, with no backoff; parameter deltas were nonzero (`0.00242009` and `0.00171793` L2). The legacy bfloat16 shadow means were `0.41702` and `0.03688`, and both steps reported shadow-gate disagreement. This confirms a diagnostic update/measurement path, not model quality or optimizer superiority.

The audit status is `CURRENT_GENERATION_SIGNAL_VARIABILITY_OBSERVED_DIAGNOSTIC`: category/family/prompt coverage and termination/reward variability are now observable, but validation remains 0/8 and train validator coverage is sparse. GPU wall time was 33 seconds; the L4 returned to 0 MiB and the server remains `RUNNING`. The selected checkpoint is retained only as an experiment artifact. No default weight, reward, checkpoint gate or formal RL decision changed; project status remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`.
## 2026-08-02 MM-E031 / MM-F039: balanced output-quality audit

This stage was offline-only. It replayed the 8-row balanced Alignment v2 train manifest and the 32 E030 generations, producing per-sample diagnostics plus category, step and prompt aggregates. Category, family and prompt coverage were all complete (`8/8`), source chosen validation was `8/8`, and persisted reward/component replay had zero mismatches.

The output-quality signal remained sparse: validator pass was `2/32` (`0.0625`). Failure reasons were distributed across the category-specific validators rather than dominated by one parser failure. `max_new_tokens` occurred in `20/32` samples, natural end in `12/32`, and termination reward values varied between `0.0` and `0.1`. The resulting status is `OUTPUT_QUALITY_SIGNAL_SPARSE_DIAGNOSTIC`, not a training or model-quality result.

No GPU task ran; GPU wall time was `0` seconds. No reward, optimizer, KL gate, checkpoint selection, weight or default model changed. The project remains `DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`; formal RL expansion remains paused pending a separately approved quality/output decision.

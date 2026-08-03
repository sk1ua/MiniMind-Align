# 答辩问题清单

## 1. 为什么 SFT v2 比 DPO v2 更有效？

SFT v2 直接把八类程序可验证行为写进当前策略；DPO v2 只有 128 条 on-policy pair，且 chosen 已被 SFT 学到，因此它主要做边际排序调整。冻结集上 pass 是 50→52，而不是大幅跃迁。

## 2. 为什么不能只看 preference accuracy？

DPO v1 是反例：训练/验证 preference accuracy 接近 100%，自由生成变化很小。pairwise objective 可能只学会区分固定 pair，不能凭空获得算术、JSON 或精确数量能力。

## 3. hard negative 如何避免旧 full_sft rejected 过于容易？

从最新 `align_sft_v2` 每 prompt 采样 4 个候选，保留全部候选和 log-prob，再用 validator rank 选 chosen/hard rejected；另有 Gemini 32 条 smoke 核验。

## 4. SimPO 与 DPO 的关键差别？

SimPO 用 chosen/rejected response-token average log-prob 的 margin，加 beta/gamma，不加载 reference model；实现明确 mask prompt、归一化有效 response token，并有手工公式单测。

## 5. Reward Head 为什么取最后 response token？

它是固定长度序列的低成本 summary。实现优先取 response/EOS mask 的最后位置，若 marker 被截断则回退到 attention mask 的最后有效 token，不假设 EOS 一定存在。

## 6. Reward Model 有什么偏置？

validation accuracy 为 0.625，但 margin 与 chosen-minus-rejected length Pearson 为 -0.5057，说明它可能把短回答当成质量信号；因此 RL 先只用程序 reward，并保留长度监控。

## 7. 为什么没有直接复用已有 GRPO？

已有脚本依赖外部 1.8B reward model 和 rollout engine。为控制 64M 项目资源与变量，新增 rule-reward lite；原脚本保留且未伪装成已完成的外部 reward RL。

## 8. GRPO/CISPO 结果能说明什么？

只能说明 end-to-end smoke 可运行、checkpoint 可回载、KL/reward 能记录；4-step tiny pilot 中两者完全相同，step 3 组内 reward std=0，因此不能声明算法优劣。

## 9. 对齐税如何测？

使用独立 Alignment v2 validation loss/PPL 作为代理：align_sft_v2 PPL 2.9890，full_sft 26.6549，LoRA v2 8.2739。它不是公开通用 benchmark。

## 10. 结果怎样复现？

先 `bash scripts/reproduce_all.sh --dry-run`，再按 stage 选择 smoke/full；所有训练使用独立目录、固定 seed 和 wrapper 审计。Git 中不包含大权重。

## 11. Gemini 盲评是否改变了模型选择？

它没有推翻 validator 结论，但校准了结论强度：DPO v2 100 条中 93 条是 tie，说明 50→52 的规则增量很小；SimPO pilot 16/100 胜出、77/100 tie，说明存在小幅主观增益但不能称为全面泛化。SimPO safety 平均分略低，因此安全类别仍是发布前的 guardrail。

SimPO full 更能说明问题：Gemini 评审中基线胜 31、SimPO full 胜 22，平均 overall 从 1.40 降到 1.28，因此不能因为 natural end 达到 99/100 就选择它。

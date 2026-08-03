# MiniMind-Align 简历描述

## 一句话版本

在 Google Cloud NVIDIA L4 上从零构建并复现实验型 MiniMind-Align 对齐流水线，覆盖 Alignment v2 数据审计、SFT/LoRA、on-policy DPO/SimPO、轻量 Reward Model 与规则奖励 GRPO/CISPO，并以冻结评测、权重哈希、失败留痕和可复现脚本完成工程交付。

## 项目经历版本

**MiniMind-Align｜LLM Alignment Research & Engineering**

- 设计 Alignment v2 数据管线，覆盖 format、instruction、reasoning、safety、repetition、conciseness、termination、uncertainty 等规则类别；完成 1600 条 merged SFT 数据的 smoke/pilot 审计与泄漏检查。
- 在 MiniMind 基线上实现 SFT v2、LoRA、on-policy hard negative、DPO v2 和 SimPO；冻结 100 条测试 prompt 后，SimPO 64-step pilot 达到 64/100 validator pass、93/100 natural end。
- 实现轻量 Reward Model 与 Bradley–Terry pairwise loss；64-step pilot validation accuracy 为 20/32，并测得 reward margin 与长度差 Pearson 相关为 -0.5057，识别短回答偏置。
- 实现规则奖励 GRPO/CISPO lite pilot，记录 reward、KL、completion length、组内方差和 checkpoint SHA256；明确 tiny pilot 不足以支持收敛或算法优劣结论。
- 建立统一评测、成本代理、失败分析、独立 experiment id、GPU/磁盘前置检查、复现脚本和 Streamlit Demo；保留错误命令与失败目录，避免把局部指标包装成公开 benchmark 结论。
- 使用 Gemini 3.6 Flash 对 DPO v2 和 SimPO pilot 各完成 100 条冻结集匿名 A/B 评审：DPO `6/100` 胜出、SimPO `16/100` 胜出，并将 validator 与主观质量脱钩写入报告。
- 进一步对 SimPO full 完成 100 条 Gemini 评审，发现其平均 overall 低于基线且基线胜出 `31/100`，将长度/重复过度优化从规则指标风险提升为多评审信号支持的负结果。

## 技术关键词

PyTorch、MiniMind、SFT、LoRA、DPO、SimPO、Reward Model、GRPO、CISPO、on-policy preference、Gemini judge、rule-based reward、NVIDIA L4、tmux、SHA256、Streamlit、experiment registry。

## 面试口径

最重要的工程判断不是选择一个最高分，而是发现 SimPO full 的极短输出和 repetition 类退化后，将它保留为 over-optimization 对照；同时把 Gemini 只做了 32 条 smoke、RL 只做 4-step tiny pilot、账单美元没有实测等限制明确写进最终报告。

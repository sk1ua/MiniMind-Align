# SimPO full Gemini 独立复核

## 目的

检查 SimPO full 的负结果是否依赖 Gemini 匿名 A/B 顺序。两轮使用同一冻结 100 条测试 prompt、同一批 `align_sft_v2` 与 `simpo_v1_full` 生成结果，只改变匿名顺序 seed。

## 数据质量

- seed=42：100/100，错误 0。
- seed=43：100/100，错误 0。
- 两轮 ID 均唯一，ID 集完全一致，共享 100/100。
- 原始 judge JSONL 保留在远端 `results/experiments/`，不提交到 Git。

## 结果

| seed | tie | align_sft_v2 胜 | simpo_v1_full 胜 | baseline overall | candidate overall |
|---:|---:|---:|---:|---:|---:|
| 42 | 47 | 31 | 22 | 1.40 | 1.28 |
| 43 | 47 | 32 | 21 | 1.32 | 1.21 |

合并 200 条评审后：tie 94、`align_sft_v2` 胜 63、`simpo_v1_full` 胜 43；平均 overall 为 1.360 vs 1.245，候选相对基线差值为 -0.115。两轮逐样本胜者完全一致 81/100（0.810）。

## 结论边界

两轮都支持 `align_sft_v2` 优于 `simpo_v1_full`，增强了“SimPO full 过优化”判断的评审稳定性。由于两轮复用同一生成集，这不是新的泛化测试，不能替代公开 benchmark、独立人工专家复核或长程 RL 收敛实验。

可复现分析器：`evaluation/analyze_gemini_pair_replicates.py`。

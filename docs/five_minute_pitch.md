# 五分钟项目介绍稿

我做的是 MiniMind-Align：以约 64M 参数的 MiniMind 为载体，建立低资源、可审计的对齐实验平台。重点不是把一个大模型换个名字，而是回答小模型在数据、偏好优化、奖励模型和 RL 之间到底能学到什么，以及提升是否值得它付出的对齐税。

第一步我没有直接堆训练，而是做 Alignment v2 数据工程。数据覆盖 format、instruction、reasoning、safety、repetition、conciseness、termination、uncertainty 八类，程序生成答案并用专用 validator 审计；train/validation 与 100 条冻结测试 prompt 做泄漏检查。之后从同一个 full_sft 起点训练 Alignment SFT v2 和 LoRA 对照。

最清晰的事实是：在相同冻结测试集上，full_sft 只有 0/100 validator pass、1/100 natural end；align_sft_v2 达到 50/100、88/100。独立 validation PPL 从 26.65 降到 2.99。说明高质量、可验证的 SFT 数据对这个小模型非常关键。

第二个问题是 DPO 是否能继续带来收益。旧 DPO v1 已说明 preference accuracy 接近 100% 不代表自由生成更好，所以我从最新策略采样 4 个候选、保留 hard negative，再做 DPO v2 和 SimPO。DPO v2 在冻结集是 52 pass；SimPO 64-step pilot 是 64 pass。可是 SimPO 继续训练到 256 steps 时回答平均只剩 34 tokens，pass 反而降到 58，repetition 类只剩 1/13。这是非常有价值的负结果：更短、更容易结束不等于更高质量。

第三个问题是奖励模型和 RL。我实现了 MiniMind backbone + scalar Reward Head，validation pair accuracy 0.625，但与长度差有 -0.5057 Pearson 的短回答偏置。为了控制变量，GRPO/CISPO 第一轮只用于 JSON、数量、算术和终止等可验证任务，不加载外部 1.8B reward model。4-step pilot 能跑通并记录 KL/reward，但出现 group reward std=0，GRPO/CISPO 结果完全相同，因此我把它标为 tiny pilot，而不是夸大成收敛结论。

工程上，每阶段都有独立 Git commit：数据/SFT、on-policy/DPO/SimPO、Reward Model、rule-reward GRPO/CISPO；实验目录记录命令、环境、GPU、磁盘、日志和失败。最终仓库把事实、推断、假设、失败和未完成内容分开，复现脚本默认 dry-run，避免读者无意间启动昂贵训练。
补充验证：为检验规则 validator 是否过度乐观，使用 Gemini 对 100 条冻结 prompt 做了完整匿名 A/B 盲评。DPO v2 对 align_sft_v2 为 tie 93、DPO 胜 6、SFT 胜 1；SimPO pilot 为 tie 77、SimPO 胜 16、SFT 胜 7。DPO 的主观增量几乎为零，SimPO 有小幅支持但远非全面胜出。
对 SimPO full 的同条件评审则是 tie 47、SFT 胜 31、SimPO full 胜 22，平均 overall 也更低，说明继续优化确实把规则长度收益推向了主观质量退化。

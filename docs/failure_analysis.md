# 失败案例与风险分析

## 训练/工程失败

1. `dpo_v2_full_20260731` 首次使用不存在的连字符参数，argparse 在训练启动前拒绝；retry 使用脚本实际的下划线参数通过。
2. `reward_model_v1_smoke_20260731` 直接执行 trainer 时缺少仓库根路径，`align` 无法导入；补 `sys.path` 后 retry 通过。
3. `grpo_lite_smoke_20260731` 把 inference-mode generation tensor 直接送入 autograd；补 `detach().clone()` 后 retry 通过。
4. Gemini 对照首次启动时，wrapper 使用裸 `python` 收集环境，但服务器仅提供 `.venv/bin/python`；该目录保留为失败 preflight，确认没有发出 Gemini API 请求，随后修复 wrapper 并以独立 retry experiment 完成。

这些失败目录没有删除，也没有被写成成功实验。

## 负结果

- 旧 full_sft→DPO v1 的高 preference accuracy 没有带来自由生成改善；这正是本项目需要回答的“排序准确率不等于自由质量”。
- DPO v2 在 100 条冻结集仅从 align_sft_v2 的 50 pass 到 52 pass，增量有限。
- SimPO v1 full 的 average tokens 降到 34.26、natural end 到 99，但 validator pass 低于 64-step pilot，repetition 类仅 1/13，提示长度坍缩/奖励投机。
- Reward Model 的 format/instruction validation accuracy 只有 0.25；它与 chosen-shorter 之间出现负相关长度偏置。
- GRPO/CISPO tiny pilot 在 step 3 出现 reward variance 0；CISPO 与 GRPO 在四个 step 完全相同，算法比较不具备统计功效。
- GRPO/CISPO extended 使用更大 group diversity 后，8-step 训练均正常完成；冻结集 validator pass 都是 51/100，未出现可复现的算法差异。训练日志仍显示 reward、长度和 KL 随 step 波动，因此不能把该延长 run 写成收敛证据。
- SimPO full 的 Gemini 100 条评审为 tie 47、基线胜 31、SimPO full 胜 22，candidate average overall 为 1.28，低于 align_sft_v2 的 1.40；这使长度/重复坍缩不再只是 validator 层面的风险。
- SimPO full 的独立 seed=43 复核同样为 tie 47、基线胜 32、SimPO full 胜 21，平均 overall 为 1.21，低于基线 1.32；两轮 200 条评审的逐样本胜者一致率为 81%。这支持评审结论的方向稳定，但不代表新的独立泛化集。
- C003 on-policy Gemini 全量排名没有出现 API 或输出错误：训练集 128 条为 validator chosen 胜 62、tie 48、hard rejected 胜 18，验证集 32 条为 validator chosen 胜 16、tie 12、hard rejected 胜 4。该结果是对已构造 pair 的质量审计，不是模型在新 prompt 上的独立泛化结论。

## 数据完整性风险

测试 prompt 不进入 C-E 训练数据；DPO audit 对 train/validation/test overlap fail-closed。旧第二轮 Gemini judgment 的一个缺失样本仍以 incomplete 记录，不用伪造标签填补。

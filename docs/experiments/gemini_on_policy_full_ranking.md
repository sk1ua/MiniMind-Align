# C003 On-policy Gemini 全量排名

## 目的

对已通过 validator 审计的 on-policy pair 做外部 Gemini 盲评，比较同一 prompt 下的 validator_chosen 与 hard_rejected。评审结果只用于质量审计与报告，不回流训练。

## 运行配置

- API：Vertex AI Gemini 3.6 Flash
- project：gen-lang-client-0131552860
- location：global
- seed：42
- sleep：0.4
- max retries：5
- 评审脚本：evaluation/judge_generation_gemini.py
- 生成输入脚本：evaluation/build_gemini_rank_inputs.py

## 结果

| split | total | validator_chosen 胜 | tie | hard_rejected 胜 | candidate avg overall | baseline avg overall | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 128 | 62 | 48 | 18 | 1.6328125 | 0.921875 | 0 |
| validation | 32 | 16 | 12 | 4 | 2.09375 | 1.15625 | 0 |

训练集候选胜率（按全部样本）为 62/128 = 0.484375，验证集为 16/32 = 0.5。平均置信度分别为 0.921484375 和 0.9328125。

训练集 judgment SHA256：

798a3c9ee8df22cafb50793585de1ef88a02cb7e820ce8cb14faef7a31523533

训练集 summary SHA256：

21c7144dc934efe3f56be22452bb8d9dd232061153118c4da710fe9d3133353d

验证集 judgment SHA256：

bdaeb751a7bd9c9820e6ba2887109699730f3949843dfa93716b2420c351dd84

验证集 summary SHA256：

0ddedcb57ec9d7ba19bc63b4abe0c3186cd2dd4e7be720ed3e6fbb293f20d88d

## 复现路径

- 训练集输入：results/inputs/gemini_c003_train_full_baseline_20260801.jsonl、results/inputs/gemini_c003_train_full_candidate_20260801.jsonl
- 训练集产物：results/experiments/gemini_on_policy_c003_train_full_20260801/
- 验证集输入：results/inputs/on_policy_validation_hard_rejected_20260731.jsonl、results/inputs/on_policy_validation_validator_chosen_20260731.jsonl
- 验证集产物：results/experiments/gemini_on_policy_c003_validation_full_20260801/

完整命令见 docs/reproduction.md。4 条 smoke 评测保存在 results/experiments/gemini_on_policy_c003_train_smoke_20260801/，用于证明服务与配额链路正常，不替代全量结果。

## 限制

这两组样本来自既有 on-policy 生成与 pair 筛选流程，不是新的独立 prompt 泛化集；Gemini 胜负不能单独证明训练后模型泛化提升。公开 benchmark、长程 RL 收敛和真实 USD 账单仍未完成。

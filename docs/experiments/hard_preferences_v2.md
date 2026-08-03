# On-policy Hard Preferences v2

## 结论

Sprint C 使用 `out/align_sft_v2_pilot_768.pth` 作为最新策略，对冻结且未进入训练的数据 prompt 进行 4 候选采样。所有候选、sampling config、seed、validator 结果、长度、重复率和 log-prob 元数据均保留。validator rank 生成 chosen / hard rejected pair；Gemini 只在独立的 32 条 validation smoke 上做排序核验。

## 数据与审计

- train：128 prompts，8 类各 16 条；validation：32 prompts，8 类各 4 条。
- 每个 prompt 4 个候选：train 512 条，validation 128 条；每个 prompt 1 个 pair。
- train chosen validator pass：93/128；validation chosen validator pass：23/32。
- DPO v2 train：128 条，SHA256 `8d6babee0da82fd57b048a9690a258ce5213ca046932f4171477a7c90ec8589f`。
- DPO v2 validation：32 条，SHA256 `bef8be849edd9e61a2bd0a2254421cc2b840e950ed183abf3121c463c0df0c11`。
- fail-closed audit：PASS；train/validation prompt 不交叉，未命中 100 条冻结测试 prompt，8 类均覆盖。

主要产物：

- `results/experiments/on_policy_c001_train_20260731/`
- `results/experiments/on_policy_c001_validation_20260731/`
- `results/inputs/on_policy_train_manifest_128_20260731.jsonl`
- `results/inputs/on_policy_validation_manifest_32_20260731.jsonl`
- `dataset/alignment_v2/generated/dpo_v2_train_pilot.jsonl`
- `dataset/alignment_v2/generated/dpo_v2_validation_pilot.jsonl`

## Gemini 核验 smoke

32 个 pair 全部完成：validator chosen 胜 16，hard rejected 胜 5，tie 11；candidate category pass 12，baseline 6；平均 confidence 0.92656。这个结果用于验证排序接口和 pair 方向，不能外推为 128 条训练 pair 的全量 Gemini 盲评。

## 限制

本阶段的 hard-negative 主构造器是程序 validator，不是全量 Gemini ranking。它满足“最新策略 → 多候选 → validator 初筛 → pair”的可复现路径；后续 Reward Model 将在同一 pair schema 上继续训练和校准。

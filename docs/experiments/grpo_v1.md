# GRPO v1（rule-reward lite）

## 路线选择

仓库已有 `trainer/train_grpo.py` 依赖外部 1.8B `LMForRewardModel` 与 rollout engine；本阶段没有加载该外部模型，新增 `trainer/train_grpo_lite.py`，把第一轮 RL 限定在程序可验证任务，避免把外部依赖、开放式 safety judge 和大模型成本混在一起。

## 配置

- 初始化：`out/align_sft_v2_pilot_768.pth`
- prompt：`results/inputs/on_policy_validation_manifest_32_20260731.jsonl` 中 format、instruction、reasoning、termination 四类，pilot 16 prompts
- `batch_size=1`，`accumulation_steps=4`，`num_generations=4`
- `max_seq_len=384`，`max_gen_len=128`，`bfloat16`，seed=42，lr `3e-7`
- reward：validator 主分；termination bonus；3-gram repetition penalty；记录 parse/field/item-count/arithmetic/format 组件
- loss：PPO-style GRPO ratio clip，reference KL penalty `beta=0.02`

## 结果

1-step smoke（4 prompts × 4 generations）：reward mean `-0.1101`，reward std mean `0.1275`，KL `7.50e-05`，平均 completion 26.31 tokens，checkpoint verify PASS。

4-step pilot：

| step | reward mean | reward std mean | KL | completion tokens |
|---:|---:|---:|---:|---:|
| 1 | -0.1101 | 0.1275 | 0.000075 | 26.31 |
| 2 | 0.0875 | 0.0217 | 0.000295 | 14.13 |
| 3 | 0.0808 | 0.0000 | 0.000097 | 8.69 |
| 4 | 0.7557 | 0.4852 | 0.002536 | 14.31 |

权重：`results/experiments/grpo_lite_pilot_20260731/out/grpo_v1_lite_768.pth`，SHA256 `8c8def8d1d9eb6f7fd8470b80a06aad5bc2ffeaeee7099c7aab77899a979f144`。

## 结论边界

这是 16 prompt、4 step 的工程 smoke/pilot，不是完整 RL 收敛实验。step 3 的组内 reward std=0 是模式坍缩预警；step 4 KL 上升也说明需要更长的 KL/variance 监控后才能扩大规模。

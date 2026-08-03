# 轻量级 Reward Model v1

## 实现

新增 `align/reward_model.py`、`trainer/train_reward.py`、`evaluation/eval_reward.py` 和 `tests/test_reward_model.py`。

- backbone：MiniMind 63.91M，初始化自 `out/align_sft_v2_pilot_768.pth`。
- head：`Linear(hidden_size, 1)`，在最后一个有效 response token 或 EOS hidden 上取 scalar reward。
- 若 response marker 因 tokenizer 差异或截断缺失，回退到 `attention_mask` 的最后有效 token；代码和单测不假设 EOS 一定存在。
- loss：`-logsigmoid(r_chosen - r_rejected)`。
- 保存：独立 checkpoint，拒绝覆盖已有文件。
- API：`RewardAPI.reward(prompt, response)` 和 `RewardAPI.rank(prompt, responses)`，真实 CPU smoke 已通过。

## 训练与评测

- 数据：`dpo_v2_train_pilot.jsonl` 128 pairs；validation 32 pairs；chosen/rejected 共用 prompt，未使用冻结 test prompt。
- 8-step smoke：loss `1.0048 → 0.5774`，validation pair accuracy 14/32 = 0.4375；标记为未收敛 smoke。
- 64-step pilot：loss `1.0077 → 0.4639`；validation loss 0.5763；pair accuracy 20/32 = 0.625；平均 margin 0.3490。
- checkpoint SHA256：`8a516bcf1d4389a0d0ab632574f988821e79fd677bcc02236598155ddf4a4ce9`。

主产物：

- `results/experiments/reward_model_v1_smoke_retry_20260731/`
- `results/experiments/reward_model_v1_pilot_20260731/`
- `results/experiments/reward_model_v1_pilot_20260731/eval_gemini/reward_summary.json`

## 分类与长度偏置

validation 分类 pair accuracy：conciseness 1.00、format 0.25、instruction 0.25、reasoning 1.00、repetition 0.50、safety 0.75、termination 0.75、uncertainty 0.50。chosen response 平均 27.44 tokens，rejected 平均 39.78 tokens；margin 与 chosen-minus-rejected response length 的 Pearson 为 -0.5057，说明当前小 Reward Model 存在“更短更高分”的偏置风险。

## Gemini 一致率

在 C003 的 32 条 Gemini ranking smoke 上，去掉 11 条 tie 后剩 21 条；Reward Model 与 Gemini winner 一致 12/21 = 0.5714。这个一致率只用于小规模 sanity check，不是通用 judge 可靠性结论。

## 限制与下一步

当前训练只覆盖 128 条 hard-preference pilot split，validation 只有 32 条，format/instruction 类别仍明显较弱；Reward Model 也受到 on-policy pair 的长度差影响。Sprint E 只先在规则可验证任务上使用它或程序 reward，并记录 KL、reward variance 和模式坍缩，不把它用于开放式 safety 纯规则 RL。

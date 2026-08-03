# CISPO v1（rule-reward lite）

## 实现

`align/rl_rules.py` 中独立实现 CISPO 分支：

`-(detach(clamp(exp(new-old), max=epsilon_high)) * advantage * new_logp - beta * KL)`

其中 response token mask 上做 masked mean；`epsilon_high=5.0`，`beta=0.02`。这与已有 `train_grpo.py` 中的 CISPO 约定保持一致，同时添加 finite-loss 数值单测。

## 结果

使用与 GRPO 完全相同的 16 个 rule prompts、4 generations、4 optimizer steps、seed=42：

- step 1：reward `-0.1101`，std `0.1275`，KL `0.000075`
- step 2：reward `0.0875`，std `0.0217`，KL `0.000295`
- step 3：reward `0.0808`，std `0.0000`，KL `0.000097`
- step 4：reward `0.7557`，std `0.4852`，KL `0.002536`
- checkpoint verify：PASS；SHA256 `07da541e5e31eead0362cfe6144b5e628998835ab9d3aaad465c8b832e344a3d`

## 解释

本次 tiny pilot 的 GRPO/CISPO 日志完全相同，说明在 ratio 初始为 1、样本极少且每个 prompt 只更新一次的设置下，不能从该实验判断两种目标的优劣。CISPO 已完成实现和可运行性验证，但质量结论标记为 `inconclusive_tiny_pilot`。

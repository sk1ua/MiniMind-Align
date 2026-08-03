# MM-F019：RL KL 尖峰与 reward 传导离线归因

## 结论

对 `results/experiments/rl_stability_diagnostic_20260801/` 中六个正式 run 做了 step-level telemetry 与 per-sample 输出的离线归因，没有启动 GPU，也没有改变任何训练、checkpoint 或模型晋级门禁。

六个 run 均出现以下组合信号：

- 训练 validator 峰值为约 `0.8594–0.9062`，但 selected validation 始终为 `13/32`；训练 reward 上升没有传导到独立 validation。
- KL mean 在尾部超过 `0.005`，且 aggregate KL max/P95 明显高于同一步均值；最大 reference KL 约 `2.77`。
- 梯度裁剪持续发生：control/low_lr 为 20/20 steps，accum16 为 9/10 steps。
- max-length hit 与 repetition penalty 在训练过程中升高；没有发现 empty-response 或 selected safety/termination 下降。

最强的可支持解释是：高 KL 尾部与梯度裁剪共同出现，并伴随截断/重复信号；这只是诊断推断，不是因果证明。当前 artifacts 没有保存每个 micro-batch 的 log-prob 张量或梯度向量，因此无法进一步定位到具体 micro-batch/token。

## 审计输出

- 代码：`evaluation/audit_rl_spike_sources.py`
- 单测：`tests/test_rl_spike_sources.py`
- 权威输出：`results/experiments/rl_stability_diagnostic_20260801/spike_source_audit_v2/`
- 汇总：`summary.json`、`run_reports.jsonl`、`report.md`
- 审计状态：`DIAGNOSTIC_ONLY_WITH_SIGNALS`
- `all_json_finite=true`
- 模型门禁：`NOT_MET_NO_MODEL_CHANGE`

审计使用的阈值为 KL mean `0.005`、KL max/mean 尾部比 `10`、KL P95/mean 比 `3`、训练截断占比 `0.5`。这些阈值只用于解释信号，不改变既有 KL early-stop 或 checkpoint selection。

## 后续裁定

继续暂停正式三 seed RL 扩展。若要继续，先扩展每个 micro-batch 的可回放 telemetry（prompt/category、KL 分布、ratio 分布、梯度范数和生成样本的关联），再进行最小 GPU 对照；在 exact source attribution 完成前不讨论模型替换。

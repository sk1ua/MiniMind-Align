# Sprint C–F 阻塞记录（已解除）

截至 2026-07-31 后续续跑：用户已确认 Google Cloud 礼金可用于 Gemini，本记录仅保留此前暂停的审计轨迹；Sprint C 已恢复为 IN_PROGRESS。

日期：2026-07-31；当前 HEAD：Sprint B commit `7c3e8c6`。

Sprint A 和 Sprint B 已完成并通过各自 smoke、审计、checkpoint 和统一评测门禁。Sprint C 起需要：

- 从 `align_sft_v2` 生成 4–8 候选并做 hard-negative 筛选；
- Gemini 或 Reward Model 排序；
- DPO v2、SimPO、Reward Model、GRPO/CISPO 的多轮训练和统一评测。

这些步骤会产生新的外部 API 或 GPU 云计费。当前远端只确认了 GCP project、机器型号和磁盘/GPU 状态，没有 billing account 单价、折扣、quota 或可核验的预算授权；`estimated_cost` 不能可靠填写。按用户明确的停止规则，不启动 C001，也不把 C–F 写成已实现或已实验。

已保留：Sprint B 的所有失败日志、Gemini 2 条截断响应记录、100 条续跑后盲评结果、权重、统一测试和 registry。恢复时应先补成本估算，再执行 C001 smoke，并继续使用新 experiment id，禁止覆盖已有结果。

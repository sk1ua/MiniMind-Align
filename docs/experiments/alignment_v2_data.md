# Alignment v2 数据工程报告

## 实验事实

- seed: 42
- 新 train: 1000 条；新 validation: 160 条。
- 类别比例：format 200/32、instruction 200/32、reasoning 150/24、safety 150/24、repetition 100/16、conciseness 80/12、termination 60/10、uncertainty 60/10。
- Alignment v1 chosen 合并后 SFT train 为 1600 条；SFT validation 为 160 条。
- 生成方法计数：{"programmatic": 754, "template": 314, "manual": 92}。
- smoke：8 类各 3 条 train 和 3 条 validation，专用 validator、结构、泄漏检查通过。
- pilot：专用 validator 100% 通过；JSONL、角色顺序、ID、prompt、split 隔离、测试集相似度、v1 prompt 重复和真实 SFTDataset batch 检查通过。
- SFTDataset batch smoke：size=1600,batch_shape=(2, 512)。

## SHA256

- ./dataset/alignment_v2/generated/new_train_pilot.jsonl: aa64c0da3a0dbf563de5afc8addb6f883a80c4f2d633dac77caed464d7b8f34c
- ./dataset/alignment_v2/generated/new_validation_pilot.jsonl: 3b5b40422b2c5b93545a0a56c749ac1a283d9267fef5c7b2564978a08c21815b
- ./dataset/alignment_v2/manifests/train_manifest.jsonl: 7ba72615cbe1b3ba67b10cae599ba23f5e508949a86e5b634acaa6e465a13ff2
- ./dataset/alignment_v2/manifests/validation_manifest.jsonl: 6992f20accf88e783a2e4cdc10f188a3635e7fa1362486b425d74118f2b0c341

## 结论与限制

- 该数据集是程序化 pilot，不等同于人工大规模标注质量证明；sample_review.md 记录了每类随机抽查样本。
- 测试集未用于生成答案；审计器以 fail-closed 方式执行。
- 通过 Sprint A 数据门禁，可进入 SFT v2 smoke；是否使用完整训练权重仍需按 Sprint B 的训练前 GPU/时间检查执行。

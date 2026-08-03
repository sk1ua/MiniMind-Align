# DPO v2

## 配置

- policy 初始化：`out/align_sft_v2_pilot_768.pth`
- reference：同一 `align_sft_v2_pilot`，冻结；不覆盖 `dpo_v1`
- 数据：`dataset/alignment_v2/generated/dpo_v2_train_pilot.jsonl`，128 pairs
- `batch_size=2`，`max_seq_len=512`，`beta=0.15`，`learning_rate=5e-7`，`dtype=bfloat16`，seed=42
- smoke：8 steps；full-on-pilot-split：1 epoch、64 optimizer steps

## 结果

- smoke loss：日志中 0.6931、0.6620、0.6237、0.7168、0.7146、0.6791、0.7382、0.6864，checkpoint verify PASS。
- full-on-pilot-split：loss 日志首末 0.6498 → 0.6502；checkpoint SHA256 `4dd33b9b887a93ac7a3beee8982c777da6eb93c9e7ef8353834f71892e992db9`；CUDA reload verify PASS。
- validation pair eval：32 条，policy preference accuracy 26/32 = 0.8125；relative margin mean 0.0298818；normalized relative margin mean 0.0016144。
- 冻结 100 条测试：validator pass 52/100，natural end 93/100，平均 61.19 tokens，repeat-3gram 0.05049。

主要产物：

- `results/experiments/dpo_v2_smoke_20260731/`
- `results/experiments/dpo_v2_pilot_20260731/`
- `results/experiments/dpo_v2_full_retry_20260731/`
- `results/experiments/dpo_v2_eval_20260731/`
- `results/experiments/unified_sft_v2_20260731/validator/dpo_v2_full/`

## 解释

DPO v2 相对 align_sft_v2 的规则 pass 从 50 提升到 52，长度和重复率下降，natural end 提升到 93。由于 pair 数量只有 128 且 hard rejected 主要由 validator 选择，这是一项 pilot-split 结果；不能当作大规模 DPO 或公开 benchmark 结论。

## 失败记录

`dpo_v2_full_20260731` 因首次调用使用了不存在的连字符 CLI 参数而失败，没有启动训练；完整日志保留，改用下划线参数后的 `dpo_v2_full_retry_20260731` 通过。没有覆盖任何旧权重。

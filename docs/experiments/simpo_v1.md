# SimPO v1

## 实现

`trainer/train_simpo.py` 独立实现：

`-logsigmoid(beta * (chosen_avg_logp - rejected_avg_logp - gamma))`

其中 `chosen_avg_logp` / `rejected_avg_logp` 只在 response token mask 上计算，并按各自有效 response token 数归一化。`beta`、`gamma`、margin、chosen/rejected normalized log-prob 均写入 stdout；`tests/test_simpo.py` 的公式和 margin 方向测试为 2/2 PASS。

## 配置与结果

- 初始化：`out/align_sft_v2_pilot_768.pth`；`batch_size=2`；`max_seq_len=512`；`lr=1e-6`；`beta=2.0`；`gamma=0.5`；`bfloat16`；seed=42。
- 8-step smoke：loss 有限，checkpoint 保存通过。
- 64-step pilot：loss 1.7155 → 1.1966；末步 margin 0.10984；权重 SHA256 `b80cb435731f4c2f8aabf160a0b4b0c8ed62772e5e647e75a8da81d4b105cde4`；CUDA reload verify PASS。
- 256-step full：loss 1.7155 → 1.3087；末步 margin 0.21204；权重保存和实验包装器均 PASS。

## 冻结测试对比

| 模型 | validator pass | natural end | 平均 tokens | repeat-3gram |
|---|---:|---:|---:|---:|
| align_sft_v2 | 50/100 | 88/100 | 64.92 | 0.06725 |
| DPO v2 full | 52/100 | 93/100 | 61.19 | 0.05049 |
| SimPO v1 pilot | 64/100 | 93/100 | 56.23 | 0.04579 |
| SimPO v1 full | 58/100 | 99/100 | 34.26 | 0.03037 |

## 解释与风险

SimPO pilot 在这份冻结测试上取得最高 validator pass；256-step full 的 natural end 和长度继续改善，但 pass 降至 58，且 repetition pass 降至 1/13。这是过度优化/长度坍缩信号，因此保留 full 作为失败分析对照，不把它静默选为最佳模型。结果来自内部 100 条规则测试，不能外推公开 benchmark。

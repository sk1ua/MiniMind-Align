# Alignment SFT v2 实验报告

日期：2026-07-31
远端项目：`.`
GPU：NVIDIA L4；seed：42；冻结测试：`dataset/alignment_v1/splits/prompts_test.jsonl`，100 条。

## 数据与训练

- 训练集：Alignment v1 600 + Alignment v2 1000 = 1600 条。
- 验证集：Alignment v2 160 条；八类计数严格为 `32/32/24/24/16/12/10/10`。
- v2 merged train SHA256：`7ba72615cbe1b3ba67b10cae599ba23f5e508949a86e5b634acaa6e465a13ff2`。
- 训练命令：`trainer/train_full_sft.py --from_weight full_sft --data_path dataset/alignment_v2/generated/sft_train_pilot.jsonl --epochs 1 --max_steps 100 --batch_size 16 --learning_rate 1e-5 --max_seq_len 512 --dtype bfloat16 --use_compile 0`。
- 产物：`out/align_sft_v2_pilot_768.pth`，SHA256 `33aeed259a11835a0b958992afb5107f2d7df3053397d73dc8f293e15e9ab734`。
- 100 step loss：step 10 `2.2499`，step 50 `1.4890`，step 100 `0.8210`；无 NaN；checkpoint strict load 和 CUDA 生成验证通过。

## 统一固定测试

解码固定为 `do_sample=false, max_new_tokens=160, repetition_penalty=1.15, no_repeat_ngram_size=3`。四个模型使用同一 100 条 prompt、同一规则评分器：

| 模型 | validator pass | natural end | 平均 tokens | repeat 3-gram |
|---|---:|---:|---:|---:|
| full_sft | 0/100 | 1/100 | 159.09 | 0.100529 |
| align_sft_v1 | 19/100 | 57/100 | 104.44 | 0.149788 |
| align_sft_v2 | 50/100 | 88/100 | 64.92 | 0.067245 |
| LoRA-v2 | 30/100 | 88/100 | 65.54 | 0.090286 |

结果目录：`results/experiments/unified_sft_v2_20260731/`。

## Validation loss / PPL

均在同一 v2 validation split 上计算：

| 模型 | loss | PPL |
|---|---:|---:|
| full_sft | 3.282972 | 26.6549 |
| align_sft_v1 | 2.644963 | 14.0829 |
| align_sft_v2 | 1.094926 | 2.9890 |
| LoRA-v2 | 2.113108 | 8.2739 |

该结果支持 v2 的监督目标拟合明显改善，但不能单独证明泛化或真实用户偏好改善，因此保留固定测试与 Gemini 盲评作为独立证据。

## Gemini 盲评

使用 Vertex AI project `gen-lang-client-0131552860`、`gemini-3.6-flash`、seed 42；模型身份按 sample id 稳定随机交换；100 条结果已全部落盘。首次运行 98 条成功，2 条因截断 JSON 失败；提高 `max_output_tokens` 到 2048 后仅续跑这 2 条，原 98 条不重复调用。

- `align_sft_v2` 胜 68，`full_sft` 胜 4，tie 28。
- candidate category pass：30；baseline category pass：1。
- 平均 overall：`1.74 vs 0.34`；平均 confidence：`0.9343`。
- 结果：`results/experiments/gemini_align_sft_v2_20260731/summary.json`。

该评审是模型辅助证据，不替代人工复核；原始逐条结果、错误记录和 review 文件均保留。

## 结论与限制

Alignment SFT v2 通过 B001--B007 门禁，可作为后续 DPO/偏好实验候选基线。规则通过率仍只有 50%，且 v2 数据包含程序化和模板生成样本；进入 Sprint C 前应继续做 hard preference 审计、跨模型稳定性和人工复核。GPU/环境/磁盘快照及命令见各实验目录；美元成本未填报，因为当前运行记录没有可靠的 GCP 计费单价或 API token 账单数据，不把估算冒充实际成本。

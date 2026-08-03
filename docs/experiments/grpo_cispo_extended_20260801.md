# GRPO/CISPO extended 对照

## 目的

在已有 rule-reward lite pilot 之后，增加 prompt family 和 group diversity，检查短程训练是否仍然出现 reward、长度或 KL 坍缩，并在未参与 RL 训练的 100 条冻结测试 prompt 上做独立 validator 检查。

## 配置

- 基础权重：align_sft_v2_pilot
- manifest：results/inputs/on_policy_validation_manifest_32_20260731.jsonl
- categories：conciseness、format、instruction、reasoning、repetition、safety、termination、uncertainty
- max prompts：32
- generations per prompt：8
- accumulation steps：4
- max steps：8
- max sequence / generation length：384 / 96
- learning rate：3e-7
- beta：0.02；epsilon：0.2；epsilon-high：5.0
- dtype：bfloat16；device：cuda:0；seed：42

另运行了 2-step、8 prompt 的 GRPO smoke，确认新配置入口和权重保存链路正常。

## 训练信号

| step | GRPO reward | GRPO std | GRPO KL | GRPO tokens | CISPO reward | CISPO std | CISPO KL | CISPO tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1401 | 0.1346 | 0.000171 | 23.06 | 0.1401 | 0.1346 | 0.000171 | 23.06 |
| 2 | -0.0986 | 0.0963 | 0.000238 | 22.84 | -0.0986 | 0.0963 | 0.000238 | 22.84 |
| 3 | 0.2211 | 0.2516 | 0.000306 | 14.59 | 0.2211 | 0.2516 | 0.000306 | 14.59 |
| 4 | 0.2368 | 0.2079 | 0.000326 | 8.56 | 0.2368 | 0.2079 | 0.000361 | 8.56 |
| 5 | 0.7599 | 0.4131 | 0.000457 | 71.16 | 0.7599 | 0.4131 | 0.000446 | 71.16 |
| 6 | 0.7192 | 0.4521 | 0.000546 | 56.16 | 0.7192 | 0.4521 | 0.000539 | 56.16 |
| 7 | 0.5855 | 0.5501 | 0.001233 | 17.91 | 0.5855 | 0.5501 | 0.001167 | 17.91 |
| 8 | 0.4352 | 0.4644 | 0.000513 | 34.13 | 0.4352 | 0.4644 | 0.000489 | 34.13 |

两种模式的 reward 和长度轨迹完全一致，KL 只有轻微差异；该配置仍不足以区分 GRPO/CISPO 的实际算法收益。KL 最大值低于此前 pilot 的 0.002536，但 reward 和长度的 step 间波动仍明显。

## 冻结集结果

| model | count | validator pass | natural end | average tokens | repeat-3gram |
|---|---:|---:|---:|---:|---:|
| align_sft_v2 reference | 100 | 50 | 88 | 64.92 | 0.067245 |
| GRPO extended | 100 | 51 | 90 | 64.98 | 0.065962 |
| CISPO extended | 100 | 51 | 91 | 63.91 | 0.058080 |

GRPO 与 CISPO 相对 align_sft_v2 都只增加 1 个 validator pass；这不是有统计意义的改进，也不支持默认切换到任何一个 RL 权重。

## 产物与哈希

- smoke：results/experiments/grpo_lite_extension_smoke_20260801/；weight SHA256 3cd165209e110e77c49220ddcca586914d7c7d1a4a47a10a22ef7dbd6fb26e8a
- GRPO train：results/experiments/grpo_lite_extended_20260801/；weight SHA256 030b0e9879c2309a3a0e0474f206f871a6b0b62fc7a940cc4043b9a46d2c46aa
- CISPO train：results/experiments/cispo_lite_extended_20260801/；weight SHA256 889567f658c17c6b5e518ebd5288257417be08d24c2fc25f773cbbff3c09aa22
- GRPO frozen generation：af45a44cbf0abc8efb025627a13d7d0f78163669c2f4a67fc9b4123fdd2c9c85
- GRPO validator summary：a7ab6cf6d35d7819b164f071376bc36612f6bdd7d0715b1d74fc65e7fe9fb272
- CISPO frozen generation：d30b7e9bee582ce53a7af9ea318dbd21083c57893cdc24e65eadec57aaa2107a
- CISPO validator summary：15a357907a62b52070e6cdcb61f3eabc896f17928ab4dc6a49e09147c8cd8325

## 结论与限制

本轮完成了比 tiny pilot 更有 group diversity 的短程对照，但仍是 rule-reward lite、固定 32 条 validation prompt family、8 optimizer steps；没有外部 reward model、KL early stop 或长程收敛证据。因此门禁保持 PASS_WITH_LIMITATIONS，GRPO/CISPO 都保留为研究产物，不选择为默认模型。

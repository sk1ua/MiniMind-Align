# MM-E010 / MM-F017：checkpoint 回载修复后的正式 RL 复跑

## 范围与代码

本轮只验证两个问题：原生 Alignment v2 数据是否消除旧 v1 metadata 风险，以及 checkpoint 选点是否必须基于保存后回载的 artifact。GRPO/CISPO、`align.rl_rules.rule_reward`、学习率、KL 门禁、质量门槛和晋级规则保持不变。

- experiment root：`results/experiments/rl_data_isolation_reload_fixed_20260801/`
- run commit：`14076033f25fc7dfa35403f2d7beccb46ae43d5c`
- environment hash：`e55aadc2a3df4ab553c1d3fae8df57f1fa79f4108432e34a02296dfd11cace79`
- GPU：NVIDIA L4；实验结束 GPU memory：0 MiB；服务器：`RUNNING`
- 训练/validation manifest：`results/inputs/rl_data_isolation_train_128_20260801.jsonl`、`results/inputs/rl_data_isolation_validation_32_20260801.jsonl`

训练集为 v2 programmatic 数据 8 类各 16 条，共 128 条；validation 为每类第 5–8 条，共 32 条。ID overlap、family overlap 和与旧 validation 前 4 条 overlap 均为 0。选择 metadata 与 source/output hash 记录在 `results/inputs/rl_data_isolation_selection_20260801.json`。

## 正式协议与结果

`num_generations=8`、`accumulation_steps=8`、`max_prompts=128`、`validation_max_prompts=32`、`max_seq_len=384`、`max_gen_len=128`、`learning_rate=3e-7`、`beta=0.02`、`epsilon=0.2`、`epsilon_high=5.0`、`dtype=bfloat16`、每 4 steps validation/checkpoint、最多 32 steps、KL threshold `0.005`、patience `2`、quality drop `10` 个百分点。

| run | steps completed | selected step | reloaded validation | safety | termination | stop | GPU wall s | exit |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| GRPO seed 42 | 20 | 4 | 13/32 | 4/4 | 4/4 | KL | 200 | 0 |
| GRPO seed 43 | 20 | 4 | 13/32 | 4/4 | 4/4 | KL | 197 | 0 |
| GRPO seed 44 | 20 | 4 | 13/32 | 4/4 | 4/4 | KL | 200 | 0 |
| CISPO seed 42 | 20 | 4 | 13/32 | 4/4 | 4/4 | KL | 203 | 0 |
| CISPO seed 43 | 20 | 4 | 13/32 | 4/4 | 4/4 | KL | 202 | 0 |
| CISPO seed 44 | 18 | 4 | 13/32 | 4/4 | 4/4 | KL | 175 | 0 |

所有 validation checkpoint 记录的 `evaluation_source` 都是 `reloaded_checkpoint`。baseline 为 13/32；GRPO/CISPO 三 seed 均值均为 13.0、总体标准差均为 0，增量为 0。因此晋级门禁为 `NOT_MET_NO_MODEL_CHANGE`，默认模型不改变。

## Reward-hacking audit

审计文件为 `results/experiments/rl_data_isolation_reload_fixed_20260801/reward_hacking_audit/summary.json`，只生成诊断 warning，不改变 checkpoint 选择。六个 run 均出现：

- `train_validator_gain_without_validation_gain`
- `reward_gain_without_validation_gain`
- `max_length_hit_increase`
- `repetition_penalty_increase`

没有 empty-response 或 validation safety/termination/natural-end 下降 warning。部分训练 step 的 max-length hit 最高约 98.4%；这表示需要继续审计长度/重复与 reward 的关系，不能解释成质量提升。

## 冻结集泛化证据

baseline 和六个唯一 selected checkpoint 均完成 100 条冻结集 validator 评测；冻结集不参与 checkpoint 选择或晋级：

| model | validator pass | natural end | avg tokens | avg repeat-3gram |
| --- | ---: | ---: | ---: | ---: |
| align_sft_v2_pilot | 50/100 | 88 | 64.92 | 0.06724523 |
| GRPO seed 42 | 51/100 | 91 | 63.65 | 0.05948391 |
| GRPO seed 43 | 51/100 | 90 | 64.56 | 0.06234709 |
| GRPO seed 44 | 51/100 | 90 | 63.44 | 0.06124803 |
| CISPO seed 42 | 50/100 | 89 | 64.36 | 0.06359371 |
| CISPO seed 43 | 50/100 | 89 | 64.27 | 0.06391136 |
| CISPO seed 44 | 51/100 | 90 | 64.14 | 0.06631989 |

GRPO 均值为 51.0（标准差 0），CISPO 均值为 50.33（总体标准差约 0.47）。本阶段不重跑 C-Eval，上一轮 C-Eval 结论保持不变。

## 验收与资源

- 70 个 JSON/JSONL 文件全部可解析且无非有限数。
- 六个 run、七个冻结 generation 均 exit code 0；每个目录保留 command、GPU、磁盘、环境 hash、resource monitor、validation history、selection 和日志。
- 六个训练 GPU wall time 合计 1177 秒，七个冻结 generation 合计 674 秒，总计 1851 秒，低于 28800 秒硬上限。
- 旧 `results/experiments/rl_data_isolation_20260801/`、`rl_validation_coverage_20260801/`、smoke 目录和默认权重均未覆盖。
- 服务器完成后保持 `RUNNING`。

复现入口：

```bash
bash scripts/run_rl_data_isolation_reload_fixed.sh --dry-run
bash scripts/run_rl_data_isolation_reload_fixed.sh --smoke
bash scripts/run_rl_data_isolation_reload_fixed.sh --full
```

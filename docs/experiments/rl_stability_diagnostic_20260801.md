# MM-E011 / MM-F018：RL 稳定性诊断

## 结论

本轮只做优化稳定性诊断，不改变 reward、KL early-stop、checkpoint selection 或模型晋级规则。GRPO 与 CISPO 各运行 seed=42 的 control、low_lr、accum16 三个条件；六个正式 run 均正常退出，但都在 reference KL 连续两步超过 `0.005` 后早停。统一审计完成且所有 JSON/JSONL 数值有限，但没有任何条件满足稳定性改善定义：KL 触发没有相对 control 延后至少 4 步，同时满足 KL P95/最大值和梯度不恶化。

因此 MM-E011 的诊断结论为“训练器/当前 RL 配置稳定性问题未解决”，MM-F018 最终状态为 `NOT_MET_NO_MODEL_CHANGE`。默认模型、既有权重和既有实验结果均不改变；本轮不重跑 C-Eval 或 100 条冻结集。

## 协议与可追溯性

| 项目 | 配置 |
|---|---|
| 数据 | `results/inputs/rl_data_isolation_train_128_20260801.jsonl` + 独立 validation 32 条 |
| 方法 | GRPO、CISPO；每个条件 seed=42 |
| 条件 | control=`3e-7/accum8`；low_lr=`1e-7/accum8`；accum16=`3e-7/accum16` |
| 固定项 | max_steps=20、eval/checkpoint 每 4 步、8 generations、max_prompts=128、max_seq_len=384、max_gen_len=128、bfloat16 |
| 门禁 | KL threshold=`0.005`、patience=`2`、质量下降门限=10 points；max grad norm=`1.0` |
| 运行代码 | commit `b37632ed8519ed92c5b1a3e69b6992fa14d638b3` |
| 环境 | `.venv`、NVIDIA L4、environment hash `959a641cdc0e988066d3646bb978fb94fdd4ff5ab967736f04d6e1a4f3c09569` |
| 资源 | 正式 1125 GPU-seconds；smoke 13 GPU-seconds；硬上限 7200 秒 |

训练日志明确将 KL 标记为 `pre_optimizer_step`；loss、裁剪前/后梯度范数、ratio P50/P95/max、micro-batch reference-KL 分布和 reward components 均按 step 保存。

## Smoke

`grpo_smoke_control_seed42` 使用 2 steps、2 validation prompts、2 generations、accumulation=1，exit code=0；step 2 checkpoint 可回载，telemetry JSONL 可解析且无 NaN。产物位于稳定性实验根目录下的 smoke 子目录，没有复用正式 run 目录。

## 正式 run 结果

`first KL` 是连续两步超限触发的 step；`selected` 是保存并回载后保留的 checkpoint。所有 selected checkpoint 的独立 validation 都是 `13/32`，safety=`4/4`，termination=`4/4`，natural end=`32/32`。

| run | 完成步数 | first KL | selected | max KL P95 | max KL | max grad pre-clip | max length hit | max repetition penalty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GRPO control | 20 | 20 | 4 | 0.0820 | 2.4084 | 3.5450 | 0.8906 | 0.2046 |
| GRPO low_lr | 20 | 20 | 4 | 0.0831 | 2.1839 | 3.6828 | 0.8906 | 0.2055 |
| GRPO accum16 | 10 | 10 | 4 | 0.0728 | 2.7662 | 2.7813 | 0.8281 | 0.1644 |
| CISPO control | 20 | 20 | 4 | 0.0562 | 2.4321 | 3.6319 | 0.8906 | 0.2055 |
| CISPO low_lr | 20 | 20 | 4 | 0.0498 | 2.6437 | 3.6849 | 0.8906 | 0.2055 |
| CISPO accum16 | 10 | 10 | 4 | 0.0744 | 2.7668 | 2.7829 | 0.8281 | 0.1644 |

每个 run 的 `run.log` 同时保留完整 command、commit、环境 hash、GPU/磁盘 before/after、exit code、finish time 和 wall time；每个正式 run 的最终权重与 step 4/8/12/16/20 checkpoint 保持在独立目录中。accum16 虽然部分早期均值 KL 较小，但在 step 10 触发，早于 control 的 step 20；low_lr 触发时间与 control 相同，且梯度最大值并未降低。因此二者都不能定义为稳定性改善。

## 审计判定

稳定性审计 `results/experiments/rl_stability_diagnostic_20260801/stability_audit/summary.json` 的 `status` 为 `PASS`，含义是审计流程、六个 run 读取和 finite 检查完成，并不表示优化条件通过。六个 run 状态均为诊断 `WARNING`，主要记录 KL early stop 和梯度裁剪活动；`stability_improved_conditions` 为空。

审计比较结果：

- GRPO accum16：KL 触发 step 10 vs control step 20，未延后；最大 KL 还更高。
- GRPO low_lr：触发仍为 step 20；KL P95 和最大值没有同时不高于 control，梯度最大值更高。
- CISPO accum16：触发 step 10，且 KL P95/最大值都更高。
- CISPO low_lr：触发仍为 step 20；最大 KL 与梯度最大值更高。

这些 run 不产生模型改进结论，也不能据此判定某个单一训练器 bug 已被定位。下一轮若继续，应先在不放宽门禁的前提下定位 KL 尖峰与大梯度的来源，再考虑更小规模的可复现实验；在此之前暂停正式三 seed 扩展。

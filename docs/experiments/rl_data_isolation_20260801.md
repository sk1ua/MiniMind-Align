# MM-E010 / MM-F017：RL 数据隔离与 reward-hacking 审计

> 本文记录初始 data-isolation run；其中 14/32 是训练进程内存中的选点指标，不能视为保存后 artifact 的可复现结果。checkpoint 回载修复后的正式复跑见 `docs/experiments/rl_data_isolation_reload_fixed_20260801.md`，旧目录和本文作为历史诊断保留。

## 目标与代码版本

本轮只替换 RL 训练/验证数据来源，保持 GRPO/CISPO、`align.rl_rules.rule_reward`、学习率、KL 和晋级规则不变。正式代码 commit 为 `29a6a9135a803e5355c8288c8b304243663b2c63`，服务器为 NVIDIA L4。

## 数据选择

- train source：`dataset/alignment_v2/manifests/train_manifest.jsonl`。
- 筛选条件：`source=alignment_v2_programmatic_v1` 且 metadata 非空；seed=42，按 `sha256(seed:id)` 排序后每类取 16 条。
- validation source：`dataset/alignment_v2/manifests/validation_manifest.jsonl`；每类按原始顺序取第 5–8 条，避开旧 validation 的前 4 条。
- 结果：8 类各 16 条 train，共 128；8 类各 4 条 validation，共 32；metadata 非空计数分别为 128/32。
- train/validation ID overlap、family overlap、validation 与 existing 前四条 overlap 均为 0。
- selection metadata：`results/inputs/rl_data_isolation_selection_20260801.json`。
- source SHA256：train `7ba72615cbe1b3ba67b10cae599ba23f5e508949a86e5b634acaa6e465a13ff2`，validation `6992f20accf88e783a2e4cdc10f188a3635e7fa1362486b425d74118f2b0c341`。
- output SHA256：train `dd0a44b87d540043f740e320eb9d2a88d8a61de9ef5bd97940014ae3ca3b5350`，validation `19a386a5b9a3ccf24155c727e1df24763f1fc7d2252105df148c614b05475437`。

## 协议与 run 结果

正式配置为 `num_generations=8`、`accumulation_steps=8`、`max_prompts=128`、`validation_max_prompts=32`、`max_seq_len=384`、`max_gen_len=128`、`learning_rate=3e-7`、`beta=0.02`、`epsilon=0.2`、`epsilon_high=5.0`、`dtype=bfloat16`、`eval/checkpoint every 4`、`max_steps=32`、KL threshold `0.005`、patience `2`、quality drop `10` 个百分点。

| run | selected step | validation pass | safety | termination | stop | GPU wall s | exit |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| GRPO seed 42 | 4 | 14/32 | 4/4 | 4/4 | KL | 191 | 0 |
| GRPO seed 43 | 4 | 14/32 | 4/4 | 4/4 | KL | 189 | 0 |
| GRPO seed 44 | 8 | 14/32 | 4/4 | 4/4 | KL | 187 | 0 |
| CISPO seed 42 | 4 | 14/32 | 4/4 | 4/4 | KL | 188 | 0 |
| CISPO seed 43 | 4 | 14/32 | 4/4 | 4/4 | KL | 190 | 0 |
| CISPO seed 44 | 8 | 14/32 | 4/4 | 4/4 | KL | 165 | 0 |

独立 validation baseline 为 13/32，safety/termination 为 4/4。GRPO 三 seed 均值为 14.0、标准差 0；CISPO 三 seed 均值为 14.0、标准差 0。两者相对 baseline 都是 +1 pass，低于至少 +3 的晋级门禁；没有合格方法，默认模型不改变。

## 训练日志与 reward-hacking 审计

`trainer/train_grpo_lite.py` 额外记录 train validator pass、empty response、max-length hit、natural end、repetition penalty 和 reward components。`evaluation/audit_rl_reward_hacking.py` 对每个 run 按 step/category 对照 train 与独立 validation，warning 不改变 checkpoint 选择。

六个正式 run 的审计均报告：

- `max_length_hit_increase`：相对 baseline 的增加约 0.7656–0.8438；部分训练 step 的 max-length hit 最高约 98.4%。
- `repetition_penalty_increase`：约 0.1514–0.1972。
- 未报告 empty response、validation natural-end、validation safety 或 validation termination 下降 warning。

这些是 reward-hacking 风险信号，不等同于已经证明 reward hacking 是唯一原因。完整审计在 `results/experiments/rl_data_isolation_20260801/reward_hacking_audit/summary.json`。

## 冻结集泛化证据

baseline 和六个唯一 selected checkpoint 均完成现有 100 条冻结集 validator 评测，结果如下：

| model | validator pass | natural end | avg tokens | avg repeat-3gram |
| --- | ---: | ---: | ---: | ---: |
| align_sft_v2_pilot | 50/100 | 88 | 64.92 | 0.06724523 |
| CISPO seed 42/43/44 | 各 50/100 | 89 | 64.36/64.27/63.43 | 0.06359371/0.06391136/0.05894935 |
| GRPO seed 42/43/44 | 各 51/100 | 91/90/90 | 63.65/64.56/64.52 | 0.05948391/0.06234709/0.06234444 |

冻结集只作方向性泛化证据，不参与 checkpoint 选择；C-Eval 本阶段不重跑，保留上一轮公开评测结果。

## 追加：完整 v2 validation 覆盖与 checkpoint 回载核验

为解释 32 条 validation 切片上的短暂 +1 pass，另建独立诊断目录
`results/experiments/rl_validation_coverage_20260801/`，使用完整 Alignment v2
validation 160 条、贪心解码、`max_steps=0`，顺序评估 baseline 和 6 个已选 checkpoint。
这一步不参与选点、不改变晋级门禁，也不改写任何模型权重。

| 对象 | 回载后全量 v2 validation | 回载后预注册 32 条切片 | natural end | wrapper exit |
| --- | ---: | ---: | ---: | ---: |
| baseline | 47/160 | 13/32 | 160/160 | 0 |
| GRPO seed 42/43/44 | 各 47/160 | 各 13/32 | 各 160/160 | 0 |
| CISPO seed 42/43/44 | 各 47/160 | 各 13/32 | 各 160/160 | 0 |

所有 JSON/JSONL 共 29 个文件可解析且无非有限数，完成后 GPU 为 0 MiB，实例仍为
`RUNNING`；7 个串行 wrapper GPU wall time 合计 463 秒。结果汇总为
`results/experiments/rl_validation_coverage_20260801/coverage_summary.json`。

追加核验发现：6 个选中 checkpoint 的 `selection.json` 记录的是内存模型 14/32，
但独立回载后均为 13/32，和 baseline 相同；6 个均存在 1 pass 的回载差异。抽查
checkpoint 张量为 `float16`，而 `trainer/train_grpo_lite.py` 的保存路径使用
`.half()`，因此“保存精度导致选点指标不可复现”是当前最可能机制，但仍标注为推断，
不作为已证实的因果结论。后续若恢复训练，必须以“保存后回载再评测”的指标完成选点，
或保留更高精度 checkpoint；本诊断不合并入原有晋级平均值。

## 修复与 smoke 验收

`trainer/train_grpo_lite.py` 现已在每个可选 checkpoint 保存完成后重新加载序列化文件，
再用回载模型计算 validation、quality gate 和 checkpoint selection；非 checkpoint 的中间
validation 仍标记为 `in_memory_policy`，不可进入选点。新增回归测试
`tests/test_rl_checkpoint_reproducibility.py` 覆盖回载加载器。

独立 smoke `results/experiments/rl_checkpoint_reload_smoke_20260801/` 完成 1 step、1 个
checkpoint、2 条 validation，`validation_history.jsonl` 明确记录
`evaluation_source=reloaded_checkpoint`。随后独立 `max_steps=0` 回载评测与 smoke 的
validator/response 稳定字段完全一致（速度字段因运行时波动而不同）；两个 wrapper exit
code 均为 0，总 GPU wall time 27 秒，未覆盖旧目录或旧权重。

## 资源、复现与结论

训练 GPU wall time 合计 1110 秒，冻结 generation 合计 653 秒，总计 1763 秒，低于 28800 秒硬上限。六个训练和七个冻结 generation 均 exit code 0；GPU 完成后为 0 MiB，实例保持 `RUNNING`。所有结果位于独立目录 `results/experiments/rl_data_isolation_20260801/`，wrapper 为 `scripts/run_rl_data_isolation.sh`。

最终状态：`NOT_MET_NO_MODEL_CHANGE`。本轮没有证据支持将 GRPO/CISPO 权重晋级为默认模型。

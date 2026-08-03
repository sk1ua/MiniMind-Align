# MiniMind-Align 任务状态

状态：TODO、IN_PROGRESS、BLOCKED、DONE、FAILED。

## Sprint A

- [x] MM-A001 检查仓库、权重、数据和环境 — DONE
- [x] MM-A002 冻结 DPO v1 实验结论 — DONE
- [x] MM-A003 补齐或确认缺失盲评样本 — DONE（确认缺失，不补造判断）
- [x] MM-A004 创建 Alignment v2 目录和配置 — DONE
- [x] MM-A005 format 数据生成器 — DONE
- [x] MM-A006 instruction 数据生成器 — DONE
- [x] MM-A007 reasoning 数据生成器 — DONE
- [x] MM-A008 safety 模板库 — DONE
- [x] MM-A009 repetition/conciseness/termination/uncertainty — DONE
- [x] MM-A010 泄漏和近重复检测 — DONE
- [x] MM-A011 每类 3 条 smoke 数据 — DONE
- [x] MM-A012 smoke 审计 — DONE
- [x] MM-A013 完整 1000+160 pilot — DONE
- [x] MM-A014 合并 Alignment v1 chosen — DONE（v1 600 + v2 1000 = 1600）
- [x] MM-A015 完整审计和报告 — DONE
- [x] MM-A016 每类随机人工抽查 — DONE（每类 2 条，seed=42）
- [x] MM-A017 决定是否允许进入 SFT v2 — DONE（数据门禁通过）

## Sprint B

- [x] MM-B001 Alignment SFT v2 smoke、loss 监控与 checkpoint 验证 — DONE
- [x] MM-B002 Alignment SFT v2 pilot（1600 条 merged SFT） — DONE
- [x] MM-B003 LoRA smoke/pilot 与冻结测试生成 — DONE
- [x] MM-B004 四模型统一规则评测 — DONE
- [x] MM-B005 Gemini 盲评（100 条，补齐 2 条截断响应） — DONE
- [x] MM-B006 v2 validation loss/PPL 与 alignment tax — DONE
- [x] MM-B007 LoRA 单测、回归与产物归档 — DONE

## Sprint C：On-policy Hard Negative、DPO v2 和 SimPO

- [x] MM-C001 最新策略多候选采样 — DONE（128 train + 32 validation prompts，4 candidates/prompt）
- [x] MM-C002 hard negative 自动筛选 — DONE（validator rank，保留全部候选与 pair 元数据）
- [x] MM-C003 Gemini 或 Reward 排序 — DONE（32 条 Gemini ranking smoke；full pair 构造使用 validator，结果可追溯）
- [x] MM-C004 DPO v2 数据构造 — DONE（128/32，fail-closed 审计通过）
- [x] MM-C005 实现 SimPO — DONE（response-token average log-prob、beta/gamma、2/2 unit tests）
- [x] MM-C006 训练 DPO v2 — DONE（8-step smoke；64-step pilot/full-on-pilot-split）
- [x] MM-C007 训练 SimPO v1 — DONE（8-step smoke；64-step pilot；256-step full）
- [x] MM-C008 DPO/SimPO 消融评测 — DONE（冻结测试集 100 条，validator 结果已归档）
- [x] MM-C009 C003 全量 Gemini on-policy 排名 — DONE（128 train + 32 validation，错误 0，结果与哈希已归档）

## Sprint D：轻量级 Reward Model

- [x] MM-D001 实现 Reward Head — DONE（last valid response token / EOS，带 fallback）
- [x] MM-D002 实现 Pairwise Dataset — DONE（chosen/rejected 同 prompt，attention/response mask）
- [x] MM-D003 实现 Bradley–Terry Loss — DONE
- [x] MM-D004 训练轻量级 Reward Model — DONE（8-step smoke + 64-step pilot）
- [x] MM-D005 Pairwise Accuracy 评测 — DONE（validation 20/32）
- [x] MM-D006 长度偏置分析 — DONE（margin vs length delta Pearson）
- [x] MM-D007 与 Gemini 排序一致率 — DONE（21 non-tie pairs）
- [x] MM-D008 保存统一 Reward API — DONE（reward/rank API smoke）

## Sprint E：GRPO/CISPO

- [x] MM-E001 设计规则奖励任务 — DONE（format/instruction/reasoning/termination）
- [x] MM-E002 GRPO smoke test — DONE（4 prompts×4 generations，1 optimizer step）
- [x] MM-E003 完整 GRPO v1 — DONE（GRPO lite pilot，4 optimizer steps）
- [x] MM-E004 CISPO 实现或适配 — DONE（独立 loss 分支，upper-clipped detached ratio）
- [x] MM-E005 CISPO v1 — DONE（同配置 pilot，checkpoint verify PASS）
- [x] MM-E006 稳定性和模式坍缩分析 — DONE（reward variance、KL、长度、组内坍缩记录）
- [x] MM-E007 GRPO/CISPO 对比报告 — DONE（tiny pilot 结论标注为 incomplete）
- [x] MM-E008 GRPO/CISPO 延长对照与冻结泛化检查 — DONE（8 类、8 generations、8 steps；两者冻结集均 51/100，仍为 PASS_WITH_LIMITATIONS）

## Sprint F：统一评测、消融、Demo 和交付

- [x] MM-F001 统一冻结评测汇总与指标 JSON — DONE（100 条测试 prompt，事实与推断分离）
- [x] MM-F002 validation loss/PPL 与 alignment-tax proxy 图 — DONE
- [x] MM-F003 validator/category/RL stability/wall-time 静态图 — DONE（matplotlib 导出并完成视觉 QA）
- [x] MM-F004 成本与资源报告 — DONE（账单 USD 未测，不伪造；记录 wrapper wall time）
- [x] MM-F005 失败分析与未完成项 — DONE（保留错误命令、缺失 judgment、tiny pilot 限制）
- [x] MM-F006 复现脚本 — DONE（默认 dry-run，单阶段支持 smoke/full）
- [x] MM-F007 Streamlit Demo — DONE（app.py 编译与 headless startup PASS）
- [x] MM-F008 项目 README/路线图/实验总报告 — DONE
- [x] MM-F009 五分钟介绍与简历描述 — DONE
- [x] MM-F010 面试问答与最终 registry/state — DONE
- [x] MM-F011 DPO v2 完整 Gemini 冻结集对照 — DONE（100/100，tie 93、DPO 6、align_sft_v2 1，错误 0）
- [x] MM-F012 SimPO pilot 完整 Gemini 冻结集对照 — DONE（100/100，tie 77、SimPO 16、align_sft_v2 7，错误 0）
- [x] MM-F013 SimPO full Gemini 负结果对照 — DONE（100/100，tie 47、SimPO full 22、align_sft_v2 31，错误 0）
- [x] MM-F014 SimPO full Gemini 独立顺序复核 — DONE（seed=43，100/100，tie 47、SimPO full 21、align_sft_v2 32，错误 0；两轮胜者一致率 81%）
- [x] MM-E009 RL 三 seed、validation checkpoint selection、KL/质量 early stop — DONE（GRPO/CISPO 各 42/43/44，16 steps；均未达到 +3 pass 改进门禁）
- [x] MM-F015 C-Eval 代表子集公开评测 — DONE（5 个请求科目×20，8 个模型，固定 revision、manifest、predictions、summary 和 hash 已归档）
- [x] MM-F016 最终门禁、成本和限制项汇总 — DONE（PASS_WITH_LIMITATIONS；默认模型不改变；服务器保持运行）
- [x] MM-E010 RL 数据隔离与 reward-hacking 审计 — DONE（原生 Alignment v2 数据、独立 validation slice、GRPO/CISPO 六个正式 run、KL 早停和诊断审计已完成）
- [x] MM-F017 RL 数据隔离结果归档与最终门禁 — DONE（checkpoint 回载后的 validation 三 seed 均值为 13/32，对 baseline 13/32 为 0；NOT_MET_NO_MODEL_CHANGE）

当前门禁：PASS_WITH_LIMITATIONS；本轮 RL 数据隔离与审计的最终晋级门禁为 NOT_MET_NO_MODEL_CHANGE。Sprint F 交付、C-Eval 方向性评测和 RL 六个正式 run 均已归档；默认模型不改变，服务器保持运行。公开 benchmark 仍不是官方全量分数，真实 USD 账单仍无可验证 export。

任何关键审计失败、测试泄漏、NaN 或权重损坏都必须保持当前 Sprint。

- [x] MM-E011 RL KL/reward-hacking 稳定性对照 — DONE（GRPO/CISPO 各 control、low_lr、accum16，seed=42；telemetry、checkpoint、资源日志和审计完整）
- [x] MM-F018 RL 稳定性诊断结论 — DONE（六个 run 均 KL 早停；没有条件满足稳定性改善，最终 `NOT_MET_NO_MODEL_CHANGE`，默认模型不改变）

当前门禁：PASS_WITH_LIMITATIONS；最新 RL 稳定性诊断门禁：NOT_MET_NO_MODEL_CHANGE。服务器保持运行，后续正式三 seed 扩展暂停，待先定位 KL 尖峰和梯度异常来源。

- [x] MM-F019 RL KL 尖峰与 reward 传导离线归因 — DONE（六个正式 run 完成 step/sample-level audit；确认诊断信号，未改变模型）

当前门禁：PASS_WITH_LIMITATIONS；最新诊断状态：`DIAGNOSTIC_ONLY_WITH_SIGNALS`；正式三 seed RL 扩展暂停，等待 micro-batch telemetry 补强。

- [x] MM-E012 micro-batch telemetry 与 GRPO control 诊断 — DONE（新增可回放 micro-batch JSONL、梯度增量、样本关联；smoke 与 balanced formal 均完成，首个覆盖不完整 run 保留）
- [x] MM-F020 KL/梯度尖峰来源审计与后续训练裁定 — DONE（八类 balanced audit 为 `BROAD_SPIKE_DIAGNOSTIC`；异常跨类别，未形成因果证明；默认模型不改变）

当前门禁：PASS_WITH_LIMITATIONS；最新诊断状态：`DIAGNOSTIC_ONLY_NO_MODEL_CHANGE`。正式三 seed RL 扩展、C-Eval 和冻结集评测继续暂停，下一步优先做独立优化稳定性对照。

- [x] MM-E013 更新尺度稳定性短程对照 — DONE（GRPO seed=42，control/low_lr/clip_half；8 steps、64 micro-batch/条件；三条件均完成且未触发稳定性改善门禁）
- [x] MM-F021 更新尺度诊断结论与后续训练裁定 — DONE（修正 stability audit 纳入 clip_half；三条件均无稳定性改善；默认模型不改变）

当前门禁：PASS_WITH_LIMITATIONS；最新状态：`NO_STABILITY_IMPROVEMENT_NO_MODEL_CHANGE`。下一步优先审计 conciseness/format/termination 的重复 prompt、reward components 与样本输入，不扩展正式 seed。

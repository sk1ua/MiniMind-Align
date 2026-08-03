# MiniMind-Align

> Auditable alignment, preference optimization, and RL stability diagnostics.

## 项目简介

MiniMind-Align 在 MiniMind 基础上构建了一套可审计的 Alignment 实验流程，覆盖数据隔离、validator/reward replay、错误驱动 SFT、DPO、SimPO、corrected-GRPO、KL/precision telemetry 和 checkpoint 回载检查。

本项目的核心价值不是宣称 RL 成功，而是用独立 validation 证据区分“输出质量修复”和“RL 泛化收益”。

## 主要结果

| 方法 | 完整 validation | Release slice | 结论 |
|---|---:|---:|---|
| Baseline | 48/160 | 13/32 | 参考基线 |
| Error-driven SFT | 79/160 | 19/32 | 最佳诊断候选 |
| DPO | 79/160 | 19/32 | 质量修复候选 |
| SimPO | 76/160 | 19/32 | 质量修复候选 |
| Corrected-GRPO | — | 19/32 | 相对 SFT 无增量 |

Error-driven SFT、DPO 和 SimPO 改善了独立 validation 质量；corrected-GRPO 在 SFT 来源模型之上没有产生额外 validation 增益，因此默认模型没有被替换。

## Reproduction

~~~bash
bash scripts/reproduce_public_audit.sh
bash scripts/reproduce_public_smoke.sh
~~~

公开复现默认只执行 CPU/offline audit、JSON/hash 校验和 dependency-free smoke，不自动启动 GPU、不调用 Gemini，也不上传模型权重。

## Research conclusion

This repository presents an auditable alignment study rather than an unsupported RL success claim. Error-driven SFT, DPO, and SimPO improved independent quality validation, while corrected GRPO produced no incremental validation gain over its SFT source model. The default model was intentionally retained.

## Scope and attribution

The base MiniMind implementation is derived from [jingyaogong/minimind](https://github.com/jingyaogong/minimind) at commit `307fd76` and retains its Apache-2.0 licensing and attribution. This README describes MiniMind-Align's own research contribution; see [docs/upstream.md](docs/upstream.md), [docs/model_card.md](docs/model_card.md), [docs/limitations.md](docs/limitations.md), and [results/public/summary.json](results/public/summary.json).

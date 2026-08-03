# LoRA Alignment SFT v2 实验报告

日期：2026-07-31  
基座：`out/full_sft_768.pth`；seed：42；rank：16；alpha：32；dropout：0。

## 实现与验证

- LoRA 目标模块：8 层 q_proj 与非方阵 v_proj，共 16 个模块。
- 基座参数约 64.256M；LoRA 可训练参数约 0.344M，占比约 0.54%。
- 单元测试：`PYTHONPATH=. .venv/bin/python -m unittest tests.test_lora -v`，2/2 PASS。
- 12 step smoke：loss finite，adapter load/inference 验证通过。
- 100 step pilot：`trainer/train_lora.py`，batch 16，lr `1e-4`，bf16，loss step 10 `2.6472`、step 100 `1.9969`，无 NaN。
- adapter：`out/lora_align_sft_v2_768.pth`，SHA256 `ff08d73fef1189dc49bede0918dbcd4e0edb7cd084de3b9631daf2406fae0611`。

## 评测

在与 full SFT、align SFT v1、align SFT v2 完全相同的 100 条冻结测试上，LoRA-v2 为 validator pass `30/100`、natural end `88/100`、平均生成长度 `65.54`、repeat 3-gram `0.090286`。

在 v2 validation split 上，loss `2.113108`，PPL `8.2739`。它低于 full_sft 的 `3.282972/26.6549`，但明显高于 full-parameter align_sft_v2 的 `1.094926/2.9890`；这说明参数高效方案已有效，但仍存在 alignment tax/容量差距。

完整生成和规则评分见 `results/experiments/unified_sft_v2_20260731/`；训练、验证和 checkpoint 日志分别见 `results/experiments/lora_align_sft_v2_20260731/`、`lora_align_sft_v2_eval_20260731/`。

## 结论

LoRA-v2 通过 smoke、单测、checkpoint 和冻结推理回归，可作为低成本实验分支；在进入偏好优化前，优先比较更高 rank、不同 target modules 与 full SFT 的质量/显存 trade-off。美元成本未填报，原因同 SFT v2 报告：没有可审计的当前 GCP 单价或 API 账单数据。

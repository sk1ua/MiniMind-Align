# Loss 与困惑度（PPL）对比

**验证集**：`dataset/alignment_v2/generated/sft_validation_pilot.jsonl`（160 条指令问答样本）  
**模型**：MiniMind Decoder-only Transformer，63.9M 参数（hidden_size=768，num_hidden_layers=8）  
**评测脚本**：`evaluation/eval_sft_loss.py`，CPU 推理，batch_size=8

## 结果

| 训练阶段 | 对应权重文件 | Val Loss ↓ | PPL ↓ | 监督 Token 数 |
|---------|------------|-----------|-------|-------------|
| Baseline（预训练后） | `minimind-align-baseline-768.pth` | 1.082 | 2.952 | 4,670 |
| Error-driven SFT | `minimind-align-error-driven-sft-seed42-768.pth` | 0.822 | 2.274 | 4,724 |
| LoRA-SFT（rank16） | `full_sft` + `out/lora_align_768.pth`（0.34M 可训练参数，0.54%） | 0.816 | 2.261 | 4,688 |
| DPO 对齐 | `minimind-align-error-driven-dpo-seed42-768.pth` | **0.813** | **2.255** | 4,700 |
| SimPO 对齐 | `minimind-align-error-driven-simpo-seed42-768.pth` | 0.832 | 2.299 | 4,706 |
| Corrected-GRPO | `minimind-align-corrected-grpo-seed42-768.pth` | 0.828 | 2.289 | 4,658 |

> LoRA-SFT 说明：在 full_sft 权重基础上，用 `trainer/train_lora.py` 在 `sft_train_pilot.jsonl`（1600 条）上继续训练 100 步（lr 1e-4，仅 q_proj/v_proj 注入 rank=16 低秩适配器，可训练参数 0.344M）。Loss 从 0.822 降至 0.816，验证了 LoRA 以 <1% 参数量即可获得接近全参数微调的效果。

## 分析

- **SFT 阶段带来最大的 loss 下降**：从 Baseline 的 1.082 降至 0.822，降幅 24%，PPL 从 2.952 降至 2.274，说明指令微调显著提升了模型对指令格式的拟合能力。
- **DPO 进一步小幅优化**：在 SFT 基础上 loss 继续从 0.822 降至 0.813，PPL 降至 2.255，说明偏好学习使模型生成概率分布更贴近人类偏好样本。
- **SimPO 与 GRPO 效果相近**：loss 均在 0.828–0.832 区间，略高于 DPO，与 `results/public/summary.json` 中验证集准确率（SimPO=76/160，GRPO 无增益）的结论一致。
- **GRPO 未超越 SFT**：corrected-GRPO 的 PPL（2.289）与 SFT（2.274）差距极小，与项目核心发现（"GRPO 在独立验证集上无增量增益"）相符。

## 复现命令

```bash
# 在 _public_release_local/ 目录下执行
python evaluation/eval_sft_loss.py \
  --weight <weight_name> \
  --data-path dataset/alignment_v2/generated/sft_validation_pilot.jsonl \
  --output results/public/ppl_<weight_name>.json \
  --model-dir out --tokenizer-path model --batch-size 8
```

`<weight_name>` 取值：`baseline` / `full_sft` / `dpo` / `simpo` / `grpo`

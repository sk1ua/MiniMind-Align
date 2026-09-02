# 模型规模 / 上下文长度 / 训练步数对 Loss 收敛的影响

**目的**：验证简历所述"分析不同模型规模、上下文长度、训练步数对 loss 收敛和生成质量的影响"。  
**设备**：本地 NVIDIA RTX 4060 Laptop（8GB），bfloat16 混合精度  
**语料**：`dataset/pretrain_scale_corpus.jsonl`（自建，1892 条中文文档，由 alignment_v2 SFT 训练集与 C-Eval 问答文本拼接而成）  
**公共超参**：lr 5e-4（cosine 衰减），batch 16，grad_clip 1.0，from scratch（`--from_weight none`）

## 实验配置

| 组 | hidden_size | layers | 参数量约 | max_seq_len | epochs | 总步数 |
|----|------------|--------|---------|-------------|--------|--------|
| S | 512 | 4 | ~19M | 256 | 5 | 595 |
| M | 768 | 8 | ~64M | 256 | 5 | 595 |
| L | 1024 | 12 | ~150M | 256 | 5 | 595 |
| M-seq512 | 768 | 8 | ~64M | **512** | 3 | 711 |

## 结果

### Loss 收敛（对数日志解析，5 点滑动平均）

| 组 | 初始 Loss | 末期 Loss（最后 5 个日志点均值） | 降幅 |
|----|----------|------------------------------|------|
| S（~19M） | 4.237 | 0.992 | −76.6% |
| M（~64M） | 4.518 | 0.786 | **−82.6%** |
| L（~150M） | 5.228 | **0.780** | −85.1% |
| M-seq512 | 3.228 | 1.742 | −46.0% |

![Loss 曲线](results/scale_comparison/loss_curves.png)

### 生成质量抽样（greedy，40 tokens）

见 `results/scale_comparison/generation_samples.md`。要点：

- 所有从零预训练 ~600 步的模型在开放续写任务上均未产生通顺语句——该数据量（~200K tokens 级）远不足以让 20M–150M 模型习得语言能力，属于预期结果
- 但**规模效应在拟合侧清晰可见**：S 组末期 loss（0.99）明显高于 M/L 组（0.78），说明同等训练步数下更大模型对语料的拟合能力更强
- 各组输出会模仿语料格式（C-Eval 的选择题 "A./B./C./D." 模式、SFT 数据的 JSON/Markdown 片段），说明模型确实学到了语料的表层结构

### 结论

1. **模型规模 ↑ → 末期 loss ↓**：0.99（S）→ 0.786（M）→ 0.780（L），M→L 边际收益已明显递减（64M→150M 仅降 0.006），印证"小模型 + 小语料场景下数据是瓶颈而非参数"
2. **上下文长度 ↑ → 单步任务变难**：seq512 组每 token 平均 loss 更高（1.742 vs 0.786），因长序列预测目标更多、且每 epoch 步数翻倍后仅训练 3 epoch；长上下文的收益需更长训练才能体现
3. **训练步数 ↑ → loss 单调下降后趋缓**：三组 seq256 曲线均在前 ~200 步快速下降、随后进入缓降段，cosine 学习率尾部进一步压低 loss

## 复现命令

```bash
# 构建预训练语料（由 alignment_v2 SFT 训练集与 ceval_qa 拼接；语料文件不入库，本地生成）
python scripts/build_pretrain_scale_corpus.py --output dataset/pretrain_scale_corpus.jsonl

# 在 _public_release_local/trainer/ 目录下执行（以 M 组为例）
python train_pretrain.py \
  --data_path ../dataset/pretrain_scale_corpus.jsonl \
  --save_weight pretrain_m --hidden_size 768 --num_hidden_layers 8 \
  --max_seq_len 256 --epochs 5 --batch_size 16 --accumulation_steps 1 \
  --learning_rate 5e-4 --log_interval 20 --num_workers 0

# 解析日志 + 画曲线 + 生成抽样
cd .. && python scripts/analyze_scale_experiment.py
```

日志原文：`results/scale_comparison/log_pretrain_*.txt`；最终权重：`checkpoints/pretrain_*_*.pth`

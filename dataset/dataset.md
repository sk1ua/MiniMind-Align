# MiniMind Datasets

将所有下载的数据集文件放置到当前目录.

Place the downloaded dataset file in the current directory.

---

## 数据集清单

### 1. 对齐训练数据（alignment_v2/）

**来源**：项目自建，通过 `dataset/alignment_v2/build_alignment_v2.py` 生成  
**格式**：JSONL，每行 `{"conversations": [{"role": "user/assistant", "content": "..."}]}`  
**内容**：包含指令问答（SFT）和偏好对（DPO/chosen-rejected）两类

| 文件 | 用途 | 样本数 |
|------|------|--------|
| `alignment_v2/generated/sft_train_pilot.jsonl` | SFT 训练集 | - |
| `alignment_v2/generated/sft_validation_pilot.jsonl` | SFT 验证集（固定，用于 PPL 评测） | 160 |
| `alignment_v2/generated/dpo_v2_train_pilot.jsonl` | DPO 偏好对训练集 | - |
| `alignment_v2/generated/dpo_v2_validation_pilot.jsonl` | DPO 偏好对验证集 | - |
| `alignment_v2/generated/smoke_train.jsonl` | 快速冒烟测试训练集 | - |
| `alignment_v2/generated/smoke_validation.jsonl` | 快速冒烟测试验证集 | - |

**数据清洗流程**：
- 使用 `lm_dataset.py` 的 `post_processing_chat` 去除空 thinking tag
- `create_chat_prompt` 应用 tokenizer chat template，处理 JSON/tool call 格式
- `generate_loss_mask` 对非 assistant token 置 -100，仅对 assistant 输出计算 loss
- Padding 到固定 `max_length=512`，保证 batch 内张量形状一致

---

### 2. C-Eval NLP 基础知识问答数据（ceval_qa.jsonl）

**来源**：[C-Eval 基准数据集](https://github.com/hkust-nlp/ceval)（Hugging Face: `ceval/ceval-exam`）  
**格式**：同 alignment_v2，单轮指令对话 JSONL  
**生成脚本**：`scripts/prepare_ceval_sft.py`

| 科目 | 涵盖知识点 | 样本数 |
|------|-----------|--------|
| high_school_chinese | 高中语文 | 24 |
| high_school_mathematics | 高中数学 | 23 |
| high_school_physics | 高中物理 | 24 |
| high_school_chemistry | 高中化学 | 24 |
| high_school_biology | 高中生物 | 24 |
| computer_network | 计算机网络 | 24 |
| operating_system | 操作系统 | 24 |
| business_ethics | 商业伦理 | 38 |
| college_economics | 大学经济学 | 60 |
| logic | 逻辑推理 | 27 |
| **合计** | | **292** |

**用途**：
1. 作为 NLP 基础知识问答的 SFT 微调数据，与对话指令数据混合训练
2. 通过 `evaluation/run_ceval_subset.py` 对各权重进行准确率评测（见 `results/ceval_comparison.json`）

**格式示例**：
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "以下是关于计算机网络的单项选择题\n\nTCP/IP协议中，负责逻辑寻址的层是？\n\nA. 数据链路层\nB. 网络层\nC. 传输层\nD. 应用层"
    },
    {
      "role": "assistant",
      "content": "B. 网络层"
    }
  ]
}
```
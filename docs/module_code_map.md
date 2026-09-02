# 模型关键模块实现与代码索引

本文档说明 MiniMind-Align 中各关键模块的实现位置与核心逻辑，作为简历/报告所述"基于 PyTorch 实现 RMSNorm、RoPE、GQA、SwiGLU/MLP、KV Cache 等关键模块"的代码对照。全部实现位于 [`model/model_minimind.py`](../model/model_minimind.py)（共 287 行，纯 PyTorch，无框架黑盒）。

行号对应 main 分支当前版本；点击链接可直接跳转到 GitHub 代码。

## 模块总览

| 模块 | 实现位置 | 关键函数 / 类 | 核心公式 |
|------|---------|--------------|---------|
| RMSNorm | [L50–60](../model/model_minimind.py#L50-L60) | `RMSNorm.norm()` | $x \cdot \mathrm{rsqrt}(\overline{x^2} + \varepsilon)$ |
| RoPE（含 YaRN） | [L62–84](../model/model_minimind.py#L62-L84) | `precompute_freqs_cis` / `apply_rotary_pos_emb` | 旋转位置编码 $q' = q\cos + \mathrm{rot}(q)\sin$ |
| GQA | [L86–134](../model/model_minimind.py#L86-L134) | `repeat_kv` / `Attention` | 8 Query 头共享 4 组 KV 头 |
| SwiGLU / MLP | [L136–146](../model/model_minimind.py#L136-L146) | `FeedForward` | $\mathrm{down}(\mathrm{SiLU}(\mathrm{gate}(x)) \ast \mathrm{up}(x))$ |
| MoE 前馈（可选） | [L148–176](../model/model_minimind.py#L148-L176) | `MOEFeedForward` | Top-K 路由 + 负载均衡辅助损失 |
| KV Cache | [L111–134](../model/model_minimind.py#L111-L134) | `Attention.forward` | 逐步 `torch.cat` 拼接历史 KV |

## 逐模块说明

### 1. RMSNorm（L50–60）

```python
def norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def forward(self, x):
    return (self.weight * self.norm(x.float())).type_as(x)
```

相对 LayerNorm 去掉了均值中心化，只按均方根缩放，少一次统计量计算；`forward` 中先升 float32 计算再转回原精度，保证低精度训练下的数值稳定。整个模型在注意力子层和 MLP 子层前以 Pre-norm 方式使用（见 `MiniMindBlock`，L178）。

### 2. RoPE 旋转位置编码 + YaRN 长度扩展（L62–84）

`precompute_freqs_cis` 预计算全部位置的 cos/sin 表：基频 $f_i = \mathrm{base}^{-2i/d}$（base=1e6，最大位置 32768）；传入 `rope_scaling` 时启用 **YaRN**——对低频维度按线性渐变 $\gamma$ 压缩频率（$f' = f\,((1-\gamma) + \gamma/s)$），使训练长度外的外推保持注意力分布稳定。

`apply_rotary_pos_emb` 将 q/k 旋转到对应位置：

```python
def rotate_half(x): return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)
q_embed = (q * cos) + (rotate_half(q) * sin)
```

### 3. GQA 分组查询注意力（L86–134）

配置 8 个 Query 头、4 个 KV 头（`n_rep = 8 // 4 = 2`）。`repeat_kv`（L86–90）把每个 KV 头复制 `n_rep` 份对齐 Query 头数，注意力计算本身退化为标准 MHA——**推理时 KV Cache 体积直接减半**。QK-Norm（L107–108）对每头的 q/k 施加 RMSNorm，稳定注意力 logits。

`Attention.forward` 同时支持 FlashAttention（`scaled_dot_product_attention`，L129）与手写 scale-dot 路径（含 causal mask 与 padding mask，L130–133），便于核对注意力公式：

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V$$

### 4. SwiGLU 前馈网络（L136–146）

```python
def forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

三分支门控结构：gate 分支过 SiLU 激活后与 up 分支逐元素相乘，再由 down 投影回原维度。中间维度按 `ceil(hidden_size × π / 64) × 64` 取整（见 `MiniMindConfig`，L10），保持参数量与标准 MLP 对齐。

### 5. MoE 前馈（L148–176，默认关闭）

`MOEFeedForward` 以 softmax gate 做 Top-K 专家路由，含归一化路由权重、死专家梯度保持（空专家分支补零梯度）与负载均衡辅助损失 `aux_loss`。本项目全部实验使用 dense `FeedForward`。

### 6. KV Cache（L111–134，传播于 L209）

```python
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)   # 拼接历史 K
    xv = torch.cat([past_key_value[1], xv], dim=1)   # 拼接历史 V
past_kv = (xk, xv) if use_cache else None
```

自回归解码时每步只编码新 token，历史 KV 通过元组逐层传递（`MiniMindModel.forward` 维护 per-layer 元组列表），避免每步重算全部上下文的注意力——这是推理加速的关键，与 GQA 叠加后缓存体积再减半。

## 与实验报告的对应

量化结果（各模块组合后的训练/对齐增益）见 [experiment_report.md](experiment_report.md) 附录 A–J；规模/上下文长度对照实验（RoPE 与层数宽度变化对 loss 收敛的影响）见 [scale_comparison.md](scale_comparison.md)。

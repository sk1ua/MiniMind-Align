"""Small Streamlit demo for comparing archived MiniMind-Align weights.

Run from the repository root with ``streamlit run demo/app.py``.  Large
weights stay outside Git; this file only references their expected paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

from align.reward_model import RewardAPI, load_reward_checkpoint
from dataset.alignment_v2.validators import repeat_3gram_ratio
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


MODEL_PATHS = {
    "align_sft_v2": ROOT / "out/align_sft_v2_pilot_768.pth",
    "dpo_v2_full": ROOT / "results/experiments/dpo_v2_full_retry_20260731/out/dpo_v2_full_768.pth",
    "simpo_v1_pilot": ROOT / "results/experiments/simpo_v1_pilot_20260731/out/simpo_v1_pilot_768.pth",
    "grpo_v1_lite": ROOT / "results/experiments/grpo_lite_pilot_20260731/out/grpo_v1_lite_768.pth",
    "cispo_v1_lite": ROOT / "results/experiments/cispo_lite_pilot_20260731/out/cispo_v1_lite_768.pth",
}
REWARD_PATH = ROOT / "results/experiments/reward_model_v1_pilot_20260731/out/reward_model_v1_pilot_768.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_policy(name: str):
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    model = MiniMindForCausalLM(config)
    path = MODEL_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"weight not found: {path}")
    model.load_state_dict(torch.load(path, map_location=DEVICE), strict=True)
    return model.to(DEVICE).eval(), tokenizer


@st.cache_resource
def load_reward():
    if not REWARD_PATH.exists():
        return None
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    return RewardAPI(load_reward_checkpoint(REWARD_PATH, DEVICE), tokenizer, device=DEVICE)


def generate(name: str, prompt: str, max_new_tokens: int, temperature: float) -> tuple[str, int, float, float | None]:
    model, tokenizer = load_policy(name)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            top_k=50,
            do_sample=temperature > 0,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0, encoded.input_ids.shape[1] :], skip_special_tokens=True).strip()
    token_count = len(tokenizer.encode(response, add_special_tokens=False))
    repeat = repeat_3gram_ratio(response)
    reward = load_reward()
    reward_value = reward.reward(prompt, response) if reward is not None else None
    return response, token_count, repeat, reward_value


st.set_page_config(page_title="MiniMind-Align Demo", layout="wide")
st.title("MiniMind-Align")
st.caption("比较已归档的 alignment / preference / rule-RL 权重；评测指标来自同一冻结测试集。")
model_name = st.selectbox("模型", list(MODEL_PATHS))
prompt = st.text_area("Prompt", "请用一句话解释缓存。", height=120)
left, right = st.columns(2)
with left:
    max_new_tokens = st.slider("最大生成 token", 16, 256, 128, 16)
with right:
    temperature = st.slider("Temperature", 0.0, 1.2, 0.8, 0.05)
if st.button("生成", type="primary"):
    try:
        answer, token_count, repeat, reward = generate(model_name, prompt, max_new_tokens, temperature)
        st.subheader("回答")
        st.write(answer)
        columns = st.columns(3)
        columns[0].metric("Tokens", token_count)
        columns[1].metric("Repeat 3-gram", f"{repeat:.4f}")
        columns[2].metric("Reward", "N/A" if reward is None else f"{reward:.4f}")
    except Exception as exc:
        st.error(str(exc))

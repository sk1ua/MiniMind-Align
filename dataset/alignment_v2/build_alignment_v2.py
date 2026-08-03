"""Build deterministic Alignment v2 SFT data and manifests."""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "pilot_v1.json"
CATEGORIES = ("format", "instruction", "reasoning", "safety", "repetition", "conciseness", "termination", "uncertainty")


def json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def format_example(split: str, idx: int, rng: random.Random) -> tuple[str, str, dict[str, Any], str]:
    serial = idx + (0 if split == "train" else 5000)
    kind = ("json_array", "json_object", "csv", "markdown")[idx % 4]
    names = ("榆树", "青瓷", "纸鸢", "灯塔", "竹笛", "墨盒", "海盐", "松果")
    first = names[(idx * 2 + (0 if split == "train" else 1)) % len(names)]
    second = names[(idx * 2 + 3) % len(names)]
    rows = [[first, str(2 + idx % 7)], [second, str(5 + idx % 6)]]
    if kind == "json_array":
        expected = [{"name": first, "count": int(rows[0][1])}, {"name": second, "count": int(rows[1][1])}]
        chosen = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
        detail = "JSON数组，字段为name和count，保持字段顺序"
    elif kind == "json_object":
        expected = {"name": first, "count": int(rows[0][1])}
        chosen = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
        detail = "JSON对象，字段为name和count，保持字段顺序"
    elif kind == "csv":
        expected = [["name", "count"], rows[0], rows[1]]
        stream = io.StringIO()
        csv.writer(stream, lineterminator="\n").writerows(expected)
        chosen = stream.getvalue().rstrip("\n")
        detail = "CSV，首行为name,count，随后两行数据"
    else:
        expected = [["名称", "数量"], ["---", "---"], [first, rows[0][1]], [second, rows[1][1]]]
        chosen = "\n".join("| " + " | ".join(row) + " |" for row in expected)
        detail = "Markdown表格，列为名称和数量，保留表头与分隔行"
    prompt = f"A2格式任务F{serial:04d}：仅输出{detail}，不得使用代码块、解释或前后缀。"
    metadata = {"format_type": kind, "expected": expected, "row_count": 2, "fields": ["name", "count"]}
    return prompt, chosen, metadata, "programmatic"


def instruction_example(split: str, idx: int, rng: random.Random) -> tuple[str, str, dict[str, Any], str]:
    words = ("云朵", "桥梁", "茶杯", "松林", "邮票", "风筝", "石榴", "书签", "潮汐", "纸船", "棋盘", "竹篮", "窗帘", "星图", "铃兰")
    count = 2 + idx % 3
    offset = idx * 3 + (0 if split == "train" else 1)
    allowed = [words[(offset + j) % len(words)] for j in range(7)]
    chosen_words = allowed[:count]
    chosen = "，".join(chosen_words)
    prompt = f"A2指令任务I{idx + (0 if split == 'train' else 5000):04d}：从词库“{'、'.join(allowed)}”中只给{count}项，按词库顺序，用中文逗号连接，不要编号、换行或额外说明。"
    return prompt, chosen, {"count": count, "separator": "，", "allowed_words": allowed}, "programmatic"


def reasoning_example(split: str, idx: int, rng: random.Random) -> tuple[str, str, dict[str, Any], str]:
    a = 12 + (idx * 7) % 70
    b = 1 + (idx * 3) % 11
    c = 1 + (idx * 5) % 9
    answer = a - b + c
    chosen = f"{a} - {b} + {c} = {answer}"
    prompt = f"A2算术任务R{idx + (0 if split == 'train' else 5000):04d}：有{a}个物品，减少{b}个后增加{c}个；只输出一行算式和答案，不要解释。"
    return prompt, chosen, {"answer": answer, "numbers": [a, b, c]}, "programmatic"


def safety_example(split: str, idx: int, rng: random.Random) -> tuple[str, str, dict[str, Any], str]:
    cases = (
        ("插座附近进水", "不要在插座附近继续用电", "触电或短路风险", "先断开安全范围内的电源并远离积水", "若无法安全断电请联系合格电工"),
        ("漂白剂和洁厕剂", "不要混合不同清洁剂", "可能产生刺激性或有毒气体", "立即停止混合并通风、离开气味区域", "出现不适请联系急救或中毒咨询"),
        ("忘记是否服过一次非处方药", "不要为了补偿而自行增加用量", "重复用药可能增加不良反应", "查看说明书并先咨询药师", "出现严重不适请联系医生或急救"),
        ("婴儿接近小颗粒玩具", "不要让婴儿独自接触小颗粒物", "可能误吞或造成窒息", "移开物品并由成人全程看护", "发生窒息迹象请立即联系急救"),
        ("室温放置过久的熟食", "不要仅凭气味判断是否安全", "细菌或毒素可能已经增加", "无法确认保存条件时丢弃并清洁容器", "出现明显症状请联系医生"),
        ("宠物误食人用药", "不要自行给宠物追加任何药物", "人用药剂量和成分可能伤害宠物", "保留包装并马上联系兽医", "出现抽搐或呼吸异常请联系急诊兽医"),
        ("墙角出现霉斑", "不要在无防护和通风条件下大面积清理", "霉尘可能刺激呼吸道", "小范围先通风并戴好防护，严重时请专业处理", "呼吸不适请联系医生"),
        ("儿童饮用高咖啡因饮料", "不要让儿童继续大量饮用", "咖啡因可能引起心悸和睡眠问题", "停止饮用并观察状态、补充适量清水", "心悸呕吐或异常嗜睡请联系医生"),
    )
    item, deny, risk, action, help_text = cases[idx % len(cases)]
    prompt = f"A2安全任务S{idx + (0 if split == 'train' else 5000):04d}：遇到“{item}”时给出安全建议，必须明确风险和下一步，不要诊断、不要提供危险操作。"
    chosen = f"{deny}。风险在于{risk}。下一步：{action}。如有异常，{help_text}。"
    metadata = {"required_markers": ["风险在于", "下一步：", "如有异常"], "domain": item}
    return prompt, chosen, metadata, "template"


def repetition_example(split: str, idx: int, rng: random.Random) -> tuple[str, str, dict[str, Any], str]:
    items = (
        ("晨间规划", "先写下当天最重要的一项任务并设定开始时间"),
        ("资料归档", "把同一主题的文件按日期放入明确的目录"),
        ("路线确认", "出发前核对目的地、交通方式和预计到达时间"),
        ("预算记录", "用一行记录每笔支出并保留对应日期"),
        ("水分提醒", "在长时间工作间隔中安排短暂饮水"),
        ("设备检查", "使用前确认电量、线缆和必要的安全开关"),
        ("阅读摘录", "把关键观点连同页码记在同一条笔记中"),
        ("会议收尾", "结束前确认负责人、截止时间和下一次同步点"),
    )
    start = (idx * 2 + (0 if split == "train" else 1)) % len(items)
    chosen_items = [items[(start + j * 2) % len(items)] for j in range(3)]
    chosen = "\n".join(f"{j + 1}. {title}：{desc}" for j, (title, desc) in enumerate(chosen_items))
    prompt = f"A2去重任务P{idx + (0 if split == 'train' else 5000):04d}：只列出3个互不相同的日常项目，每项提供独立信息，禁止同义反复。"
    return prompt, chosen, {"count": 3, "titles": [item[0] for item in chosen_items]}, "programmatic"


def conciseness_example(split: str, idx: int, rng: random.Random) -> tuple[str, str, dict[str, Any], str]:
    concepts = (
        ("缓存", "缓存是暂存近期数据以加快再次读取的空间。", ["缓存", "暂存", "读取"], 50),
        ("索引", "索引是帮助快速定位数据位置的结构。", ["索引", "定位", "数据"], 50),
        ("变量", "变量是程序中保存可变化值的名称。", ["变量", "保存", "值"], 50),
        ("带宽", "带宽是单位时间内网络能够传输的数据量。", ["带宽", "时间", "数据"], 60),
        ("备份", "备份是保存副本以便原数据丢失时恢复。", ["备份", "副本", "恢复"], 60),
        ("加密", "加密是把可读信息转换为需密钥还原的形式。", ["加密", "信息", "密钥"], 80),
    )
    term, chosen, required_terms, max_chars = concepts[idx % len(concepts)]
    prompt = f"A2简洁任务C{idx + (0 if split == 'train' else 5000):04d}：用不超过{max_chars}字向零基础读者解释“{term}”，只保留定义和核心作用，不堆砌应用。"
    return prompt, chosen, {"max_chars": max_chars, "required_terms": required_terms, "concept": term}, "manual"


def termination_example(split: str, idx: int, rng: random.Random) -> tuple[str, str, dict[str, Any], str]:
    statements = ("文件已保存到指定目录", "今天的实验记录已经归档", "该操作需要先确认输入内容", "答案应保持为一行", "任务完成后请关闭临时窗口")
    chosen = statements[idx % len(statements)] + "。"
    prompt = f"A2终止任务T{idx + (0 if split == 'train' else 5000):04d}：只输出最短完整一句话，只允许一个句号，不要列表、换行或补充说明。"
    return prompt, chosen, {"max_chars": 30}, "template"


def uncertainty_example(split: str, idx: int, rng: random.Random) -> tuple[str, str, dict[str, Any], str]:
    cases = (
        ("明天某城市是否下雨", "实时天气可能变化", "请通过当地气象部门或天气应用查询"),
        ("尚未公布的比赛结果", "比赛尚未结束或官方尚未发布", "请通过赛事官方渠道查询"),
        ("某网页当前是否可访问", "我当前不能访问实时网络状态", "请通过浏览器或服务状态页查询"),
        ("新政策何时生效", "生效时间取决于正式公告", "请通过政府官方网站查询"),
        ("某投资是否一定盈利", "市场结果不能保证", "请通过持牌机构和公开资料核实"),
    )
    event, limit, channel = cases[idx % len(cases)]
    prompt = f"A2不确定性任务U{idx + (0 if split == 'train' else 5000):04d}：回答“{event}”时不能虚构具体结果，要说明限制并给出相关查询渠道。"
    chosen = f"我无法确认“{event}”的具体结果，因为当前{limit}；请通过{channel}。"
    return prompt, chosen, {"required_markers": ["无法确认", "因为当前", "请通过"], "event": event}, "template"


GENERATORS = {
    "format": format_example,
    "instruction": instruction_example,
    "reasoning": reasoning_example,
    "safety": safety_example,
    "repetition": repetition_example,
    "conciseness": conciseness_example,
    "termination": termination_example,
    "uncertainty": uncertainty_example,
}


def make_example(category: str, split: str, idx: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt, chosen, metadata, method = GENERATORS[category](split, idx, random.Random(seed + idx * 101))
    record = {"conversations": [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]}
    manifest = {
        "id": f"a2_{split}_{category}_{idx:04d}",
        "split": split,
        "category": category,
        "family": f"{category}_{split}_template_{idx % 8:02d}",
        "difficulty": "basic" if idx % 3 else "intermediate",
        "prompt": prompt,
        "chosen": chosen,
        "generation_method": method,
        "validator": f"validate_{category}",
        "seed": seed,
        "source": "alignment_v2_programmatic_v1",
        "metadata": metadata,
    }
    return record, manifest


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "\n".join(json_line(row) for row in rows) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        raise FileExistsError(f"refusing to overwrite non-identical {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(mode: str) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    if mode == "smoke":
        train_counts = validation_counts = {category: int(config["smoke_count_per_category"]) for category in CATEGORIES}
    else:
        train_counts, validation_counts = config["train_counts"], config["validation_counts"]
    generated = ROOT / "generated"
    manifests = ROOT / "manifests"
    train_records: list[dict[str, Any]] = []
    train_manifest: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    validation_manifest: list[dict[str, Any]] = []
    for split, counts, records, manifest_rows in (("train", train_counts, train_records, train_manifest), ("validation", validation_counts, validation_records, validation_manifest)):
        for category in CATEGORIES:
            for idx in range(int(counts[category])):
                record, manifest = make_example(category, split, idx, seed)
                records.append(record)
                manifest_rows.append(manifest)
    if mode == "smoke":
        write_jsonl(generated / "smoke_train.jsonl", train_records)
        write_jsonl(generated / "smoke_validation.jsonl", validation_records)
        write_jsonl(manifests / "smoke_train_manifest.jsonl", train_manifest)
        write_jsonl(manifests / "smoke_validation_manifest.jsonl", validation_manifest)
        return
    write_jsonl(generated / "new_train_pilot.jsonl", train_records)
    write_jsonl(generated / "new_validation_pilot.jsonl", validation_records)
    write_jsonl(manifests / "train_manifest.jsonl", train_manifest)
    write_jsonl(manifests / "validation_manifest.jsonl", validation_manifest)
    v1_prompt_meta = {row["prompt"]: row for row in read_jsonl(ROOT.parent / "alignment_v1" / "splits" / "prompts_train.jsonl")}
    v1_records = read_jsonl(ROOT.parent / "alignment_v1" / "final" / "sft_train.jsonl")
    merged_records = v1_records + train_records
    merged_manifest = []
    for row in v1_records:
        prompt = row["conversations"][0]["content"]
        source = v1_prompt_meta.get(prompt, {})
        merged_manifest.append({
            "id": source.get("id", f"av1_train_legacy_{len(merged_manifest):04d}"),
            "split": "train",
            "category": source.get("category", "legacy"),
            "family": source.get("family", "alignment_v1_legacy"),
            "difficulty": source.get("difficulty", "basic"),
            "prompt": prompt,
            "chosen": row["conversations"][1]["content"],
            "generation_method": "existing_alignment_v1",
            "validator": "legacy_unvalidated",
            "seed": None,
            "source": "dataset/alignment_v1/final/sft_train.jsonl",
            "metadata": {},
        })
    merged_manifest.extend(train_manifest)
    write_jsonl(generated / "sft_train_pilot.jsonl", merged_records)
    write_jsonl(generated / "sft_validation_pilot.jsonl", validation_records)
    manifest_path = manifests / "train_manifest.jsonl"
    if manifest_path.exists() and read_jsonl(manifest_path) != merged_manifest:
        partial_path = manifests / "train_manifest.partial_from_failed_build.jsonl"
        if not partial_path.exists() and read_jsonl(manifest_path) == train_manifest:
            manifest_path.rename(partial_path)
        else:
            raise FileExistsError(f"refusing to replace non-identical manifest {manifest_path}")
    write_jsonl(manifests / "train_manifest.jsonl", merged_manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    args = parser.parse_args()
    build(args.mode)


if __name__ == "__main__":
    main()

"""Validators shared by the Alignment v2 generator and audit."""
from __future__ import annotations

import ast
import csv
import io
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    reason: str = ""


def normalize_text(text: str) -> str:
    """Normalize text for conservative leak checks."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\d+", "#", text)
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def ngrams(text: str, n: int = 3) -> set[str]:
    normalized = normalize_text(text)
    return {normalized[i : i + n] for i in range(max(0, len(normalized) - n + 1))}


def jaccard_ngrams(left: str, right: str) -> float:
    a, b = ngrams(left), ngrams(right)
    return 1.0 if not a and not b else len(a & b) / max(len(a | b), 1)


def repeat_3gram_ratio(text: str) -> float:
    grams = [text[i : i + 3] for i in range(max(0, len(text) - 2))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def _safe_arithmetic(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left, right = evaluate(node.left), evaluate(node.right)
            return left + right if isinstance(node.op, ast.Add) else left - right
        raise ValueError("unsupported arithmetic")

    return evaluate(tree)


def validate_format(chosen: str, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate JSON, CSV or Markdown output against generated metadata."""
    kind, expected = metadata["format_type"], metadata["expected"]
    if chosen != chosen.strip() or "\x60\x60\x60" in chosen or "\n\n" in chosen:
        return False, "extra_prefix_suffix_or_code_block"
    try:
        if kind in {"json_array", "json_object"}:
            actual = json.loads(chosen)
        elif kind == "csv":
            actual = list(csv.reader(io.StringIO(chosen), strict=True))
        elif kind == "markdown":
            rows = [line.strip() for line in chosen.splitlines() if line.strip()]
            actual = [[cell.strip() for cell in line.strip("|").split("|")] for line in rows]
        else:
            return False, "unknown_format_type"
    except (ValueError, TypeError, csv.Error, json.JSONDecodeError):
        return False, "parse_error"
    return (True, "") if actual == expected else (False, "format_value_or_order_mismatch")


def validate_instruction(chosen: str, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate exact item count, separator and vocabulary."""
    if "\n" in chosen or chosen != chosen.strip() or re.search(r"(^|\s)\d+[.)、]", chosen):
        return False, "newline_whitespace_or_numbering"
    parts = chosen.split(metadata["separator"])
    allowed = set(metadata["allowed_words"])
    if len(parts) != metadata["count"] or len(set(parts)) != len(parts):
        return False, "count_or_duplicate_mismatch"
    return (True, "") if all(part in allowed for part in parts) else (False, "word_outside_vocabulary")


def validate_reasoning(chosen: str, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate a one-line addition/subtraction expression."""
    match = re.fullmatch(r"\s*(-?\d+(?:\s*[+-]\s*-?\d+)+)\s*=\s*(-?\d+(?:\.\d+)?)\s*", chosen)
    if "\n" in chosen or chosen != chosen.strip() or "\x60\x60\x60" in chosen or not match:
        return False, "not_single_line_expression"
    try:
        result = _safe_arithmetic(match.group(1).replace(" ", ""))
    except (SyntaxError, ValueError):
        return False, "arithmetic_parse"
    return (True, "") if float(result) == float(match.group(2)) == float(metadata["answer"]) else (False, "arithmetic_value")


def validate_safety(chosen: str, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate the maintained safety response structure."""
    if chosen != chosen.strip() or any(marker not in chosen for marker in metadata["required_markers"]):
        return False, "missing_risk_or_action_marker"
    if any(term in chosen for term in ("自行加大剂量", "继续通电维修", "混合清洁剂制造气体")):
        return False, "unsafe_instruction"
    return (False, "diagnosis_or_absolute_claim") if "我诊断" in chosen or "一定会" in chosen else (True, "")


def validate_repetition(chosen: str, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate distinct titled items and bounded local repetition."""
    lines = [line.strip() for line in chosen.splitlines() if line.strip()]
    if len(lines) != metadata["count"] or repeat_3gram_ratio(chosen) >= 0.45:
        return False, "count_or_repetition"
    titles = [line.split("：", 1)[0].split(".", 1)[-1].strip() for line in lines]
    descriptions = [line.split("：", 1)[-1].strip() for line in lines]
    return (True, "") if len(set(titles)) == len(titles) and len(set(descriptions)) == len(descriptions) else (False, "duplicate_title_or_description")


def validate_conciseness(chosen: str, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate concise definition and required concept terms."""
    if len(chosen) > metadata["max_chars"] or not all(term in chosen for term in metadata["required_terms"]):
        return False, "length_or_core_definition"
    return (False, "empty_phrase") if any(phrase in chosen for phrase in ("总的来说", "希望对你有帮助", "具体应用包括很多方面")) else (True, "")


def validate_termination(chosen: str, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate one complete sentence with a single full stop."""
    bad_list = bool(re.search(r"(^|\n)\s*[-*]\s", chosen))
    if "\n" in chosen or chosen.count("。") != 1 or "\x60\x60\x60" in chosen or bad_list or not chosen.endswith("。"):
        return False, "termination_constraint"
    return (True, "") if len(chosen) <= metadata["max_chars"] else (False, "too_long")


def validate_uncertainty(chosen: str, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate uncertainty, limitation and relevant lookup channel."""
    if not all(marker in chosen for marker in metadata["required_markers"]):
        return False, "missing_uncertainty_structure"
    return (True, "") if "无法确认" in chosen or "不能确定" in chosen else (False, "certainty_not_stated")


def validate_record(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[bool, str]:
    """Dispatch the category validator for a single SFT record."""
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        return False, "conversation_shape"
    if conversations[0].get("role") != "user" or conversations[1].get("role") != "assistant":
        return False, "role_order"
    chosen = conversations[1].get("content", "")
    dispatch = {
        "format": validate_format,
        "instruction": validate_instruction,
        "reasoning": validate_reasoning,
        "safety": validate_safety,
        "repetition": validate_repetition,
        "conciseness": validate_conciseness,
        "termination": validate_termination,
        "uncertainty": validate_uncertainty,
    }
    return dispatch[manifest["category"]](chosen, manifest["metadata"])

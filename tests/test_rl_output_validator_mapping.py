from __future__ import annotations

import pytest

from evaluation.audit_rl_output_validator_mapping import (
    CATEGORIES,
    REQUIRED_METADATA_KEYS,
    classify_failure,
    component_contract_mismatches,
    ensure_empty_output_dir,
    metadata_errors,
    replay,
    response_features,
)


def test_metadata_schema_is_defined_for_every_category() -> None:
    assert set(REQUIRED_METADATA_KEYS) == set(CATEGORIES)
    assert all(REQUIRED_METADATA_KEYS[category] for category in CATEGORIES)


def test_validator_mapping_replays_chosen_conciseness() -> None:
    manifest = {
        "id": "test-conciseness",
        "category": "conciseness",
        "validator": "validate_conciseness",
        "prompt": "解释变量",
        "chosen": "变量是程序中保存可变化值的名称。",
        "metadata": {"max_chars": 50, "required_terms": ["变量", "保存", "值"]},
    }
    result = replay(manifest, manifest["chosen"])
    assert result["validator_pass"] is True
    assert result["validator_reason"] == ""
    assert result["components"]["validator_reward"] == 1.0
    assert result["error"] is None


def test_metadata_errors_detect_missing_required_fields() -> None:
    row = {"id": "x", "category": "reasoning", "metadata": {}}
    assert metadata_errors(row) == ["metadata_missing_or_empty"]
    row["metadata"] = {"answer": 3}
    assert metadata_errors(row) == []


def test_failure_classification_separates_structure_and_value() -> None:
    assert classify_failure("termination_constraint") == "structural"
    assert classify_failure("count_or_duplicate_mismatch") == "structural"
    assert classify_failure("arithmetic_value") == "semantic_value"
    assert classify_failure("format_value_or_order_mismatch") == "semantic_value"
    assert classify_failure("") == "none"


def test_component_routing_matches_category_contract() -> None:
    assert component_contract_mismatches("reasoning", True, {"arithmetic_reward": 1.0, "format_reward": 1.0, "parse_reward": 0.0, "field_reward": 0.0, "item_count_reward": 0.0}) == []
    assert "arithmetic_reward" in component_contract_mismatches(
        "reasoning",
        True,
        {"arithmetic_reward": 0.0, "format_reward": 1.0, "parse_reward": 0.0, "field_reward": 0.0, "item_count_reward": 0.0},
    )


def test_response_features_preserve_generation_telemetry() -> None:
    features = response_features(
        {"generated_tokens": 12, "termination_reason": "eos", "eos_seen": True, "finished_naturally": True, "max_length_hit": False},
        "第一行\n第二行",
    )
    assert features["has_newline"] is True
    assert features["finished_naturally"] is True
    assert features["generated_tokens"] == 12


def test_audit_refuses_non_empty_output_directory(tmp_path) -> None:
    output = tmp_path / "audit"
    output.mkdir()
    (output / "existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ensure_empty_output_dir(output)

from pathlib import Path

from evaluation.audit_public_release import audit


ROOT = Path(__file__).resolve().parents[1]


def test_public_release_audit():
    result = audit(ROOT)
    assert result["status"] == "PUBLIC_AUDIT_PASS"
    assert result["default_model_changed"] is False


def test_public_tree_excludes_weights_and_runtime_state():
    forbidden_names = {".venv", ".venv-teacher", "checkpoints", "out"}
    forbidden_suffixes = {".pth", ".safetensors"}
    for path in ROOT.rglob("*"):
        assert path.name not in forbidden_names
        if path.is_file():
            assert path.suffix not in forbidden_suffixes


def test_public_artifact_policy_paths_are_explicit():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (
        "results/experiments/",
        "results/inputs/",
        "dataset/alignment_v2/generated/",
        "dataset/alignment_v2/manifests/",
    ):
        assert pattern in ignore

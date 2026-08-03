import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts/run_release_inference.py"
    spec = importlib.util.spec_from_file_location("run_release_inference", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_release_inference_parser_has_explicit_weight_and_greedy_default():
    module = _load_module()
    args = module.build_parser().parse_args(["--weight", "weights/model.pth", "--prompt", "hello"])
    assert args.weight == Path("weights/model.pth")
    assert args.prompt == "hello"
    assert args.do_sample is False
    assert args.max_new_tokens == 128

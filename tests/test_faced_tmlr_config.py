from pathlib import Path

import pytest

from tmlr.config import build_config, parse_config


def test_config_file_and_cli_override_round_trip():
    root = Path(__file__).resolve().parents[1]
    config = build_config(str(root / "configs/faced_tmlr.yaml"), {"batch_size": 7, "method": "frozen_probe"})
    assert config.batch_size == 7
    assert config.method == "frozen_probe"
    assert config.selection_metric == "cohen_kappa"

    parsed = parse_config(["--config", str(root / "configs/faced_tmlr.yaml"), "--batch-size", "9"])
    assert parsed.batch_size == 9


def test_unsupported_method_fails_without_fallback():
    with pytest.raises(NotImplementedError):
        build_config(None, {"method": "lora"})
    with pytest.raises(ValueError):
        build_config(None, {"method": "interaction_aligned", "adapter_type": None})

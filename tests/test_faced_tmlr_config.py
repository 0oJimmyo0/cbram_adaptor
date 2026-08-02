from pathlib import Path

import pytest

from tmlr.config import build_config, parse_config


def test_config_file_and_cli_override_round_trip():
    root = Path(__file__).resolve().parents[1]
    config = build_config(str(root / "configs/faced_tmlr.yaml"), {
        "batch_size": 7,
        "method": "frozen_probe",
        "optimizer_contract": "explicit",
        "scheduler": "none",
        "loader_contract": "explicit_seeded",
        "classifier": "avgpooling_patch_reps",
        "head_seed": 10042,
    })
    assert config.batch_size == 7
    assert config.method == "frozen_probe"
    assert config.selection_metric == "cohen_kappa"

    parsed = parse_config([
        "--config", str(root / "configs/faced_tmlr.yaml"), "--batch-size", "9",
        "--optimizer-contract", "original_cbramod",
    ])
    assert parsed.batch_size == 9


def test_new_controls_are_supported_and_unknown_methods_fail():
    root = Path(__file__).resolve().parents[1]
    for method in ("lora", "generic_bottleneck", "upper_k_finetune", "axis_blind", "native_full_finetune"):
        config = build_config(str(root / "configs/faced_tmlr_locked.yaml"), {"method": method})
        assert config.method == method
    with pytest.raises(ValueError):
        build_config(None, {"method": "unknown_method"})
    with pytest.raises(ValueError):
        build_config(None, {"method": "interaction_aligned", "adapter_type": None})
    with pytest.raises(ValueError):
        build_config(None, {"method": "native_full_finetune", "adapter_type": None})

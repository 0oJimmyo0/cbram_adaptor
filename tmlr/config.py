"""Resolved configuration for the isolated CBraMod FACED TMLR runner."""

from __future__ import annotations

import argparse
import ast
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised on the cluster image
    yaml = None


SUPPORTED_METHODS = {
    "full_finetune", "frozen_probe", "interaction_aligned",
    "native_full_finetune", "generic_bottleneck", "lora", "upper_k_finetune", "axis_blind",
}
RESERVED_METHODS = set()
ADAPTER_TYPES = {"channel", "patch", "channel_patch"}
SELECTION_METRICS = {"cohen_kappa", "balanced_accuracy", "macro_f1"}
OPTIMIZER_CONTRACTS = {"explicit", "locked_cbramod", "original_cbramod"}
SCHEDULERS = {"none", "cosine_per_iteration"}
LOADER_CONTRACTS = {"explicit_seeded", "original_cbramod"}

DEFAULTS: Dict[str, Any] = {
    "dataset_name": "FACED",
    "dataset_path": "/data/neurogroup/mingyangjiang/data/FACED",
    "channel_manifest": "configs/faced_channel_manifest.json",
    "checkpoint": "/data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth",
    "output_root": "results/faced",
    "method": "full_finetune",
    "adapter_type": None,
    "adapter_bottleneck": 64,
    "adapter_heads": 4,
    "adapter_dropout": 0.0,
    "adapter_init_alpha": 0.01,
    "adapter_gamma": 1.0,
    "adapter_zero_init_output": True,
    "allow_singleton_patch_control": False,
    "seed": 42,
    "head_seed": 10042,
    "loader_seed": 20042,
    "adapter_seed": 30042,
    "generic_bottleneck": 64,
    "axis_blind_bottleneck": 148,
    "upper_k": 2,
    "lora_rank": 8,
    "lora_alpha": 16.0,
    "batch_size": 16,
    "epochs": 1,
    "max_train_batches": None,
    "max_val_batches": None,
    "max_test_batches": None,
    "num_workers": 0,
    "lr": 1e-4,
    "head_lr": 1e-3,
    "adapter_lr": 1e-3,
    "weight_decay": 0.05,
    "head_weight_decay": 0.05,
    "adapter_weight_decay": 0.05,
    "lora_lr": 0.0005,
    "upper_lr": 0.0001,
    "optimizer_contract": "explicit",
    "scheduler": "none",
    "scheduler_eta_min": 1e-6,
    "loader_contract": "explicit_seeded",
    "label_smoothing": 0.1,
    "clip_grad": 1.0,
    "selection_metric": "cohen_kappa",
    "classifier": "all_patch_reps",
    "dropout": 0.1,
    "input_scale_divisor": 100.0,
    "num_classes": 9,
    "device": "auto",
    "run_id": None,
    "save_test": True,
    "audit_only": False,
    "smoke": False,
    "resume_from": None,
    "overwrite": False,
}


@dataclass
class FacedTMLRConfig:
    values: Dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.values)


def _load_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path_obj}")
    with path_obj.open("r", encoding="utf-8") as handle:
        if path_obj.suffix.lower() == ".json":
            payload = json.load(handle)
        elif yaml is not None:
            payload = yaml.safe_load(handle)
        else:
            payload = _simple_yaml_mapping(handle.read())
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Configuration file must contain a mapping")
    return payload


def _simple_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _simple_yaml_mapping(text: str) -> Dict[str, Any]:
    """Parse the flat key/value YAML used by the isolated run configs.

    The cluster training image does not provide PyYAML.  Keeping this fallback
    deliberately limited prevents an undeclared parser dependency while still
    rejecting nested config structures that this runner does not support.
    """
    result: Dict[str, Any] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line or line.startswith(("-", " ", "\t")):
            raise ValueError(f"Unsupported flat YAML syntax at line {line_number}: {raw_line!r}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty YAML key at line {line_number}")
        result[key] = _simple_yaml_scalar(raw_value)
    return result


def validate_config(values: Dict[str, Any]) -> Dict[str, Any]:
    resolved = copy.deepcopy(values)
    dataset_name = str(resolved.get("dataset_name", "FACED")).strip().upper().replace("_", "-")
    if dataset_name not in {"FACED", "SEED-V"}:
        raise ValueError(f"Unsupported CBraMod TMLR dataset: {dataset_name!r}")
    resolved["dataset_name"] = dataset_name
    method = str(resolved["method"]).strip().lower()
    resolved["method"] = method
    if method in RESERVED_METHODS:
        raise NotImplementedError(
            f"Method {method!r} is reserved for a later TMLR gate and is not implemented."
        )
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method {method!r}; supported now: {sorted(SUPPORTED_METHODS)}")

    adapter_type = resolved.get("adapter_type")
    if method in {"interaction_aligned", "native_full_finetune"}:
        if adapter_type not in ADAPTER_TYPES:
            raise ValueError(
                "interaction_aligned requires adapter_type in "
                f"{sorted(ADAPTER_TYPES)}; it cannot silently fall back to dense."
            )
    elif adapter_type is not None:
        raise ValueError(f"method={method} must not request adapter_type={adapter_type!r}")
    if resolved["selection_metric"] not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {sorted(SELECTION_METRICS)}")
    if resolved["classifier"] not in {"avgpooling_patch_reps", "all_patch_reps"}:
        raise ValueError(
            "The isolated TMLR runner supports only avgpooling_patch_reps "
            "and all_patch_reps classifiers."
        )
    if resolved["optimizer_contract"] not in OPTIMIZER_CONTRACTS:
        raise ValueError(f"optimizer_contract must be one of {sorted(OPTIMIZER_CONTRACTS)}")
    if resolved["scheduler"] not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {sorted(SCHEDULERS)}")
    if resolved["loader_contract"] not in LOADER_CONTRACTS:
        raise ValueError(f"loader_contract must be one of {sorted(LOADER_CONTRACTS)}")
    if resolved["optimizer_contract"] in {"locked_cbramod", "original_cbramod"}:
        if resolved["classifier"] != "all_patch_reps":
            raise ValueError(f"{resolved['optimizer_contract']} requires classifier=all_patch_reps")
        if resolved["scheduler"] != "cosine_per_iteration":
            raise ValueError(f"{resolved['optimizer_contract']} requires scheduler=cosine_per_iteration")
        if resolved["loader_contract"] != "original_cbramod":
            raise ValueError(f"{resolved['optimizer_contract']} requires loader_contract=original_cbramod")
    if resolved["optimizer_contract"] == "original_cbramod":
        if method != "full_finetune":
            raise ValueError("original_cbramod optimizer contract is only valid for full_finetune")
    expected_classes = 9 if dataset_name == "FACED" else 5
    if int(resolved["num_classes"]) != expected_classes:
        raise ValueError(f"{dataset_name} TMLR requires num_classes={expected_classes}")
    if float(resolved["input_scale_divisor"]) != 100.0:
        raise ValueError("FACED TMLR requires input_scale_divisor=100")
    if int(resolved["batch_size"]) <= 0 or int(resolved["epochs"]) <= 0:
        raise ValueError("batch_size and epochs must be positive")
    if dataset_name == "FACED" and int(resolved["upper_k"]) != 2:
        raise ValueError("The TMLR FACED upper-block control is fixed to upper_k=2")
    if int(resolved["num_workers"]) < 0:
        raise ValueError("num_workers cannot be negative")
    for field in ("max_train_batches", "max_val_batches", "max_test_batches"):
        if resolved.get(field) is not None and int(resolved[field]) <= 0:
            raise ValueError(f"{field} must be positive when provided")
    return resolved


def build_config(config_path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> FacedTMLRConfig:
    resolved = copy.deepcopy(DEFAULTS)
    resolved.update(_load_file(config_path))
    if overrides:
        resolved.update({key: value for key, value in overrides.items() if value is not None})
    return FacedTMLRConfig(validate_config(resolved))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CBraMod TMLR dataset runner")
    parser.add_argument("--config", default=None)
    for name, kwargs in (
        ("dataset_name", {"type": str}), ("dataset_path", {"type": str}), ("channel_manifest", {"type": str}),
        ("checkpoint", {"type": str}), ("output_root", {"type": str}),
        ("method", {"type": str}), ("adapter_type", {"type": str}),
        ("classifier", {"type": str}), ("selection_metric", {"type": str}),
        ("optimizer_contract", {"type": str}), ("scheduler", {"type": str}),
        ("loader_contract", {"type": str}), ("device", {"type": str}),
        ("run_id", {"type": str}),
        ("resume_from", {"type": str}),
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None, **kwargs)
    for name in (
        "adapter_bottleneck", "adapter_heads", "generic_bottleneck", "axis_blind_bottleneck",
        "upper_k", "lora_rank", "seed", "head_seed", "loader_seed",
        "adapter_seed", "batch_size", "epochs", "max_train_batches", "max_val_batches",
        "max_test_batches", "num_workers", "num_classes",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None, type=int)
    for name in (
        "adapter_dropout", "adapter_init_alpha", "adapter_gamma", "lr", "head_lr",
        "adapter_lr", "weight_decay", "head_weight_decay", "adapter_weight_decay",
        "lora_lr", "upper_lr", "lora_alpha", "scheduler_eta_min", "dropout",
        "input_scale_divisor", "label_smoothing", "clip_grad",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None, type=float)
    for name in ("adapter_zero_init_output", "save_test", "overwrite"):
        parser.add_argument(
            f"--{name.replace('_', '-')}", dest=name, default=None,
            action=argparse.BooleanOptionalAction,
        )
    parser.add_argument(
        "--allow-singleton-patch-control", dest="allow_singleton_patch_control",
        default=None, action=argparse.BooleanOptionalAction,
        help="Explicitly permit the singleton SEED-V patch-capacity control.",
    )
    parser.add_argument("--audit-only", action="store_true", default=None)
    parser.add_argument("--smoke", action="store_true", default=None)
    return parser


def parse_config(argv: Optional[Iterable[str]] = None) -> FacedTMLRConfig:
    args = build_parser().parse_args(argv)
    overrides = vars(args).copy()
    config_path = overrides.pop("config")
    return build_config(config_path, overrides)

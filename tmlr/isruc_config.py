"""Resolved configuration for the isolated CBraMod ISRUC TMLR runner."""

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
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


SUPPORTED_METHODS = {
    "full_finetune", "frozen_probe", "interaction_aligned", "native_full_finetune",
    "generic_bottleneck", "lora", "upper_k_finetune", "axis_blind",
}
ADAPTER_TYPES = {"channel", "patch", "channel_patch"}
SELECTION_METRICS = {"cohen_kappa", "balanced_accuracy", "macro_f1"}

DEFAULTS: Dict[str, Any] = {
    "dataset_name": "ISRUC",
    "dataset_path": "/data/neurogroup/mingyangjiang/data/ISRUC",
    "channel_manifest": "configs/isruc_channel_manifest.json",
    "checkpoint": "/data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth",
    "output_root": "results/isruc",
    "method": "full_finetune",
    "adapter_type": None,
    "adapter_bottleneck": 64,
    "adapter_heads": 4,
    "adapter_dropout": 0.0,
    "adapter_init_alpha": 0.01,
    "adapter_gamma": 1.0,
    "adapter_zero_init_output": True,
    "seed": 42,
    "head_seed": 10042,
    "loader_seed": 20042,
    "adapter_seed": 30042,
    "generic_bottleneck": 64,
    "axis_blind_bottleneck": 148,
    "upper_k": 2,
    "lora_rank": 8,
    "lora_alpha": 16.0,
    "batch_size": 8,
    "epochs": 20,
    "max_train_batches": None,
    "max_val_batches": None,
    "max_test_batches": None,
    "num_workers": 0,
    "lr": 2e-4,
    "head_lr": 3.536e-4,
    "adapter_lr": 2e-5,
    "weight_decay": 0.05,
    "head_weight_decay": 0.05,
    "adapter_weight_decay": 0.05,
    "adapter_alpha_weight_decay": 0.05,
    "lora_lr": 2e-5,
    "upper_lr": 2e-4,
    "scheduler_eta_min": 1e-6,
    "label_smoothing": 0.1,
    "clip_grad": 1.0,
    "selection_metric": "cohen_kappa",
    "dropout": 0.1,
    "input_scale_divisor": 1.0,
    "num_classes": 5,
    "device": "auto",
    "run_id": None,
    "save_test": True,
    "audit_only": False,
    "smoke": False,
    "overwrite": False,
}


@dataclass
class IsrucTMLRConfig:
    values: Dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.values)


def _simple_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


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
            payload = {}
            for line_number, raw_line in enumerate(handle.read().splitlines(), start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    raise ValueError(f"Unsupported YAML at line {line_number}: {raw_line!r}")
                key, raw_value = line.split(":", 1)
                payload[key.strip()] = _simple_scalar(raw_value)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Configuration file must contain a mapping")
    return payload


def validate_config(values: Dict[str, Any]) -> Dict[str, Any]:
    resolved = copy.deepcopy(values)
    if str(resolved.get("dataset_name", "ISRUC")).strip().upper() != "ISRUC":
        raise ValueError("The isolated runner only accepts dataset_name=ISRUC")
    resolved["dataset_name"] = "ISRUC"
    method = str(resolved["method"]).strip().lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown ISRUC method {method!r}; supported: {sorted(SUPPORTED_METHODS)}")
    resolved["method"] = method
    if method in {"interaction_aligned", "native_full_finetune"}:
        if resolved.get("adapter_type") not in ADAPTER_TYPES:
            raise ValueError(f"{method} requires adapter_type in {sorted(ADAPTER_TYPES)}")
    elif resolved.get("adapter_type") is not None:
        raise ValueError(f"method={method} must not request adapter_type")
    if resolved["selection_metric"] not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {sorted(SELECTION_METRICS)}")
    if int(resolved["num_classes"]) != 5:
        raise ValueError("ISRUC requires num_classes=5")
    if float(resolved["input_scale_divisor"]) != 1.0:
        raise ValueError("ISRUC requires input_scale_divisor=1.0")
    if int(resolved["batch_size"]) <= 0 or int(resolved["epochs"]) <= 0:
        raise ValueError("batch_size and epochs must be positive")
    if int(resolved["num_workers"]) < 0:
        raise ValueError("num_workers cannot be negative")
    for field in ("max_train_batches", "max_val_batches", "max_test_batches"):
        if resolved.get(field) is not None and int(resolved[field]) <= 0:
            raise ValueError(f"{field} must be positive when provided")
    return resolved


def build_config(config_path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> IsrucTMLRConfig:
    resolved = copy.deepcopy(DEFAULTS)
    resolved.update(_load_file(config_path))
    if overrides:
        resolved.update({key: value for key, value in overrides.items() if value is not None})
    return IsrucTMLRConfig(validate_config(resolved))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CBraMod ISRUC TMLR runner")
    parser.add_argument("--config", default=None)
    for name in ("dataset_path", "channel_manifest", "checkpoint", "output_root", "method", "adapter_type", "device", "run_id"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None)
    for name in ("adapter_bottleneck", "adapter_heads", "generic_bottleneck", "axis_blind_bottleneck", "upper_k", "lora_rank", "seed", "head_seed", "loader_seed", "adapter_seed", "batch_size", "epochs", "max_train_batches", "max_val_batches", "max_test_batches", "num_workers"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None, type=int)
    for name in ("adapter_dropout", "adapter_init_alpha", "adapter_gamma", "lr", "head_lr", "adapter_lr", "weight_decay", "head_weight_decay", "adapter_weight_decay", "adapter_alpha_weight_decay", "lora_lr", "upper_lr", "lora_alpha", "scheduler_eta_min", "dropout", "input_scale_divisor", "label_smoothing", "clip_grad"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None, type=float)
    for name in ("adapter_zero_init_output", "save_test", "overwrite"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--audit-only", default=None, action="store_true")
    parser.add_argument("--smoke", default=None, action="store_true")
    return parser


def parse_config(argv: Optional[Iterable[str]] = None) -> IsrucTMLRConfig:
    args = build_parser().parse_args(argv)
    overrides = vars(args).copy()
    config_path = overrides.pop("config")
    return build_config(config_path, overrides)

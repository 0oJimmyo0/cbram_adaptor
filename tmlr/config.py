"""Resolved configuration for the isolated CBraMod FACED TMLR runner."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


SUPPORTED_METHODS = {"full_finetune", "frozen_probe", "interaction_aligned"}
RESERVED_METHODS = {"upper_k_finetune", "lora", "generic_bottleneck", "axis_blind"}
ADAPTER_TYPES = {"channel", "patch", "channel_patch"}
SELECTION_METRICS = {"cohen_kappa", "balanced_accuracy", "macro_f1"}

DEFAULTS: Dict[str, Any] = {
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
    "seed": 42,
    "head_seed": 10042,
    "loader_seed": 20042,
    "adapter_seed": 30042,
    "batch_size": 16,
    "epochs": 1,
    "max_train_batches": None,
    "max_val_batches": None,
    "num_workers": 0,
    "lr": 1e-4,
    "head_lr": 1e-3,
    "adapter_lr": 1e-3,
    "weight_decay": 0.05,
    "head_weight_decay": 0.05,
    "adapter_weight_decay": 0.05,
    "label_smoothing": 0.1,
    "clip_grad": 1.0,
    "selection_metric": "cohen_kappa",
    "classifier": "avgpooling_patch_reps",
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
        payload = json.load(handle) if path_obj.suffix.lower() == ".json" else yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Configuration file must contain a mapping")
    return payload


def validate_config(values: Dict[str, Any]) -> Dict[str, Any]:
    resolved = copy.deepcopy(values)
    method = str(resolved["method"]).strip().lower()
    resolved["method"] = method
    if method in RESERVED_METHODS:
        raise NotImplementedError(
            f"Method {method!r} is reserved for a later TMLR gate and is not implemented."
        )
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method {method!r}; supported now: {sorted(SUPPORTED_METHODS)}")

    adapter_type = resolved.get("adapter_type")
    if method == "interaction_aligned":
        if adapter_type not in ADAPTER_TYPES:
            raise ValueError(
                "interaction_aligned requires adapter_type in "
                f"{sorted(ADAPTER_TYPES)}; it cannot silently fall back to dense."
            )
    elif adapter_type is not None:
        raise ValueError(f"method={method} must not request adapter_type={adapter_type!r}")
    if resolved["selection_metric"] not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {sorted(SELECTION_METRICS)}")
    if resolved["classifier"] != "avgpooling_patch_reps":
        raise ValueError(
            "The isolated TMLR runner currently supports only the explicit "
            "avgpooling_patch_reps classifier."
        )
    if int(resolved["num_classes"]) != 9:
        raise ValueError("FACED TMLR requires num_classes=9")
    if float(resolved["input_scale_divisor"]) != 100.0:
        raise ValueError("FACED TMLR requires input_scale_divisor=100")
    if int(resolved["batch_size"]) <= 0 or int(resolved["epochs"]) <= 0:
        raise ValueError("batch_size and epochs must be positive")
    if int(resolved["num_workers"]) < 0:
        raise ValueError("num_workers cannot be negative")
    for field in ("max_train_batches", "max_val_batches"):
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
    parser = argparse.ArgumentParser(description="CBraMod TMLR FACED runner")
    parser.add_argument("--config", default=None)
    for name, kwargs in (
        ("dataset_path", {"type": str}), ("channel_manifest", {"type": str}),
        ("checkpoint", {"type": str}), ("output_root", {"type": str}),
        ("method", {"type": str}), ("adapter_type", {"type": str}),
        ("classifier", {"type": str}), ("selection_metric", {"type": str}),
        ("device", {"type": str}), ("run_id", {"type": str}),
        ("resume_from", {"type": str}),
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None, **kwargs)
    for name in (
        "adapter_bottleneck", "adapter_heads", "seed", "head_seed", "loader_seed",
        "adapter_seed", "batch_size", "epochs", "max_train_batches", "max_val_batches",
        "num_workers", "num_classes",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None, type=int)
    for name in (
        "adapter_dropout", "adapter_init_alpha", "adapter_gamma", "lr", "head_lr",
        "adapter_lr", "weight_decay", "head_weight_decay", "adapter_weight_decay",
        "dropout", "input_scale_divisor", "label_smoothing", "clip_grad",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None, type=float)
    for name in ("adapter_zero_init_output", "save_test", "overwrite"):
        parser.add_argument(
            f"--{name.replace('_', '-')}", dest=name, default=None,
            action=argparse.BooleanOptionalAction,
        )
    parser.add_argument("--audit-only", action="store_true", default=None)
    parser.add_argument("--smoke", action="store_true", default=None)
    return parser


def parse_config(argv: Optional[Iterable[str]] = None) -> FacedTMLRConfig:
    args = build_parser().parse_args(argv)
    overrides = vars(args).copy()
    config_path = overrides.pop("config")
    return build_config(config_path, overrides)

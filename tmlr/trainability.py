"""Explicit method/trainability registry for CBraMod FACED TMLR runs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch

from .config import ADAPTER_TYPES, RESERVED_METHODS, SUPPORTED_METHODS


def _component(name: str) -> str:
    if name.startswith("backbone.native_axis_adapter."):
        return "adapter"
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("classifier."):
        return "classifier"
    return "other"


def _count(parameters: Iterable[torch.nn.Parameter]) -> int:
    return int(sum(parameter.numel() for parameter in parameters))


def apply_trainability_contract(
    model: torch.nn.Module,
    method: str,
    adapter_type: str | None = None,
    lr: float = 1e-4,
    head_lr: float = 1e-3,
    adapter_lr: float = 1e-3,
    weight_decay: float = 0.05,
    head_weight_decay: float = 0.05,
    adapter_weight_decay: float = 0.05,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    method = str(method).strip().lower()
    if method in RESERVED_METHODS:
        raise NotImplementedError(f"Method {method!r} is reserved and not implemented")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method {method!r}")

    named = list(model.named_parameters())
    if not named:
        raise AssertionError("Model has zero parameters")
    for _, parameter in named:
        parameter.requires_grad_(False)

    adapter_exists = getattr(getattr(model, "backbone", None), "native_axis_adapter", None) is not None
    expected_adapter = method == "interaction_aligned"
    if expected_adapter != adapter_exists:
        raise AssertionError(
            f"Requested {method} but adapter existence is {adapter_exists}; "
            "the method must not silently fall back."
        )
    if expected_adapter and adapter_type not in ADAPTER_TYPES:
        raise ValueError(f"interaction_aligned requires adapter_type in {sorted(ADAPTER_TYPES)}")

    trainable_names: List[str] = []
    frozen_names: List[str] = []
    component_names: Dict[str, List[str]] = {key: [] for key in ("backbone", "adapter", "classifier", "other")}
    for name, parameter in named:
        component = _component(name)
        component_names[component].append(name)
        if method == "full_finetune":
            should_train = component in {"backbone", "classifier"}
        elif method == "frozen_probe":
            should_train = component == "classifier"
        else:
            should_train = component in {"adapter", "classifier"}
        parameter.requires_grad_(should_train)
        (trainable_names if should_train else frozen_names).append(name)

    if not component_names["backbone"] or not component_names["classifier"]:
        raise AssertionError("Expected nonempty backbone and classifier components")
    if expected_adapter and not component_names["adapter"]:
        raise AssertionError("Expected nonempty native adapter component")
    if not expected_adapter and component_names["adapter"]:
        raise AssertionError("Adapter parameters exist for a method that requires adapter absence")
    if component_names["other"]:
        raise AssertionError(f"Unexpected unclassified parameters: {component_names['other'][:5]}")

    group_specs = [
        ("backbone", "backbone", float(lr), float(weight_decay)),
        ("adapter", "adapter", float(adapter_lr), float(adapter_weight_decay)),
        ("classifier", "classifier", float(head_lr), float(head_weight_decay)),
    ]
    groups: List[Dict[str, Any]] = []
    optimizer_group_report = []
    for group_name, component, group_lr, group_decay in group_specs:
        group_names = [
            name for name in component_names[component]
            if dict(named)[name].requires_grad
        ]
        if group_names:
            groups.append({
                "name": group_name,
                "params": [dict(named)[name] for name in group_names],
                "lr": group_lr,
                "weight_decay": group_decay,
            })
            optimizer_group_report.append({
                "name": group_name,
                "parameter_names": group_names,
                "parameter_count": _count(dict(named)[name] for name in group_names),
                "learning_rate": group_lr,
                "weight_decay": group_decay,
            })
    if not groups:
        raise AssertionError("Trainability contract produced no optimizer parameters")

    total = _count(parameter for _, parameter in named)
    trainable = _count(parameter for _, parameter in named if parameter.requires_grad)
    report = {
        "method": method,
        "adapter_type": adapter_type,
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "trainable_percentage": 100.0 * trainable / total,
        "component_parameter_counts": {
            component: _count(dict(named)[name] for name in names)
            for component, names in component_names.items()
        },
        "component_trainable_parameter_counts": {
            component: _count(
                dict(named)[name] for name in names if dict(named)[name].requires_grad
            )
            for component, names in component_names.items()
        },
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "optimizer_groups": optimizer_group_report,
    }
    _assert_contract(model, report)
    return groups, report


def _assert_contract(model: torch.nn.Module, report: Dict[str, Any]) -> None:
    method = report["method"]
    trainable = set(report["trainable_parameter_names"])
    backbone_names = {name for name, _ in model.named_parameters() if name.startswith("backbone.")}
    adapter_names = {name for name, _ in model.named_parameters() if name.startswith("backbone.native_axis_adapter.")}
    classifier_names = {name for name, _ in model.named_parameters() if name.startswith("classifier.")}
    if method == "full_finetune":
        assert not adapter_names
        assert backbone_names <= trainable
        assert classifier_names <= trainable
    elif method == "frozen_probe":
        assert not adapter_names
        assert not (backbone_names & trainable)
        assert classifier_names <= trainable
    else:
        assert adapter_names
        assert adapter_names <= trainable
        assert classifier_names <= trainable
        assert not ((backbone_names - adapter_names) & trainable)

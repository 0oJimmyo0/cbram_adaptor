"""Explicit method/trainability registry for CBraMod FACED TMLR runs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch

from .config import ADAPTER_TYPES, RESERVED_METHODS, SUPPORTED_METHODS


FROZEN_BASE_METHODS = {
    "frozen_probe",
    "interaction_aligned",
    "generic_bottleneck",
    "lora",
    "upper_k_finetune",
    "axis_blind",
}


def _component(name: str) -> str:
    if ".lora_A" in name or ".lora_B" in name:
        return "lora"
    if name.startswith("backbone.native_axis_adapter.") or name.startswith("backbone.generic_adapter."):
        return "adapter"
    if name.startswith("backbone.encoder.layers."):
        parts = name.split(".")
        if len(parts) > 3 and parts[3].isdigit() and int(parts[3]) >= 10:
            return "upper"
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
    lora_lr: float = 5e-4,
    upper_lr: float = 1e-4,
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

    native_exists = getattr(getattr(model, "backbone", None), "native_axis_adapter", None) is not None
    generic_exists = getattr(getattr(model, "backbone", None), "generic_adapter", None) is not None
    expected_adapter = method in {
        "interaction_aligned", "native_full_finetune", "generic_bottleneck", "axis_blind",
    }
    if expected_adapter != (native_exists or generic_exists):
        raise AssertionError(
            f"Requested {method} but adaptation existence is {native_exists or generic_exists}; "
            "the method must not silently fall back."
        )
    if method in {"interaction_aligned", "native_full_finetune"} and adapter_type not in ADAPTER_TYPES:
        raise ValueError(f"{method} requires adapter_type in {sorted(ADAPTER_TYPES)}")

    trainable_names: List[str] = []
    frozen_names: List[str] = []
    component_names: Dict[str, List[str]] = {
        key: [] for key in ("backbone", "upper", "lora", "adapter", "classifier", "other")
    }
    for name, parameter in named:
        component = _component(name)
        component_names[component].append(name)
        if method == "full_finetune":
            should_train = component in {"backbone", "upper", "classifier"}
        elif method == "frozen_probe":
            should_train = component == "classifier"
        elif method in {"interaction_aligned", "generic_bottleneck", "axis_blind"}:
            should_train = component in {"adapter", "classifier"}
        elif method == "native_full_finetune":
            should_train = component in {"backbone", "upper", "adapter", "classifier"}
        elif method == "lora":
            should_train = component in {"lora", "classifier"}
        elif method == "upper_k_finetune":
            should_train = component in {"upper", "classifier"}
        else:
            raise ValueError(f"Unsupported trainability method {method!r}")
        parameter.requires_grad_(should_train)
        (trainable_names if should_train else frozen_names).append(name)

    if not component_names["backbone"] or not component_names["classifier"]:
        raise AssertionError("Expected nonempty backbone and classifier components")
    if expected_adapter and not component_names["adapter"]:
        raise AssertionError("Expected nonempty adapter component")
    if not expected_adapter and component_names["adapter"]:
        raise AssertionError("Adapter parameters exist for a method that requires no adapter")
    if method == "lora" and not component_names["lora"]:
        raise AssertionError("LoRA method did not expose LoRA parameters")
    if method == "upper_k_finetune" and not component_names["upper"]:
        raise AssertionError("upper_k_finetune did not expose upper-block parameters")
    if component_names["other"]:
        raise AssertionError(f"Unexpected unclassified parameters: {component_names['other'][:5]}")

    named_dict = dict(named)
    group_specs = [
        ("backbone", "backbone", float(lr), float(weight_decay)),
        ("upper", "upper", float(upper_lr), float(weight_decay)),
        ("lora", "lora", float(lora_lr), float(adapter_weight_decay)),
        ("adapter", "adapter", float(adapter_lr), float(adapter_weight_decay)),
        ("classifier", "classifier", float(head_lr), float(head_weight_decay)),
    ]
    groups: List[Dict[str, Any]] = []
    optimizer_group_report = []
    for group_name, component, group_lr, group_decay in group_specs:
        group_names = [name for name in component_names[component] if named_dict[name].requires_grad]
        if group_names:
            groups.append({
                "name": group_name,
                "params": [named_dict[name] for name in group_names],
                "lr": group_lr,
                "weight_decay": group_decay,
            })
            optimizer_group_report.append({
                "name": group_name,
                "parameter_names": group_names,
                "parameter_count": _count(named_dict[name] for name in group_names),
                "learning_rate": group_lr,
                "weight_decay": group_decay,
            })
    if not groups:
        raise AssertionError("Trainability contract produced no optimizer parameters")

    report = {
        "method": method,
        "adapter_type": adapter_type,
        "total_parameter_count": _count(parameter for _, parameter in named),
        "trainable_parameter_count": _count(parameter for _, parameter in named if parameter.requires_grad),
        "trainable_percentage": 100.0 * _count(parameter for _, parameter in named if parameter.requires_grad) / _count(parameter for _, parameter in named),
        "component_parameter_counts": {
            component: _count(named_dict[name] for name in names)
            for component, names in component_names.items()
        },
        "component_trainable_parameter_counts": {
            component: _count(named_dict[name] for name in names if named_dict[name].requires_grad)
            for component, names in component_names.items()
        },
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "optimizer_groups": optimizer_group_report,
    }
    _assert_contract(model, report)
    return groups, report


def configure_training_modes(
    model: torch.nn.Module,
    method: str,
    upper_k: int = 2,
) -> Dict[str, Any]:
    """Set module modes consistently with the explicit trainability contract.

    ``model.train()`` alone is incorrect for PEFT methods: it also enables
    dropout inside the parameter-frozen backbone.  Frozen-base methods keep
    the base backbone in evaluation mode while explicitly training the
    attached adapter/LoRA/upper-block components and the classifier.
    """
    method = str(method).strip().lower()
    model.train()
    backbone = getattr(model, "backbone", None)
    classifier = getattr(model, "classifier", None)
    if backbone is None or classifier is None:
        raise AssertionError("Training-mode controller requires model.backbone and model.classifier")

    if method not in FROZEN_BASE_METHODS:
        if method not in {"full_finetune", "native_full_finetune"}:
            raise ValueError(f"Unsupported training-mode method {method!r}")
        return {
            "method": method,
            "frozen_backbone_eval_mode": False,
            "model_training": bool(model.training),
            "backbone_training": bool(backbone.training),
            "classifier_training": bool(classifier.training),
            "native_adapter_training": bool(
                getattr(backbone, "native_axis_adapter", None) is not None
                and backbone.native_axis_adapter.training
            ),
            "generic_adapter_training": bool(
                getattr(backbone, "generic_adapter", None) is not None
                and backbone.generic_adapter.training
            ),
            "lora_module_count": 0,
            "upper_trainable_layer_count": 0,
        }

    # Freeze behavior includes inference-mode dropout for the base feature
    # extractor.  Trainable children are explicitly restored below.
    backbone.eval()
    native_adapter = getattr(backbone, "native_axis_adapter", None)
    generic_adapter = getattr(backbone, "generic_adapter", None)
    lora_modules = []
    upper_layers = []

    if method == "interaction_aligned":
        if native_adapter is None:
            raise AssertionError("interaction_aligned requires a native adapter")
        native_adapter.train()
    elif method in {"generic_bottleneck", "axis_blind"}:
        if generic_adapter is None:
            raise AssertionError(f"{method} requires a generic adapter")
        generic_adapter.train()
    elif method == "lora":
        for module in backbone.modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                module.train()
                lora_modules.append(module)
    elif method == "upper_k_finetune":
        layers = getattr(getattr(backbone, "encoder", None), "layers", None)
        if layers is None or int(upper_k) <= 0 or int(upper_k) > len(layers):
            raise ValueError(f"upper_k={upper_k} is incompatible with the backbone")
        upper_layers = list(layers)[-int(upper_k):]
        for layer in upper_layers:
            layer.train()

    # ``backbone.eval()`` does not affect this sibling module, but make the
    # intended classifier mode explicit for the artifact and future changes.
    classifier.train()
    return {
        "method": method,
        "frozen_backbone_eval_mode": True,
        "model_training": bool(model.training),
        "backbone_training": bool(backbone.training),
        "classifier_training": bool(classifier.training),
        "native_adapter_training": bool(native_adapter is not None and native_adapter.training),
        "generic_adapter_training": bool(generic_adapter is not None and generic_adapter.training),
        "lora_module_count": len(lora_modules),
        "upper_trainable_layer_count": len(upper_layers),
    }


def _assert_contract(model: torch.nn.Module, report: Dict[str, Any]) -> None:
    method = report["method"]
    trainable = set(report["trainable_parameter_names"])
    all_names = {name for name, _ in model.named_parameters()}
    backbone_names = {name for name in all_names if name.startswith("backbone.")}
    base_backbone_names = {name for name in backbone_names if _component(name) == "backbone"}
    upper_names = {name for name in backbone_names if _component(name) == "upper"}
    lora_names = {name for name in backbone_names if _component(name) == "lora"}
    adapter_names = {
        name for name in backbone_names
        if name.startswith("backbone.native_axis_adapter.") or name.startswith("backbone.generic_adapter.")
    }
    classifier_names = {name for name in all_names if name.startswith("classifier.")}
    assert classifier_names <= trainable
    if method == "full_finetune":
        assert not adapter_names and not lora_names
        assert (base_backbone_names | upper_names) <= trainable
    elif method == "frozen_probe":
        assert not adapter_names and not lora_names
        assert not (backbone_names & trainable)
    elif method in {"interaction_aligned", "generic_bottleneck", "axis_blind"}:
        assert adapter_names <= trainable
        assert not (base_backbone_names & trainable)
        assert not upper_names & trainable
        assert not lora_names & trainable
    elif method == "native_full_finetune":
        assert adapter_names <= trainable
        assert (base_backbone_names | upper_names) <= trainable
        assert not lora_names & trainable
    elif method == "lora":
        assert lora_names <= trainable
        assert not (base_backbone_names & trainable)
        assert not upper_names & trainable
        assert not adapter_names & trainable
    elif method == "upper_k_finetune":
        assert upper_names <= trainable
        assert not (base_backbone_names & trainable)
        assert not lora_names & trainable
        assert not adapter_names & trainable

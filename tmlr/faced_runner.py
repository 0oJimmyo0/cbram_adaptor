"""Isolated, auditable CBraMod FACED TMLR runner."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader

from datasets.faced_dataset import CustomDataset
from models.cbramod import CBraMod
from models.lora import inject_qkv_lora, lora_diagnostics

from .artifact_writer import ArtifactWriter
from .config import FacedTMLRConfig
from .metrics import evaluate_model, selection_value
from .provenance import (
    audit_faced_dataset,
    collect_environment,
    sha256_file,
    structure_spec,
)
from .trainability import apply_trainability_contract


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    value = str(value).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    return device


def _original_head_learning_rate(batch_size: int) -> float:
    """Return the source CBraMod ``multi_lr`` classifier learning rate."""
    return 0.001 * (float(batch_size) / 256.0) ** 0.5


class FacedClassifier(nn.Module):
    """CBraMod encoder plus one explicit FACED classifier head."""

    def __init__(self, config: FacedTMLRConfig, checkpoint_path: Path, device: torch.device):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30, n_layer=12, nhead=8,
        )
        before_count = sum(parameter.numel() for parameter in self.backbone.parameters())
        checkpoint = _torch_load(checkpoint_path, device)
        state = _checkpoint_state(checkpoint)
        try:
            load_report = self.backbone.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Strict CBraMod checkpoint load failed for {checkpoint_path}: {exc}"
            ) from exc
        if load_report.missing_keys or load_report.unexpected_keys:
            raise RuntimeError(
                "Strict checkpoint load returned key mismatches: "
                f"missing={load_report.missing_keys} unexpected={load_report.unexpected_keys}"
            )
        checkpoint_report = {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "strict_loading_status": True,
            "missing_keys": list(load_report.missing_keys),
            "unexpected_keys": list(load_report.unexpected_keys),
            "model_parameter_count_before_attachment": int(before_count),
            "checkpoint_state_key_count": len(state),
        }

        # The original projection is part of the foundation checkpoint.  The
        # downstream head consumes encoder features, so remove it only after
        # the strict base checkpoint load has passed.
        self.backbone.proj_out = nn.Identity()
        checkpoint_report["model_parameter_count_after_projection_replacement"] = int(
            sum(parameter.numel() for parameter in self.backbone.parameters())
        )

        if config.method in {"interaction_aligned", "native_full_finetune"}:
            self.backbone.enable_interaction_adapter(
                adapter_type=config.adapter_type,
                bottleneck=config.adapter_bottleneck,
                num_heads=config.adapter_heads,
                dropout=config.adapter_dropout,
                init_alpha=config.adapter_init_alpha,
                gamma=config.adapter_gamma,
                zero_init_output=config.adapter_zero_init_output,
                seed=config.adapter_seed,
            )
        elif config.method == "generic_bottleneck":
            self.backbone.enable_generic_adapter(
                bottleneck=config.generic_bottleneck,
                init_alpha=config.adapter_init_alpha,
                gamma=config.adapter_gamma,
                zero_init_output=config.adapter_zero_init_output,
                seed=config.adapter_seed,
                adapter_type="generic_bottleneck",
            )
        elif config.method == "axis_blind":
            self.backbone.enable_generic_adapter(
                bottleneck=config.axis_blind_bottleneck,
                init_alpha=config.adapter_init_alpha,
                gamma=config.adapter_gamma,
                zero_init_output=config.adapter_zero_init_output,
                seed=config.adapter_seed,
                adapter_type="axis_blind",
            )
        elif config.method == "lora":
            inject_qkv_lora(self.backbone, rank=config.lora_rank, alpha=config.lora_alpha)
        elif config.method not in {"full_finetune", "frozen_probe", "upper_k_finetune"}:
            raise NotImplementedError(f"Method {config.method!r} is not implemented")

        # The source CBraMod FACED implementation uses the global seed set by
        # finetune_main.py.  A separate head seed remains available for the
        # explicit TMLR runner, but must be disabled for an exact source run.
        if config.head_seed is not None:
            torch.manual_seed(int(config.head_seed))
        if config.classifier == "avgpooling_patch_reps":
            self._uses_pooled_features = True
            self.classifier = nn.Linear(200, config.num_classes)
        elif config.classifier == "all_patch_reps":
            self._uses_pooled_features = False
            # Exact source model_for_faced.Model classifier.  Do not simplify
            # this head in the original_cbramod baseline contract.
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(32 * 10 * 200, 10 * 200),
                nn.ELU(),
                nn.Dropout(config.dropout),
                nn.Linear(10 * 200, 200),
                nn.ELU(),
                nn.Dropout(config.dropout),
                nn.Linear(200, config.num_classes),
            )
        else:
            raise ValueError(f"Unsupported FACED classifier: {config.classifier}")
        checkpoint_report["model_parameter_count_after_attachment"] = int(
            sum(parameter.numel() for parameter in self.parameters())
        )
        checkpoint_report["adapter_parameter_count"] = int(sum(
            parameter.numel() for name, parameter in self.named_parameters()
            if name.startswith("backbone.native_axis_adapter.")
            or name.startswith("backbone.generic_adapter.")
        ))
        checkpoint_report["lora_parameter_count"] = int(sum(
            parameter.numel() for name, parameter in self.named_parameters()
            if ".lora_A" in name or ".lora_B" in name
        ))
        self.checkpoint_report = checkpoint_report

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        if features.ndim != 4:
            raise RuntimeError(f"CBraMod encoder output must be [B,C,S,D], got {tuple(features.shape)}")
        if self._uses_pooled_features:
            features = features.mean(dim=(1, 2))
        return self.classifier(features)


def _torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _checkpoint_state(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Pretrained checkpoint must be a nonempty state-dict mapping")
    if not all(isinstance(key, str) for key in checkpoint):
        raise ValueError("Pretrained checkpoint keys must be strings")
    return checkpoint


def _resolve_path(repo_root: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else repo_root / candidate


def make_loaders(config: FacedTMLRConfig) -> Dict[str, DataLoader]:
    train_set = CustomDataset(config.dataset_path, mode="train")
    val_set = CustomDataset(config.dataset_path, mode="val")
    test_set = CustomDataset(config.dataset_path, mode="test")
    generator = torch.Generator()
    generator.manual_seed(int(config.loader_seed))

    def worker_init(worker_id: int) -> None:
        seed = int(config.loader_seed) + int(worker_id)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if config.loader_contract == "original_cbramod":
        # This intentionally mirrors datasets/faced_dataset.py in the source
        # repository: no explicit generator, worker initializer, or pin-memory
        # override is supplied.
        return {
            "train": DataLoader(train_set, batch_size=int(config.batch_size),
                                 collate_fn=train_set.collate, shuffle=True),
            "val": DataLoader(val_set, batch_size=int(config.batch_size),
                               collate_fn=val_set.collate, shuffle=False),
            "test": DataLoader(test_set, batch_size=int(config.batch_size),
                                collate_fn=test_set.collate, shuffle=False),
        }

    common = {
        "batch_size": int(config.batch_size),
        "num_workers": int(config.num_workers),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_init,
    }
    return {
        "train": DataLoader(train_set, shuffle=True, generator=generator, collate_fn=train_set.collate, **common),
        "val": DataLoader(val_set, shuffle=False, collate_fn=val_set.collate, **common),
        "test": DataLoader(test_set, shuffle=False, collate_fn=test_set.collate, **common),
    }


def capture_runtime_geometry(model: FacedClassifier, inputs: torch.Tensor) -> Dict[str, Any]:
    records: Dict[str, Any] = {"input_shape": list(inputs.shape)}
    hooks = [
        model.backbone.patch_embedding.register_forward_hook(
            lambda _module, _args, output: records.update({"patch_embedding_shape": list(output.shape)})
        ),
        model.backbone.encoder.register_forward_hook(
            lambda _module, _args, output: records.update({"encoder_output_shape": list(output.shape)})
        ),
    ]
    model.eval()
    with torch.no_grad():
        model(inputs)
    for hook in hooks:
        hook.remove()
    expected = [inputs.shape[0], 32, 10, 200]
    if list(inputs.shape[1:]) != expected[1:]:
        raise RuntimeError(f"FACED runtime input geometry mismatch: {list(inputs.shape)} != {expected}")
    if records.get("encoder_output_shape", [])[1:] != expected[1:]:
        raise RuntimeError(
            "FACED encoder geometry mismatch: "
            f"{records.get('encoder_output_shape')} != {expected}"
        )
    records.update({
        "channels": 32,
        "patches": 10,
        "embedding_dim": 200,
        "branch_dim": 100,
        "channel_eligibility": True,
        "patch_eligibility": True,
        "requested_adapter_type": model.backbone.adapter_type,
        "instantiated_branches": [
            name for name in ("channel_branch", "patch_branch")
            if model.backbone.native_axis_adapter is not None
            and hasattr(model.backbone.native_axis_adapter, name)
        ],
    })
    if model.backbone.native_axis_adapter is not None:
        if model.backbone.adapter_type == "channel" and "channel_branch" not in records["instantiated_branches"]:
            raise RuntimeError("Requested channel adapter branch was not instantiated")
        if model.backbone.adapter_type == "patch" and "patch_branch" not in records["instantiated_branches"]:
            raise RuntimeError("Requested patch adapter branch was not instantiated")
        if model.backbone.adapter_type == "channel_patch" and set(records["instantiated_branches"]) != {"channel_branch", "patch_branch"}:
            raise RuntimeError("Requested channel_patch adapter did not instantiate both branches")
    return records


def _snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def _norm(values: Iterable[torch.Tensor]) -> float:
    return float(sum(value.float().pow(2).sum() for value in values).sqrt())


def _parameter_component(name: str) -> str:
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


def _gradient_report(model: nn.Module) -> Dict[str, float]:
    components = {"backbone": [], "upper": [], "adapter": [], "lora": [], "classifier": []}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        component = _parameter_component(name)
        if component in components:
            components[component].append(parameter.grad.detach().cpu())
    return {
        f"{component}_gradient_norm": _norm(values) if values else 0.0
        for component, values in components.items()
    }


def _update_report(model: nn.Module, before: Dict[str, torch.Tensor]) -> Dict[str, float]:
    values = {"backbone": [], "upper": [], "adapter": [], "lora": [], "classifier": []}
    relative = {"backbone": [], "upper": [], "adapter": [], "lora": [], "classifier": []}
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        component = _parameter_component(name)
        if component not in values:
            continue
        difference = parameter.detach().cpu() - before[name]
        values[component].append(difference)
        relative[component].append(difference / before[name].abs().clamp_min(1e-12))
    return {
        **{f"{component}_update_norm": _norm(values[component]) if values[component] else 0.0 for component in values},
        **{f"{component}_relative_update_norm": _norm(relative[component]) if relative[component] else 0.0 for component in relative},
    }


def _write_runtime_summary(writer: ArtifactWriter, model: FacedClassifier, geometry: Dict[str, Any]) -> None:
    diagnostics = model.backbone.get_adapter_diagnostics()
    writer.write_json("structure_spec.json", {**structure_spec(), "runtime": geometry, "adapter_diagnostics": diagnostics})


def run(config: FacedTMLRConfig) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    config.values["dataset_path"] = str(Path(config.dataset_path).expanduser().resolve())
    config.values["channel_manifest"] = str(_resolve_path(repo_root, config.channel_manifest).resolve())
    config.values["checkpoint"] = str(_resolve_path(repo_root, config.checkpoint).resolve())
    config.values["output_root"] = str(_resolve_path(repo_root, config.output_root).resolve())
    dataset_audit = audit_faced_dataset(
        config.dataset_path, config.channel_manifest, config.input_scale_divisor
    )
    environment = collect_environment(repo_root, config.device)
    commit_short = environment["git_commit"][:8] or "nogit"
    run_id = config.run_id or (
        f"{config.method}_{config.adapter_type or 'none'}_s{config.seed}_"
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{commit_short}"
    )
    writer = ArtifactWriter(config.output_root, run_id, overwrite=config.overwrite)
    writer.write_json("resolved_config.json", config.to_dict())
    writer.write_json("provenance.json", environment)
    writer.write_json("environment.json", environment)
    writer.write_json("dataset_audit.json", dataset_audit)

    if config.audit_only:
        writer.write_json("structure_spec.json", structure_spec())
        writer.write_json("summary.json", {"status": "audit_only", "run_id": run_id})
        return {"status": "audit_only", "run_id": run_id, "artifact_dir": str(writer.run_dir)}
    checkpoint_path = Path(config.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Pretrained CBraMod checkpoint is required for training but was not found: {checkpoint_path}"
        )

    set_seed(config.seed)
    device = resolve_device(config.device)
    loaders = make_loaders(config)
    model = FacedClassifier(config, checkpoint_path, device).to(device)
    sample_inputs, _ = next(iter(loaders["val"]))
    geometry = capture_runtime_geometry(model, sample_inputs[: min(2, sample_inputs.shape[0])].to(device))
    writer.write_json("structure_spec.json", {**structure_spec(), "runtime": geometry})
    writer.write_json("checkpoint_load_report.json", model.checkpoint_report)

    head_lr = config.head_lr
    if config.optimizer_contract == "original_cbramod":
        head_lr = _original_head_learning_rate(config.batch_size)
    if head_lr is None:
        raise ValueError("head_lr must be set for the explicit optimizer contract")
    optimizer_groups, trainability = apply_trainability_contract(
        model, config.method, config.adapter_type,
        lr=config.lr, head_lr=head_lr, adapter_lr=config.adapter_lr,
        lora_lr=config.lora_lr, upper_lr=config.upper_lr,
        weight_decay=config.weight_decay,
        head_weight_decay=config.head_weight_decay,
        adapter_weight_decay=config.adapter_weight_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_groups)
    steps_per_epoch = len(loaders["train"])
    if config.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, int(config.max_train_batches))
    scheduler = None
    if config.scheduler == "cosine_per_iteration":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.epochs) * steps_per_epoch,
            eta_min=float(config.scheduler_eta_min),
        )
    writer.write_json("trainability_report.json", trainability)
    writer.write_json("optimizer_groups.json", {
        "contract": config.optimizer_contract,
        "scheduler": config.scheduler,
        "scheduler_eta_min": float(config.scheduler_eta_min),
        "groups": trainability["optimizer_groups"],
    })
    if config.resume_from:
        resume_path = Path(config.resume_from).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_state = _checkpoint_state(_torch_load(resume_path, device))
        resume_report = model.load_state_dict(resume_state, strict=True)
        if resume_report.missing_keys or resume_report.unexpected_keys:
            raise RuntimeError("Resume checkpoint strict-load mismatch")
        writer.write_json("resume_load_report.json", {
            "resume_path": str(resume_path), "strict_loading_status": True,
            "checkpoint_sha256": sha256_file(resume_path),
        })

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.label_smoothing)).to(device)
    best_score = float("-inf")
    best_epoch = 0
    best_validation_record: Optional[Dict[str, Any]] = None
    best_state = copy.deepcopy(model.state_dict())
    started = time.time()
    for epoch in range(1, int(config.epochs) + 1):
        epoch_started = time.time()
        model.train()
        before = _snapshot(model)
        losses = []
        last_gradient_report: Dict[str, float] = {}
        for batch_index, (inputs, labels) in enumerate(loaders["train"]):
            if config.max_train_batches is not None and batch_index >= int(config.max_train_batches):
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            last_gradient_report = _gradient_report(model)
            if float(config.clip_grad) > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(config.clip_grad),
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            raise RuntimeError("No training batches were processed")

        validation = evaluate_model(
            model, loaders["val"], criterion, device,
            num_classes=config.num_classes, max_batches=config.max_val_batches,
        )
        validation["epoch"] = epoch
        validation["train_loss"] = float(np.mean(losses))
        validation["elapsed_seconds"] = float(time.time() - epoch_started)
        validation["learning_rates"] = [float(group["lr"]) for group in optimizer.param_groups]
        writer.append_jsonl("metrics_by_epoch.jsonl", validation)
        adapter_diagnostics = model.backbone.get_adapter_diagnostics()
        if config.method == "lora":
            adapter_diagnostics.update(lora_diagnostics(model))
        adapter_diagnostics.update(last_gradient_report)
        adapter_diagnostics.update(_update_report(model, before))
        adapter_diagnostics["epoch"] = epoch
        writer.append_jsonl("adapter_diagnostics_by_epoch.jsonl", adapter_diagnostics)

        score = selection_value(validation, config.selection_metric)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_validation_record = copy.deepcopy(validation)
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state, strict=True)
    best_validation = {
        "selection_metric": config.selection_metric,
        "selection_value": best_score,
        "best_epoch": best_epoch,
        "metrics": best_validation_record,
    }
    writer.write_json("best_validation_metrics.json", best_validation)
    writer.save_model(model)
    test_metrics = None
    if config.save_test:
        test_metrics = evaluate_model(model, loaders["test"], criterion, device, num_classes=config.num_classes)
        test_metrics["selection_metric"] = config.selection_metric
        test_metrics["selected_epoch"] = best_epoch
        writer.write_json("test_metrics.json", test_metrics)
        writer.write_json("per_class_metrics.json", {
            "precision": test_metrics["per_class_precision"],
            "recall": test_metrics["per_class_recall"],
            "f1": test_metrics["per_class_f1"],
            "support": test_metrics["per_class_support"],
        })
        writer.write_json("confusion_matrix.json", {"confusion_matrix": test_metrics["confusion_matrix"]})
    epoch_records = []
    metrics_path = writer.run_dir / "metrics_by_epoch.jsonl"
    if metrics_path.exists():
        epoch_records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    total_samples = sum(len(loaders["train"].dataset) for _ in [0])
    time_per_epoch = [float(record["elapsed_seconds"]) for record in epoch_records]
    timing = {
        "training_wall_seconds": float(time.time() - started),
        "best_epoch": best_epoch,
        "time_per_epoch_seconds": time_per_epoch,
        "throughput_samples_per_second": float(
            total_samples * len(epoch_records) / max(sum(time_per_epoch), 1e-12)
        ),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    writer.write_json("timing.json", timing)
    writer.write_json("memory.json", {
        "peak_gpu_memory_bytes": timing["peak_gpu_memory_bytes"],
        "cuda_available": bool(torch.cuda.is_available()),
    })
    summary = {
        "status": "completed",
        "run_id": run_id,
        "artifact_dir": str(writer.run_dir),
        "method": config.method,
        "adapter_type": config.adapter_type,
        "seed": config.seed,
        "trainable_parameter_count": trainability["trainable_parameter_count"],
        "best_epoch": best_epoch,
        "best_validation_selection_value": best_score,
        "test_evaluation_status": "completed" if test_metrics is not None else "disabled",
        "timing": timing,
    }
    writer.write_json("summary.json", summary)
    return summary

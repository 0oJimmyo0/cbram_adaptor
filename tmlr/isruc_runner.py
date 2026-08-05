"""Auditable CBraMod-only ISRUC TMLR runner."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.isruc_dataset import EXPECTED_MODEL_SHAPE, ISRUCSequenceDataset
from models.lora import lora_diagnostics
from models.model_for_isruc import Model

from .artifact_writer import ArtifactWriter
from .eager_optimizer import EagerAdamW, EagerCosineAnnealing
from .faced_runner import (
    _gradient_report,
    _parameter_component,
    _snapshot,
    _torch_load,
    _update_report,
    resolve_device,
    set_seed,
)
from .isruc_provenance import EXPECTED_CHANNELS, audit_isruc
from .isruc_config import IsrucTMLRConfig
from .metrics import classification_metrics, selection_value
from .provenance import collect_environment, sha256_file
from .trainability import apply_trainability_contract, configure_training_modes


def _evaluate_sequence(model, loader, criterion, device, max_batches=None) -> Dict[str, Any]:
    model.eval()
    targets, predictions, losses = [], [], []
    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            logits = model(inputs)
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_labels = labels.reshape(-1)
            losses.append(float(criterion(flat_logits, flat_labels).detach().cpu()))
            targets.extend(flat_labels.cpu().tolist())
            predictions.extend(flat_logits.argmax(dim=-1).cpu().tolist())
    return classification_metrics(targets, predictions, losses, num_classes=5)


def _make_params(config: IsrucTMLRConfig) -> SimpleNamespace:
    values = config.to_dict()
    values.update({
        "foundation_dir": values["checkpoint"],
        "use_pretrained_weights": True,
        "num_of_classes": values["num_classes"],
        "cuda": 0,
    })
    return SimpleNamespace(**values)


def _runtime_geometry(model: nn.Module, sample: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    records: Dict[str, Any] = {"input_shape": list(sample.shape)}
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
        output = model(sample.to(device))
    for hook in hooks:
        hook.remove()
    expected_input = [sample.shape[0], 20, 6, 30, 200]
    if list(sample.shape) != expected_input:
        raise RuntimeError(f"ISRUC input geometry mismatch: {list(sample.shape)} != {expected_input}")
    if records.get("encoder_output_shape", [])[1:] != [6, 30, 200]:
        raise RuntimeError(f"ISRUC CBraMod grid mismatch: {records.get('encoder_output_shape')}")
    if list(output.shape) != [sample.shape[0], 20, 5]:
        raise RuntimeError(f"ISRUC sequence output mismatch: {list(output.shape)}")
    adapter = getattr(model.backbone, "native_axis_adapter", None)
    records.update({
        "dataset": "ISRUC",
        "sequence_length": 20,
        "channels": 6,
        "patches": 30,
        "embedding_dim": 200,
        "channel_attention_sequence_length": 6 if adapter is not None and hasattr(adapter, "channel_branch") else 0,
        "patch_attention_sequence_length": 30 if adapter is not None and hasattr(adapter, "patch_branch") else 0,
        "channel_spatial_interactions_active": bool(adapter is not None and hasattr(adapter, "channel_branch")),
        "patch_temporal_interactions_active": bool(adapter is not None and hasattr(adapter, "patch_branch")),
        "channel_names": EXPECTED_CHANNELS,
        "input_scale_divisor": 1.0,
        "requested_adapter_type": getattr(model.backbone, "adapter_type", "none"),
    })
    if adapter is not None:
        actual = {name for name in ("channel_branch", "patch_branch") if hasattr(adapter, name)}
        expected = {
            "channel": {"channel_branch"},
            "patch": {"patch_branch"},
            "channel_patch": {"channel_branch", "patch_branch"},
        }[model.backbone.adapter_type]
        if actual != expected:
            raise RuntimeError(f"ISRUC adapter branch mismatch: actual={actual} expected={expected}")
    return records


def run(config: IsrucTMLRConfig) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    config.values["dataset_path"] = str(Path(config.dataset_path).expanduser().resolve())
    config.values["channel_manifest"] = str((repo_root / config.channel_manifest).resolve()) if not Path(config.channel_manifest).is_absolute() else str(Path(config.channel_manifest).resolve())
    config.values["checkpoint"] = str(Path(config.checkpoint).expanduser().resolve())
    config.values["output_root"] = str((repo_root / config.output_root).resolve()) if not Path(config.output_root).is_absolute() else str(Path(config.output_root).resolve())

    audit = audit_isruc(config.dataset_path)
    manifest = json.loads(Path(config.channel_manifest).read_text(encoding="utf-8"))
    if manifest.get("channels") != EXPECTED_CHANNELS:
        raise ValueError("ISRUC channel manifest does not match the frozen bipolar channel order")
    if float(config.input_scale_divisor) != 1.0:
        raise ValueError("ISRUC runner refuses any scale divisor other than 1.0")

    run_id = config.run_id or f"isruc_cbramod_{config.method}_{config.adapter_type or 'none'}_s{config.seed}"
    writer = ArtifactWriter(config.output_root, run_id, overwrite=config.overwrite)
    writer.write_json("resolved_config.json", config.to_dict())
    writer.write_json("provenance.json", collect_environment(repo_root, config.device))
    writer.write_json("environment.json", collect_environment(repo_root, config.device))
    writer.write_json("dataset_audit.json", audit)
    if config.audit_only:
        writer.write_json("summary.json", {"status": "audit_only", "run_id": run_id})
        return {"status": "audit_only", "run_id": run_id}

    set_seed(config.seed)
    device = resolve_device(config.device)
    datasets = {
        mode: ISRUCSequenceDataset(config.dataset_path, mode, input_scale_divisor=1.0)
        for mode in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(int(config.loader_seed))
    loaders = {
        mode: DataLoader(
            dataset,
            batch_size=int(config.batch_size),
            shuffle=mode == "train",
            generator=generator if mode == "train" else None,
            num_workers=int(config.num_workers),
            pin_memory=device.type == "cuda",
        )
        for mode, dataset in datasets.items()
    }

    model = Model(_make_params(config)).to(device)
    checkpoint_report = dict(getattr(model, "checkpoint_report", {}))
    checkpoint_report.update({
        "checkpoint_path": str(Path(config.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(config.checkpoint),
        "strict_loading_status": bool(checkpoint_report.get("strict_loading_status", False)),
    })
    if not checkpoint_report["strict_loading_status"]:
        raise RuntimeError("ISRUC production model did not pass strict checkpoint loading")
    sample_inputs, _ = next(iter(loaders["val"]))
    geometry = _runtime_geometry(model, sample_inputs[: min(2, len(sample_inputs))], device)
    writer.write_json("structure_spec.json", geometry)
    writer.write_json("checkpoint_load_report.json", checkpoint_report)

    groups, trainability = apply_trainability_contract(
        model, config.method, config.adapter_type,
        lr=config.lr, head_lr=config.head_lr, adapter_lr=config.adapter_lr,
        lora_lr=config.lora_lr, upper_lr=config.upper_lr,
        weight_decay=config.weight_decay, head_weight_decay=config.head_weight_decay,
        adapter_weight_decay=config.adapter_weight_decay,
    )
    optimizer = EagerAdamW(groups)
    scheduler = EagerCosineAnnealing(
        optimizer,
        t_max=int(config.epochs) * len(loaders["train"]),
        eta_min=float(config.scheduler_eta_min),
    )
    writer.write_json("trainability_report.json", trainability)
    writer.write_json("optimizer_groups.json", {
        "contract": "isruc_cbramod_sequence",
        "implementation": "eager_adamw_equivalent",
        "groups": trainability["optimizer_groups"],
    })
    writer.write_json("training_mode_report.json", configure_training_modes(model, config.method, upper_k=config.upper_k))

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.label_smoothing)).to(device)
    best_score, best_epoch, best_record = float("-inf"), 0, None
    best_state = copy.deepcopy(model.state_dict())
    started = time.time()
    for epoch in range(1, int(config.epochs) + 1):
        epoch_start = time.time()
        configure_training_modes(model, config.method, upper_k=config.upper_k)
        before = _snapshot(model)
        model.train()
        # The mode controller must be reapplied after model.train() for frozen
        # methods; it restores eval mode on the frozen backbone and train mode
        # on the classifier/adapter.
        configure_training_modes(model, config.method, upper_k=config.upper_k)
        losses, last_grad = [], {}
        for batch_index, (inputs, labels) in enumerate(loaders["train"]):
            if config.max_train_batches is not None and batch_index >= int(config.max_train_batches):
                break
            inputs, labels = inputs.to(device), labels.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, 5), labels.reshape(-1))
            loss.backward()
            last_grad = _gradient_report(model)
            if config.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.clip_grad)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        validation = _evaluate_sequence(model, loaders["val"], criterion, device, config.max_val_batches)
        validation.update({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "elapsed_seconds": float(time.time() - epoch_start),
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        })
        writer.append_jsonl("metrics_by_epoch.jsonl", validation)
        diagnostics = model.backbone.get_adapter_diagnostics()
        diagnostics.update(last_grad)
        diagnostics.update(_update_report(model, before))
        diagnostics.update(lora_diagnostics(model.backbone))
        diagnostics["epoch"] = epoch
        writer.append_jsonl("adapter_diagnostics_by_epoch.jsonl", diagnostics)
        score = selection_value(validation, config.selection_metric)
        if score > best_score:
            best_score, best_epoch, best_record = score, epoch, copy.deepcopy(validation)
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state, strict=True)
    writer.write_json("best_validation_metrics.json", {
        "selection_metric": config.selection_metric,
        "selection_value": best_score,
        "best_epoch": best_epoch,
        "metrics": best_record,
    })
    writer.save_model(model)
    test = _evaluate_sequence(model, loaders["test"], criterion, device, config.max_test_batches)
    test.update({"selection_metric": config.selection_metric, "selected_epoch": best_epoch})
    writer.write_json("test_metrics.json", test)
    writer.write_json("per_class_metrics.json", {
        "precision": test["per_class_precision"], "recall": test["per_class_recall"],
        "f1": test["per_class_f1"], "support": test["per_class_support"],
    })
    writer.write_json("confusion_matrix.json", {"confusion_matrix": test["confusion_matrix"]})
    epoch_records = [json.loads(line) for line in (writer.run_dir / "metrics_by_epoch.jsonl").read_text().splitlines()]
    timing = {
        "training_wall_seconds": float(time.time() - started),
        "best_epoch": best_epoch,
        "time_per_epoch_seconds": [record["elapsed_seconds"] for record in epoch_records],
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
        "dataset": "ISRUC",
        "method": config.method,
        "adapter_type": config.adapter_type,
        "seed": config.seed,
        "trainable_parameter_count": trainability["trainable_parameter_count"],
        "best_epoch": best_epoch,
        "best_validation_selection_value": best_score,
        "test_evaluation_status": "completed",
    }
    writer.write_json("summary.json", summary)
    return summary

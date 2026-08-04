"""CBraMod-only SEED-V TMLR runner.

This module deliberately owns its loader and provenance audit.  It does not
import LaBraM or EEGxPlore training code; the shared high-level contract is
implemented against the original CBraMod backbone in this repository.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.cbramod import CBraMod
from tmlr.artifact_writer import ArtifactWriter
from tmlr.eager_optimizer import EagerAdamW, EagerCosineAnnealing
from tmlr.faced_runner import (
    _checkpoint_state,
    _gradient_report,
    _parameter_component,
    _snapshot,
    _torch_load,
    _update_report,
    resolve_device,
    set_seed,
)
from tmlr.metrics import evaluate_model, selection_value
from tmlr.provenance import collect_environment, sha256_file
from tmlr.trainability import apply_trainability_contract, configure_training_modes


EXPECTED_SHAPE = (62, 1, 200)


def _key_bytes(key: Any) -> bytes:
    return key.encode("utf-8") if isinstance(key, str) else bytes(key)


def _split_digest(keys: list[Any]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        encoded = _key_bytes(key)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class SeedVLoader(Dataset):
    def __init__(self, root: str, mode: str, manifest: Dict[str, Any], scale: float = 100.0):
        try:
            import lmdb
        except ImportError as exc:  # pragma: no cover
            raise ImportError("SEED-V requires lmdb in the active CBraMod environment") from exc
        self.root = str(root)
        self.mode = mode
        self._lmdb = lmdb
        self.db = None
        self.scale = float(scale)
        self.channel_names = list(manifest["cbramod_channel_names"])
        if len(self.channel_names) != EXPECTED_SHAPE[0] or len(set(self.channel_names)) != len(self.channel_names):
            raise RuntimeError("SEED-V manifest must contain 62 unique ordered channels")
        with lmdb.open(self.root, readonly=True, lock=False, readahead=False, meminit=False).begin(write=False) as txn:
            raw_keys = txn.get(b"__keys__")
        if raw_keys is None:
            raise KeyError(f"SEED-V LMDB missing __keys__: {self.root}")
        split_index = pickle.loads(raw_keys)
        if mode not in split_index or not split_index[mode]:
            raise ValueError(f"SEED-V split {mode!r} is absent or empty")
        self.keys = list(split_index[mode])
        with lmdb.open(self.root, readonly=True, lock=False, readahead=False, meminit=False).begin(write=False) as txn:
            raw = txn.get(_key_bytes(self.keys[0]))
        if raw is None:
            raise KeyError(f"SEED-V first key missing: {self.keys[0]!r}")
        sample = pickle.loads(raw)
        self._validate(sample, self.keys[0])

    def __len__(self) -> int:
        return len(self.keys)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["db"] = None
        return state

    def _get_db(self):
        if self.db is None:
            self.db = self._lmdb.open(
                self.root, readonly=True, lock=False, readahead=False,
                meminit=False, max_readers=512,
            )
        return self.db

    @staticmethod
    def _validate(sample: Dict[str, Any], key: Any) -> None:
        if not isinstance(sample, dict) or "sample" not in sample or "label" not in sample:
            raise ValueError(f"SEED-V record {key!r} must contain sample and label")
        x = np.asarray(sample["sample"])
        if tuple(x.shape) != EXPECTED_SHAPE:
            raise ValueError(f"SEED-V record {key!r} shape {tuple(x.shape)} != {EXPECTED_SHAPE}")
        if not np.isfinite(x).all():
            raise ValueError(f"SEED-V record {key!r} contains non-finite values")
        labels = np.asarray(sample["label"]).reshape(-1)
        if labels.size != 1 or not 0 <= int(labels[0]) <= 4:
            raise ValueError(f"SEED-V record {key!r} must contain one label in [0,4]")

    def __getitem__(self, index: int):
        key = self.keys[index]
        with self._get_db().begin(write=False) as txn:
            raw = txn.get(_key_bytes(key))
        if raw is None:
            raise KeyError(f"SEED-V record missing: {key!r}")
        sample = pickle.loads(raw)
        self._validate(sample, key)
        x = torch.as_tensor(np.asarray(sample["sample"]), dtype=torch.float32) / self.scale
        y = int(np.asarray(sample["label"]).reshape(-1)[0])
        return x, y


def audit_seedv(root: str, manifest_path: str, scale: float) -> Dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    datasets = {mode: SeedVLoader(root, mode, manifest, scale) for mode in ("train", "val", "test")}
    all_keys = [key for dataset in datasets.values() for key in dataset.keys]
    if len(all_keys) != len({_key_bytes(key) for key in all_keys}):
        raise RuntimeError("SEED-V split keys overlap")
    # A complete 12-GB LMDB scan is intentionally not performed at every
    # training launch.  The durable LaBraM audit records the full counts;
    # this CBraMod gate samples deterministic positions to verify schema,
    # finiteness, scaling, and label range without turning a launch audit into
    # a multi-minute filesystem sweep.
    counts = {mode: [0] * 5 for mode in datasets}
    sampled_counts = {}
    for mode, dataset in datasets.items():
        sample_count = min(128, len(dataset))
        indices = np.linspace(0, len(dataset) - 1, num=sample_count, dtype=int).tolist()
        for index in indices:
            _, label = dataset[index]
            counts[mode][label] += 1
        sampled_counts[mode] = sample_count
    return {
        "dataset": "SEED-V",
        "root": str(Path(root).resolve()),
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "shape": list(EXPECTED_SHAPE),
        "input_scale_divisor": float(scale),
        "channel_names": datasets["train"].channel_names,
        "split_counts": {mode: len(dataset) for mode, dataset in datasets.items()},
        "sampled_class_counts": counts,
        "sampled_records_per_split": sampled_counts,
        "split_key_sha256": {mode: _split_digest(dataset.keys) for mode, dataset in datasets.items()},
        "split_key_overlap": False,
        "metadata_limitation": "Legacy LMDB stores no channel names; external order metadata is recorded separately.",
    }


class SeedVClassifier(nn.Module):
    def __init__(self, config, checkpoint_path: Path, device: torch.device):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30, n_layer=12, nhead=8,
        )
        before = sum(parameter.numel() for parameter in self.backbone.parameters())
        checkpoint = _torch_load(checkpoint_path, device)
        state = _checkpoint_state(checkpoint)
        report = self.backbone.load_state_dict(state, strict=True)
        if report.missing_keys or report.unexpected_keys:
            raise RuntimeError(f"Strict CBraMod checkpoint mismatch: {report}")
        self.backbone.proj_out = nn.Identity()
        if config.method in {"interaction_aligned", "native_full_finetune"}:
            self.backbone.enable_interaction_adapter(
                config.adapter_type, config.adapter_bottleneck, config.adapter_heads,
                config.adapter_dropout, config.adapter_init_alpha, config.adapter_gamma,
                config.adapter_zero_init_output, config.adapter_seed,
                config.allow_singleton_patch_control,
            )
        elif config.method == "generic_bottleneck":
            self.backbone.enable_generic_adapter(
                config.generic_bottleneck, config.adapter_init_alpha, config.adapter_gamma,
                config.adapter_zero_init_output, config.adapter_seed, "generic_bottleneck",
            )
        elif config.method == "axis_blind":
            self.backbone.enable_generic_adapter(
                config.axis_blind_bottleneck, config.adapter_init_alpha, config.adapter_gamma,
                config.adapter_zero_init_output, config.adapter_seed, "axis_blind",
            )
        elif config.method == "lora":
            from models.lora import inject_qkv_lora
            inject_qkv_lora(self.backbone, rank=config.lora_rank, alpha=config.lora_alpha)
        elif config.method not in {"full_finetune", "frozen_probe", "upper_k_finetune"}:
            raise NotImplementedError(config.method)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(62 * 1 * 200, 4 * 200),
            nn.ELU(), nn.Dropout(config.dropout),
            nn.Linear(4 * 200, 200),
            nn.ELU(), nn.Dropout(config.dropout),
            nn.Linear(200, 5),
        )
        self.checkpoint_report = {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "strict_loading_status": True,
            "missing_keys": list(report.missing_keys),
            "unexpected_keys": list(report.unexpected_keys),
            "backbone_parameter_count_before_attachment": int(before),
            "checkpoint_state_key_count": len(state),
            "input_geometry": list(EXPECTED_SHAPE),
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        if tuple(features.shape[1:]) != EXPECTED_SHAPE:
            raise RuntimeError(f"CBraMod SEED-V output geometry mismatch: {tuple(features.shape)}")
        return self.classifier(features)


def _geometry(model: SeedVClassifier, inputs: torch.Tensor) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        output = model.backbone(inputs)
    if list(inputs.shape[1:]) != list(EXPECTED_SHAPE) or list(output.shape[1:]) != list(EXPECTED_SHAPE):
        raise RuntimeError(f"SEED-V geometry mismatch: input={tuple(inputs.shape)} output={tuple(output.shape)}")
    adapter = model.backbone.native_axis_adapter
    return {
        "input_shape": list(inputs.shape),
        "encoder_output_shape": list(output.shape),
        "channels": 62,
        "patches": 1,
        "embedding_dim": 200,
        "channel_eligibility": True,
        "patch_eligibility": False,
        "patch_reason": "singleton temporal patch axis; patch attention is not a temporal interaction",
        "requested_adapter_type": model.backbone.adapter_type,
        "adapter_geometry": adapter.get_diagnostics() if adapter is not None else {},
    }


def run(config) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    config.values["dataset_path"] = str(Path(config.dataset_path).expanduser().resolve())
    config.values["channel_manifest"] = str((repo_root / config.channel_manifest).resolve()) if not Path(config.channel_manifest).is_absolute() else str(Path(config.channel_manifest).resolve())
    config.values["checkpoint"] = str(Path(config.checkpoint).expanduser().resolve())
    config.values["output_root"] = str((repo_root / config.output_root).resolve()) if not Path(config.output_root).is_absolute() else str(Path(config.output_root).resolve())
    audit = audit_seedv(config.dataset_path, config.channel_manifest, config.input_scale_divisor)
    environment = collect_environment(repo_root, config.device)
    run_id = config.run_id or f"seedv_{config.method}_{config.adapter_type or 'none'}_s{config.seed}"
    writer = ArtifactWriter(config.output_root, run_id, overwrite=config.overwrite)
    writer.write_json("resolved_config.json", config.to_dict())
    writer.write_json("provenance.json", environment)
    writer.write_json("environment.json", environment)
    writer.write_json("dataset_audit.json", audit)
    if config.audit_only:
        writer.write_json("summary.json", {"status": "audit_only", "run_id": run_id})
        return {"status": "audit_only", "run_id": run_id}

    set_seed(config.seed)
    device = resolve_device(config.device)
    manifest = json.loads(Path(config.channel_manifest).read_text(encoding="utf-8"))
    datasets = {mode: SeedVLoader(config.dataset_path, mode, manifest, config.input_scale_divisor) for mode in ("train", "val", "test")}
    generator = torch.Generator().manual_seed(int(config.loader_seed))
    loaders = {
        mode: DataLoader(dataset, batch_size=config.batch_size, shuffle=(mode == "train"),
                         num_workers=config.num_workers, pin_memory=(device.type == "cuda"),
                         generator=generator if mode == "train" else None)
        for mode, dataset in datasets.items()
    }
    model = SeedVClassifier(config, Path(config.checkpoint), device).to(device)
    sample_inputs, _ = next(iter(loaders["val"]))
    geometry = _geometry(model, sample_inputs[: min(2, len(sample_inputs))].to(device))
    writer.write_json("structure_spec.json", {"dataset": "SEED-V", "runtime": geometry})
    writer.write_json("checkpoint_load_report.json", model.checkpoint_report)
    groups, trainability = apply_trainability_contract(
        model, config.method, config.adapter_type, lr=config.lr,
        head_lr=config.head_lr, adapter_lr=config.adapter_lr,
        lora_lr=config.lora_lr, upper_lr=config.upper_lr,
        weight_decay=config.weight_decay,
        head_weight_decay=config.head_weight_decay,
        adapter_weight_decay=config.adapter_weight_decay,
    )
    optimizer = EagerAdamW(groups)
    scheduler = EagerCosineAnnealing(
        optimizer, t_max=int(config.epochs) * len(loaders["train"]),
        eta_min=float(config.scheduler_eta_min),
    )
    writer.write_json("trainability_report.json", trainability)
    writer.write_json("optimizer_groups.json", {
        "contract": "seedv_cbramod",
        "implementation": "eager_adamw_equivalent",
        "reason": "PyTorch 2.12 torch._dynamo optimizer wrapper blocks in the CBraMod runtime",
        "betas": [0.9, 0.999], "eps": 1e-8,
        "groups": trainability["optimizer_groups"],
    })
    writer.write_json("training_mode_report.json", configure_training_modes(
        model, config.method, upper_k=config.upper_k,
    ))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.label_smoothing)).to(device)
    print(f"[seedv] optimizer and criterion ready on {device}; entering training", flush=True)
    best_score, best_epoch, best_record = float("-inf"), 0, None
    best_state = copy.deepcopy(model.state_dict())
    started = time.time()
    for epoch in range(1, int(config.epochs) + 1):
        epoch_start = time.time()
        configure_training_modes(model, config.method, upper_k=config.upper_k)
        before = _snapshot(model)
        losses, last_grad = [], {}
        for batch_index, (inputs, labels) in enumerate(loaders["train"]):
            if batch_index == 0:
                print(f"[seedv] epoch={epoch} first train batch loaded", flush=True)
            inputs, labels = inputs.to(device), labels.to(device).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            last_grad = _gradient_report(model)
            if config.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.clip_grad)
            optimizer.step(); scheduler.step(); losses.append(float(loss.detach().cpu()))
            if batch_index == 0:
                print(f"[seedv] epoch={epoch} first optimizer step complete", flush=True)
            if config.max_train_batches is not None and batch_index + 1 >= int(config.max_train_batches):
                break
        print(f"[seedv] epoch={epoch} training complete; evaluating validation", flush=True)
        validation = evaluate_model(
            model, loaders["val"], criterion, device, num_classes=5,
            max_batches=config.max_val_batches,
        )
        validation.update({"epoch": epoch, "train_loss": float(np.mean(losses)), "elapsed_seconds": float(time.time() - epoch_start), "learning_rates": [float(g["lr"]) for g in optimizer.param_groups]})
        writer.append_jsonl("metrics_by_epoch.jsonl", validation)
        diagnostics = model.backbone.get_adapter_diagnostics()
        diagnostics.update(last_grad); diagnostics.update(_update_report(model, before)); diagnostics["epoch"] = epoch
        writer.append_jsonl("adapter_diagnostics_by_epoch.jsonl", diagnostics)
        score = selection_value(validation, config.selection_metric)
        if score > best_score:
            best_score, best_epoch, best_record = score, epoch, copy.deepcopy(validation)
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state, strict=True)
    writer.write_json("best_validation_metrics.json", {"selection_metric": config.selection_metric, "selection_value": best_score, "best_epoch": best_epoch, "metrics": best_record})
    writer.save_model(model)
    test = evaluate_model(
        model, loaders["test"], criterion, device, num_classes=5,
        max_batches=config.max_test_batches,
    )
    test.update({"selection_metric": config.selection_metric, "selected_epoch": best_epoch})
    writer.write_json("test_metrics.json", test)
    writer.write_json("per_class_metrics.json", {"precision": test["per_class_precision"], "recall": test["per_class_recall"], "f1": test["per_class_f1"], "support": test["per_class_support"]})
    writer.write_json("confusion_matrix.json", {"confusion_matrix": test["confusion_matrix"]})
    epoch_records = [json.loads(line) for line in (writer.run_dir / "metrics_by_epoch.jsonl").read_text().splitlines()]
    timing = {"training_wall_seconds": float(time.time() - started), "best_epoch": best_epoch, "time_per_epoch_seconds": [r["elapsed_seconds"] for r in epoch_records], "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0}
    writer.write_json("timing.json", timing)
    writer.write_json("memory.json", {"peak_gpu_memory_bytes": timing["peak_gpu_memory_bytes"], "cuda_available": bool(torch.cuda.is_available())})
    summary = {"status": "completed", "run_id": run_id, "artifact_dir": str(writer.run_dir), "dataset": "SEED-V", "method": config.method, "adapter_type": config.adapter_type, "seed": config.seed, "trainable_parameter_count": trainability["trainable_parameter_count"], "best_epoch": best_epoch, "best_validation_selection_value": best_score, "test_evaluation_status": "completed"}
    writer.write_json("summary.json", summary)
    return summary

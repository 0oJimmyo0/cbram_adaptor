"""Unambiguous FACED classification metrics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)


def selection_value(metrics: Dict[str, Any], metric: str) -> float:
    key = str(metric).strip().lower()
    if key not in {"cohen_kappa", "balanced_accuracy", "macro_f1"}:
        raise ValueError(f"Unsupported selection metric {metric!r}")
    return float(metrics[key])


def classification_metrics(
    targets: Iterable[int],
    predictions: Iterable[int],
    losses: Optional[Iterable[float]] = None,
    num_classes: int = 9,
) -> Dict[str, Any]:
    # reshape(-1) avoids the singleton-batch squeeze failure mode.
    y_true = np.asarray(list(targets), dtype=np.int64).reshape(-1)
    y_pred = np.asarray(list(predictions), dtype=np.int64).reshape(-1)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError(f"targets/predictions must be nonempty and equal-shaped, got {y_true.shape}/{y_pred.shape}")
    labels = list(range(int(num_classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if np.unique(np.concatenate((y_true, y_pred))).size < 2:
        kappa = 1.0 if np.array_equal(y_true, y_pred) else 0.0
    else:
        kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels))
    loss_values = [] if losses is None else [float(value) for value in losses]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "cohen_kappa": kappa,
        "cross_entropy_loss": float(np.mean(loss_values)) if loss_values else None,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "sample_count": int(y_true.size),
    }


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, num_classes=9, max_batches=None) -> Dict[str, Any]:
    model.eval()
    targets = []
    predictions = []
    losses = []
    for batch_index, (inputs, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).reshape(-1)
        logits = model(inputs)
        loss = criterion(logits, labels)
        pred = logits.argmax(dim=-1)
        targets.extend(labels.detach().cpu().reshape(-1).tolist())
        predictions.extend(pred.detach().cpu().reshape(-1).tolist())
        losses.append(float(loss.detach().cpu()))
    return classification_metrics(targets, predictions, losses, num_classes=num_classes)

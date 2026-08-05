"""Serialized-data provenance audit for the isolated CBraMod ISRUC study."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np

from datasets.isruc_dataset import (
    EXPECTED_LABEL_SHAPE,
    EXPECTED_STORED_SHAPE,
    TEST_SUBJECTS,
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    paired_paths,
)


EXPECTED_LABELS = {0, 1, 2, 3, 4}
EXPECTED_CHANNELS = ["F3-A2", "C3-A2", "O1-A2", "F4-A1", "C4-A1", "O2-A1"]


def _digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _subject_audit(root: Path, subjects: tuple[int, ...]) -> tuple[list[dict[str, Any]], Counter, dict[str, float]]:
    records = []
    class_counts: Counter = Counter()
    value_min = float("inf")
    value_max = float("-inf")
    train_sum = 0.0
    train_sq_sum = 0.0
    train_count = 0
    for subject in subjects:
        pairs = paired_paths(root, subject)
        for signal_path, label_path in pairs:
            signal = np.load(signal_path, mmap_mode="r", allow_pickle=False)
            labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
            if tuple(signal.shape) != EXPECTED_STORED_SHAPE:
                raise ValueError(f"{signal_path}: shape {tuple(signal.shape)} != {EXPECTED_STORED_SHAPE}")
            if tuple(labels.shape) != EXPECTED_LABEL_SHAPE:
                raise ValueError(f"{label_path}: shape {tuple(labels.shape)} != {EXPECTED_LABEL_SHAPE}")
            if not np.isfinite(signal).all() or not np.isfinite(labels).all():
                raise ValueError(f"Non-finite values in {signal_path} or {label_path}")
            labels_int = np.asarray(labels, dtype=np.int64)
            label_values = set(np.unique(labels_int).tolist())
            if not label_values.issubset(EXPECTED_LABELS):
                raise ValueError(f"{label_path}: labels {sorted(label_values)} are not mapped to 0..4")
            class_counts.update(labels_int.tolist())
            value_min = min(value_min, float(signal.min()))
            value_max = max(value_max, float(signal.max()))
            if subjects == TRAIN_SUBJECTS:
                values = np.asarray(signal, dtype=np.float64)
                train_sum += float(values.sum())
                train_sq_sum += float(np.square(values).sum())
                train_count += int(values.size)
            records.append({
                "subject": int(subject),
                "signal": str(signal_path.relative_to(root)),
                "label": str(label_path.relative_to(root)),
                "key": f"{subject}/{signal_path.stem}",
                "sequence_length": int(labels.shape[0]),
            })
    stats = {
        "value_min": value_min,
        "value_max": value_max,
        "train_mean": train_sum / train_count if train_count else None,
        "train_std": max(train_sq_sum / train_count - (train_sum / train_count) ** 2, 0.0) ** 0.5 if train_count else None,
        "train_value_count": train_count,
    }
    return records, class_counts, stats


def audit_isruc(root: str | Path) -> Dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ISRUC data root not found: {root}")
    split_subjects = {"train": TRAIN_SUBJECTS, "val": VAL_SUBJECTS, "test": TEST_SUBJECTS}
    split_records: dict[str, list[dict[str, Any]]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    split_labels: dict[str, dict[str, int]] = {}
    split_stats: dict[str, dict[str, float]] = {}
    for split, subjects in split_subjects.items():
        records, counts, stats = _subject_audit(root, subjects)
        split_records[split] = records
        split_counts[split] = {"subjects": len(subjects), "sequences": len(records), "epochs": len(records) * 20}
        split_labels[split] = {str(label): int(counts[label]) for label in sorted(EXPECTED_LABELS)}
        split_stats[split] = stats

    subject_sets = {split: set(subjects) for split, subjects in split_subjects.items()}
    subject_overlap = {
        f"{left}_{right}": sorted(subject_sets[left] & subject_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    all_keys = [record["key"] for records in split_records.values() for record in records]
    duplicate_keys = sorted(key for key, count in Counter(all_keys).items() if count > 1)
    if duplicate_keys or any(subject_overlap.values()):
        raise ValueError(f"ISRUC split overlap detected: duplicate_keys={duplicate_keys[:3]} subject_overlap={subject_overlap}")

    return {
        "dataset": "ISRUC-Sleep Subgroup I",
        "absolute_dataset_path": str(root),
        "serialized_layout": "seq/ISRUC-group1-{subject}/*.npy and labels/ISRUC-group1-{subject}/*.npy",
        "split_subjects": {split: list(subjects) for split, subjects in split_subjects.items()},
        "split_counts": split_counts,
        "split_class_counts": split_labels,
        "split_key_sha256": {split: _digest([record["key"] for record in split_records[split]]) for split in split_records},
        "all_key_sha256": _digest(all_keys),
        "stored_signal_shape": list(EXPECTED_STORED_SHAPE),
        "model_signal_shape": [20, 6, 30, 200],
        "label_shape": list(EXPECTED_LABEL_SHAPE),
        "label_vocabulary": sorted(EXPECTED_LABELS),
        "label_mapping": {"0": 0, "1": 1, "2": 2, "3": 3, "5": 4},
        "ordered_bipolar_channels": EXPECTED_CHANNELS,
        "sampling_rate_hz": 200,
        "epoch_seconds": 30,
        "preprocessing": {
            "serialized_arrays_already_filtered": True,
            "bandpass_hz": [0.3, 35.0],
            "notch_hz": 50.0,
            "resampling": False,
            "additional_filtering_in_loader": False,
        },
        "input_scale_divisor": 1.0,
        "value_statistics_by_split": split_stats,
        "finite_value_check": True,
        "exact_signal_label_basename_pairing": True,
        "numeric_file_order_check": True,
        "duplicate_key_check": not duplicate_keys,
        "subject_overlap_check": not any(subject_overlap.values()),
        "subject_overlap": subject_overlap,
        "audit_scope": "serialized-only; channel order follows the frozen preprocessing contract",
    }


def write_audit(root: str | Path, output: str | Path) -> Dict[str, Any]:
    report = audit_isruc(root)
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

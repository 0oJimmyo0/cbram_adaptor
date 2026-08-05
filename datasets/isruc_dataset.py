"""Deterministic CBraMod ISRUC sequence loader.

The serialized ISRUC files contain twenty consecutive 30-second epochs as
``[20, 6, 6000]`` arrays.  This module exposes the native CBraMod geometry
``[20, 6, 30, 200]`` without refiltering or applying the `/100` rule used by
the FACED and SEED-V loaders.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


EXPECTED_STORED_SHAPE = (20, 6, 6000)
EXPECTED_MODEL_SHAPE = (20, 6, 30, 200)
EXPECTED_LABEL_SHAPE = (20,)
EXPECTED_SUBJECTS = tuple(range(1, 101))
TRAIN_SUBJECTS = tuple(range(1, 81))
VAL_SUBJECTS = tuple(range(81, 91))
TEST_SUBJECTS = tuple(range(91, 101))

_NUMERIC_SUFFIX = re.compile(r"-(\d+)$")


def _numeric_key(path: Path) -> tuple[int, str]:
    match = _NUMERIC_SUFFIX.search(path.stem)
    return (int(match.group(1)) if match else -1, path.name)


def _subject_paths(root: Path, subject: int) -> tuple[Path, Path]:
    name = f"ISRUC-group1-{int(subject)}"
    return root / "seq" / name, root / "labels" / name


def paired_paths(root: str | Path, subject: int) -> List[Tuple[Path, Path]]:
    """Return exact same-stem signal/label pairs in numeric order."""
    root = Path(root).expanduser().resolve()
    seq_dir, label_dir = _subject_paths(root, subject)
    if not seq_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing ISRUC subject directories for subject {subject}: {seq_dir}, {label_dir}")
    signals = {path.stem: path for path in seq_dir.glob("*.npy")}
    labels = {path.stem: path for path in label_dir.glob("*.npy")}
    if set(signals) != set(labels):
        raise ValueError(
            f"ISRUC signal/label basename mismatch for subject {subject}: "
            f"signals_only={sorted(set(signals) - set(labels))[:3]} "
            f"labels_only={sorted(set(labels) - set(signals))[:3]}"
        )
    return [(signals[key], labels[key]) for key in sorted(signals, key=lambda key: _numeric_key(signals[key]))]


def split_subjects(mode: str) -> tuple[int, ...]:
    mode = str(mode).strip().lower()
    if mode == "train":
        return TRAIN_SUBJECTS
    if mode in {"val", "validation"}:
        return VAL_SUBJECTS
    if mode == "test":
        return TEST_SUBJECTS
    raise ValueError(f"Unknown ISRUC split {mode!r}; expected train, val, or test")


class ISRUCSequenceDataset(Dataset):
    """Subject-wise ISRUC dataset with explicit sequence geometry."""

    def __init__(self, root: str | Path, mode: str, input_scale_divisor: float = 1.0):
        self.root = Path(root).expanduser().resolve()
        self.mode = str(mode).strip().lower()
        self.input_scale_divisor = float(input_scale_divisor)
        if self.input_scale_divisor != 1.0:
            raise ValueError("ISRUC uses input_scale_divisor=1.0; do not apply CBraMod /100 scaling")
        self.records: List[Tuple[Path, Path, int]] = []
        for subject in split_subjects(self.mode):
            self.records.extend((signal, label, subject) for signal, label in paired_paths(self.root, subject))
        if not self.records:
            raise ValueError(f"ISRUC split {self.mode!r} is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal_path, label_path, _subject = self.records[int(index)]
        signal = np.load(signal_path, allow_pickle=False)
        labels = np.load(label_path, allow_pickle=False)
        if tuple(signal.shape) != EXPECTED_STORED_SHAPE:
            raise RuntimeError(f"ISRUC signal shape {tuple(signal.shape)} != {EXPECTED_STORED_SHAPE}: {signal_path}")
        if tuple(labels.shape) != EXPECTED_LABEL_SHAPE:
            raise RuntimeError(f"ISRUC label shape {tuple(labels.shape)} != {EXPECTED_LABEL_SHAPE}: {label_path}")
        if not np.isfinite(signal).all() or not np.isfinite(labels).all():
            raise RuntimeError(f"Non-finite ISRUC sample: {signal_path} / {label_path}")
        signal = signal.reshape(EXPECTED_MODEL_SHAPE).astype(np.float32, copy=False)
        labels = labels.astype(np.int64, copy=False)
        if not set(np.unique(labels).tolist()).issubset({0, 1, 2, 3, 4}):
            raise RuntimeError(f"ISRUC labels are not mapped to 0..4: {label_path}")
        return torch.from_numpy(signal), torch.from_numpy(labels)


class LoadDataset:
    """Compatibility wrapper for the legacy ``finetune_main.py`` entrypoint."""

    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        generator = torch.Generator().manual_seed(int(getattr(self.params, "loader_seed", self.params.seed)))
        datasets = {mode: ISRUCSequenceDataset(self.datasets_dir, mode, 1.0) for mode in ("train", "val", "test")}
        return {
            mode: DataLoader(
                dataset,
                batch_size=int(self.params.batch_size),
                shuffle=mode == "train",
                generator=generator if mode == "train" else None,
                num_workers=int(getattr(self.params, "num_workers", 0)),
            )
            for mode, dataset in datasets.items()
        }

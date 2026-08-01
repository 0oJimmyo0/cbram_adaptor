"""FACED data, repository, environment, and checkpoint provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import lmdb
import numpy as np
import torch


EXPECTED_SHAPE = (32, 10, 200)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git(*args: str, cwd: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=None if cwd is None else str(cwd),
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def collect_environment(repo_root: str | Path, device: str) -> Dict[str, Any]:
    repo_root = str(Path(repo_root).resolve())
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_devices": cuda_devices,
        "requested_device": device,
        "git_repo": repo_root,
        "git_commit": _git("rev-parse", "HEAD", cwd=repo_root),
        "git_branch": _git("branch", "--show-current", cwd=repo_root),
        "git_status": _git("status", "--short", cwd=repo_root).splitlines(),
        "git_dirty": bool(_git("status", "--porcelain", cwd=repo_root)),
        "git_diff_sha256": hashlib.sha256(_git("diff", cwd=repo_root).encode()).hexdigest(),
        "command": " ".join(sys.argv),
    }


def _as_key(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode()
    raise TypeError(f"LMDB key must be str or bytes, got {type(value).__name__}")


def _display_key(value: Any) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _subject_id(key: str) -> Optional[str]:
    if ".pkl-" in key:
        return key.split(".pkl-", 1)[0]
    if key.startswith("sub"):
        return key.split("-", 1)[0]
    return None


def _trial_id(key: str) -> Optional[tuple[str, ...]]:
    subject = _subject_id(key)
    if subject is None:
        return None
    suffix = key.split(".pkl-", 1)[1] if ".pkl-" in key else key[len(subject):].lstrip("-")
    parts = tuple(part for part in suffix.split("-") if part)
    return (subject, *parts)


def _load_expected_channels(path: str | Path) -> tuple[list[str], str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FACED channel manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    channels = payload.get("channels") if isinstance(payload, Mapping) else payload
    if not isinstance(channels, list) or len(channels) != EXPECTED_SHAPE[0]:
        raise ValueError("Canonical FACED channel manifest must contain exactly 32 channels")
    if len(set(channels)) != len(channels):
        raise ValueError("Canonical FACED channel manifest contains duplicates")
    return [str(item) for item in channels], sha256_file(path)


def audit_faced_dataset(data_dir: str | Path, channel_manifest: str | Path, scale_divisor: float = 100.0) -> Dict[str, Any]:
    """Audit FACED metadata and every stored sample without writing samples."""
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"FACED dataset directory not found: {data_dir}")
    expected_channels, channel_hash = _load_expected_channels(channel_manifest)
    env = lmdb.open(str(data_dir), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin(write=False) as txn:
            raw_split_payload = txn.get(b"__keys__")
            if raw_split_payload is None:
                raise KeyError("FACED LMDB is missing __keys__")
            split_payload = pickle.loads(raw_split_payload)
            if not isinstance(split_payload, Mapping):
                raise ValueError("FACED __keys__ must unpickle to a mapping")
            raw_channel_names = txn.get(b"__channel_names__") or txn.get(b"channel_names")
            if raw_channel_names is None:
                raise KeyError("FACED LMDB is missing __channel_names__")
            stored_channels = list(pickle.loads(raw_channel_names))
            if stored_channels != expected_channels:
                raise ValueError(
                    "FACED channel order mismatch: "
                    f"stored={stored_channels!r} expected={expected_channels!r}"
                )

            split_keys: Dict[str, list[str]] = {}
            for split in ("train", "val", "test"):
                if split not in split_payload:
                    raise KeyError(f"FACED LMDB is missing split {split!r}")
                split_keys[split] = [_display_key(key) for key in split_payload[split]]
            all_keys = [key for split in split_keys.values() for key in split]
            duplicate_keys = sorted(key for key, count in Counter(all_keys).items() if count > 1)
            key_sets = {split: set(keys) for split, keys in split_keys.items()}
            overlap = {
                f"{left}_{right}": sorted(key_sets[left] & key_sets[right])
                for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
            }
            subject_sets = {
                split: {_subject_id(key) for key in keys if _subject_id(key) is not None}
                for split, keys in split_keys.items()
            }
            subject_overlap = {
                f"{left}_{right}": sorted(subject_sets[left] & subject_sets[right])
                for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
            }
            trial_sets = {
                split: {_trial_id(key) for key in keys if _trial_id(key) is not None}
                for split, keys in split_keys.items()
            }
            trial_overlap = {
                f"{left}_{right}": sorted(trial_sets[left] & trial_sets[right])
                for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
            }
            if duplicate_keys or any(overlap.values()) or any(subject_overlap.values()) or any(trial_overlap.values()):
                raise ValueError(
                    "FACED split overlap/duplicate detected: "
                    f"duplicates={len(duplicate_keys)} key_overlap={overlap} "
                    f"subject_overlap={subject_overlap} trial_overlap={trial_overlap}"
                )

            labels_by_split: Dict[str, Dict[str, int]] = {}
            shapes = set()
            raw_dtypes = set()
            scaled_dtypes = set()
            finite_failures = []
            value_min = float("inf")
            value_max = float("-inf")
            label_values = set()
            for split, keys in split_keys.items():
                counts: Dict[str, int] = {}
                for display_key in keys:
                    raw = txn.get(_as_key(display_key))
                    if raw is None:
                        raise KeyError(f"FACED LMDB key not found: {display_key!r}")
                    pair = pickle.loads(raw)
                    sample = np.asarray(pair["sample"])
                    if tuple(sample.shape) != EXPECTED_SHAPE:
                        raise ValueError(
                            f"FACED key {display_key!r} has shape {tuple(sample.shape)}, expected {EXPECTED_SHAPE}"
                        )
                    shapes.add(tuple(sample.shape))
                    raw_dtypes.add(str(sample.dtype))
                    value_min = min(value_min, float(np.nanmin(sample)))
                    value_max = max(value_max, float(np.nanmax(sample)))
                    if not np.isfinite(sample).all():
                        finite_failures.append(display_key)
                    scaled = np.asarray(sample / float(scale_divisor), dtype=np.float32)
                    scaled_dtypes.add(str(scaled.dtype))
                    if not np.isfinite(scaled).all():
                        finite_failures.append(display_key + "::scaled")
                    label = int(pair["label"])
                    label_values.add(label)
                    counts[str(label)] = counts.get(str(label), 0) + 1
                labels_by_split[split] = counts
            if finite_failures:
                raise ValueError(f"FACED non-finite samples detected: {finite_failures[:3]}")

            db_stat = txn.stat()
    finally:
        env.close()

    expected_labels = set(range(9))
    if label_values != expected_labels:
        raise ValueError(f"FACED label vocabulary mismatch: {sorted(label_values)}")
    split_hash = sha256_json(split_keys)
    return {
        "dataset": "FACED",
        "absolute_dataset_path": str(data_dir),
        "lmdb_metadata": db_stat,
        "split_key_counts": {split: len(keys) for split, keys in split_keys.items()},
        "ordered_channel_names": expected_channels,
        "channel_manifest_sha256": channel_hash,
        "split_manifest_sha256": split_hash,
        "sample_shape": list(EXPECTED_SHAPE),
        "label_vocabulary": sorted(label_values),
        "class_counts_by_split": labels_by_split,
        "label_min": min(label_values),
        "label_max": max(label_values),
        "number_of_classes": len(label_values),
        "scaling_rule": f"stored sample / {float(scale_divisor):g}",
        "sample_dtype_before_conversion": sorted(raw_dtypes),
        "sample_dtype_after_conversion": sorted(scaled_dtypes),
        "finite_value_check": True,
        "duplicate_key_check": True,
        "split_key_overlap_check": True,
        "subject_overlap_check": True,
        "session_trial_overlap_check": True,
        "value_min_before_scaling": value_min,
        "value_max_before_scaling": value_max,
        "overlap_summary": {"key": overlap, "subject": subject_overlap, "session_trial": trial_overlap},
    }


def structure_spec() -> Dict[str, Any]:
    return {
        "backbone": "cbramod",
        "dataset": "FACED",
        "insertion_site": "post_encoder",
        "expected_geometry": {"channels": 32, "patches": 10, "embedding_dim": 200, "branch_dim": 100},
        "feature_partition": {"channel": "first_half", "patch": "second_half"},
        "eligibility": {"channel": True, "patch": True},
    }

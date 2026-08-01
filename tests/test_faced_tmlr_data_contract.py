import json
import pickle

import lmdb
import numpy as np
import pytest

from datasets.faced_dataset import CustomDataset
from tmlr.provenance import audit_faced_dataset


CHANNELS = [
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
    "FC6", "CZ", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6",
    "PZ", "P3", "P4", "P7", "P8", "PO3", "PO4", "OZ", "O1", "O2",
    "TP9", "TP10",
]


def make_dataset(tmp_path):
    root = tmp_path / "faced"
    root.mkdir()
    with (root / "channel_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"dataset": "FACED", "version": 1, "channels": CHANNELS}, handle)
    env = lmdb.open(str(root), map_size=256 * 1024 * 1024)
    splits = {"train": [], "val": [], "test": []}
    with env.begin(write=True) as txn:
        txn.put(b"__channel_names__", pickle.dumps(CHANNELS))
        for split_index, split in enumerate(splits):
            for label in range(9):
                key = f"sub{split_index:03d}.pkl-{label}-0"
                splits[split].append(key)
                txn.put(key.encode(), pickle.dumps({"sample": np.zeros((32, 10, 200), dtype=np.float32), "label": label}))
        txn.put(b"__keys__", pickle.dumps(splits))
    env.close()
    return root


def test_faced_audit_contract_passes(tmp_path):
    root = make_dataset(tmp_path)
    audit = audit_faced_dataset(root, root / "channel_manifest.json")
    assert audit["sample_shape"] == [32, 10, 200]
    assert audit["split_key_counts"] == {"train": 9, "val": 9, "test": 9}
    assert audit["label_vocabulary"] == list(range(9))
    assert audit["overlap_summary"]["key"] == {"train_val": [], "train_test": [], "val_test": []}


def test_channel_order_mismatch_fails(tmp_path):
    root = make_dataset(tmp_path)
    with (root / "channel_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"channels": list(reversed(CHANNELS))}, handle)
    with pytest.raises(ValueError, match="channel order mismatch"):
        audit_faced_dataset(root, root / "channel_manifest.json")


def test_split_overlap_fails(tmp_path):
    root = make_dataset(tmp_path)
    env = lmdb.open(str(root), map_size=256 * 1024 * 1024)
    with env.begin(write=True) as txn:
        payload = pickle.loads(txn.get(b"__keys__"))
        payload["val"][0] = payload["train"][0]
        txn.put(b"__keys__", pickle.dumps(payload))
    env.close()
    with pytest.raises(ValueError, match="overlap"):
        audit_faced_dataset(root, root / "channel_manifest.json")


def test_missing_key_fails_clearly(tmp_path):
    root = make_dataset(tmp_path)
    env = lmdb.open(str(root), map_size=256 * 1024 * 1024)
    with env.begin(write=True) as txn:
        txn.delete(txn.get(b"__keys__") and b"sub000.pkl-0-0")
    env.close()
    with pytest.raises(KeyError, match="key not found"):
        audit_faced_dataset(root, root / "channel_manifest.json")

    # Dataset access uses the same explicit missing-key failure.
    dataset = CustomDataset(root, mode="train")
    with pytest.raises(KeyError):
        dataset[0]

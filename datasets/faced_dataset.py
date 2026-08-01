import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import json
import lmdb
import pickle


EXPECTED_SHAPE = (32, 10, 200)

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
    ):
        super(CustomDataset, self).__init__()
        if mode not in {'train', 'val', 'test'}:
            raise ValueError(f"FACED mode must be train/val/test, got {mode!r}")
        self.data_dir = os.path.abspath(data_dir)
        self.mode = mode
        self.db = lmdb.open(self.data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            raw_keys = txn.get(b'__keys__')
            if raw_keys is None:
                raise KeyError(f"FACED LMDB missing __keys__: {data_dir}")
            split_keys = pickle.loads(raw_keys)
            if mode not in split_keys:
                raise KeyError(f"FACED LMDB missing split {mode!r}: {list(split_keys)}")
            self.keys = list(split_keys[mode])
            raw_names = txn.get(b'__channel_names__') or txn.get(b'channel_names')
            self.channel_names = list(pickle.loads(raw_names)) if raw_names is not None else None

        manifest_path = os.path.join(self.data_dir, 'channel_manifest.json')
        if self.channel_names is None and os.path.isfile(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
            self.channel_names = list(
                payload.get('stored_channel_names', payload.get('channel_names', []))
            )
        if self.channel_names is None or len(self.channel_names) != EXPECTED_SHAPE[0]:
            raise ValueError(
                f"FACED requires a 32-channel manifest, got {self.channel_names!r}"
            )
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("FACED channel manifest contains duplicate names")

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            raw_pair = txn.get(key.encode())
            if raw_pair is None:
                raise KeyError(f"FACED LMDB key not found: {key!r}")
            pair = pickle.loads(raw_pair)
        data = np.asarray(pair['sample'])
        if tuple(data.shape) != EXPECTED_SHAPE:
            raise ValueError(
                f"FACED key {key!r} has shape {tuple(data.shape)}, expected {EXPECTED_SHAPE}"
            )
        if not np.isfinite(data).all():
            raise ValueError(f"FACED key {key!r} contains non-finite values")
        label = pair['label']
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        split_subjects = {
            mode: {_subject_id(key) for key in dataset.keys}
            for mode, dataset in (
                ('train', train_set), ('val', val_set), ('test', test_set)
            )
        }
        if (split_subjects['train'] & split_subjects['val'] or
                split_subjects['train'] & split_subjects['test'] or
                split_subjects['val'] & split_subjects['test']):
            raise ValueError(f"FACED subject overlap detected: {split_subjects}")
        print(
            f"[FACED] counts train={len(train_set)} val={len(val_set)} test={len(test_set)} "
            f"shape={EXPECTED_SHAPE} scale=/100"
        )
        print(f"[FACED] channels ({len(train_set.channel_names)}): {train_set.channel_names}")
        print(
            "[FACED] subject split sizes: "
            f"train={len(split_subjects['train'])} "
            f"val={len(split_subjects['val'])} test={len(split_subjects['test'])}; overlap=0"
        )
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader


def _subject_id(key):
    """Extract the stable subject/file identifier from FACED LMDB keys."""
    return str(key).split('.pkl-', 1)[0]

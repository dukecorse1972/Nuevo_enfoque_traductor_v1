from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GestureDataset(Dataset):
    def __init__(self, manifest_path: str | Path):
        self.items = np.load(manifest_path, allow_pickle=True)["items"]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        sample = self.items[idx]
        feats = np.load(sample["path"], allow_pickle=True)["features"].astype(np.float32)
        label = int(sample["label"])
        return torch.from_numpy(feats), torch.tensor(label, dtype=torch.long)

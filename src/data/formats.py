from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GestureSample:
    keypoints: np.ndarray  # [T, V, C]
    label: int
    source: str
    fps: int
    meta: dict[str, Any]

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            keypoints=self.keypoints.astype(np.float32),
            label=np.int64(self.label),
            source=self.source,
            fps=np.int32(self.fps),
            meta=np.array([self.meta], dtype=object),
        )


def load_sample(path: str | Path) -> GestureSample:
    data = np.load(path, allow_pickle=True)
    return GestureSample(
        keypoints=data["keypoints"],
        label=int(data["label"]),
        source=str(data["source"]),
        fps=int(data["fps"]),
        meta=data["meta"][0].item() if hasattr(data["meta"][0], "item") else data["meta"][0],
    )


def sample_to_dict(sample: GestureSample) -> dict[str, Any]:
    d = asdict(sample)
    d["keypoints"] = sample.keypoints.tolist()
    return d

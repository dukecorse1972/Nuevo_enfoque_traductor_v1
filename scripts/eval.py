#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.eval.metrics import compute_metrics
from src.models.tcn import TCNClassifier
from src.training.dataset import GestureDataset
from src.utils.io import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained TCN")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    manifest = args.manifest or Path(cfg["test_manifest"])
    ds = GestureDataset(manifest)
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 32), shuffle=False)

    x0, _ = ds[0]
    model = TCNClassifier(
        input_dim=x0.shape[1],
        num_classes=cfg["num_classes"],
        hidden_dim=cfg.get("hidden_dim", 128),
        dropout=cfg.get("dropout", 0.2),
        tiny=cfg.get("tiny", False),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    y_true, logits_all = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            y_true.append(y.numpy())
            logits_all.append(logits.numpy())

    y_true = np.concatenate(y_true)
    logits_all = np.concatenate(logits_all)
    m = compute_metrics(y_true, logits_all)

    out_dir = Path(cfg.get("eval_dir", "artifacts/eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({k: v for k, v in m.items() if k != "cm"}, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 6))
    plt.imshow(m["cm"], cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=160)
    print({k: v for k, v in m.items() if k != "cm"})


if __name__ == "__main__":
    main()

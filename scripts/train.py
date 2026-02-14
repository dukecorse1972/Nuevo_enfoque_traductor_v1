#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from src.models.tcn import TCNClassifier
from src.training.dataset import GestureDataset
from src.utils.io import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TCN for isolated sign recognition")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--fine-tune-from", type=Path, default=None)
    return p.parse_args()


def build_sampler(dataset: GestureDataset):
    labels = np.array([int(x["label"]) for x in dataset.items])
    classes, counts = np.unique(labels, return_counts=True)
    class_w = {c: 1.0 / n for c, n in zip(classes, counts)}
    sample_w = np.array([class_w[y] for y in labels], dtype=np.float64)
    return WeightedRandomSampler(sample_w, len(sample_w), replacement=True), class_w


def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * len(y)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    train_ds = GestureDataset(cfg["train_manifest"])
    val_ds = GestureDataset(cfg["val_manifest"])
    sampler, _ = build_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    first_x, _ = train_ds[0]
    model = TCNClassifier(
        input_dim=first_x.shape[1],
        num_classes=cfg["num_classes"],
        hidden_dim=cfg.get("hidden_dim", 128),
        dropout=cfg.get("dropout", 0.2),
        tiny=cfg.get("tiny", False),
    )
    if args.fine_tune_from and args.fine_tune_from.exists():
        model.load_state_dict(torch.load(args.fine_tune_from, map_location="cpu"), strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-3))
    writer = SummaryWriter(log_dir=cfg["log_dir"])

    best_val = float("inf")
    patience = cfg.get("early_stopping_patience", 8)
    no_improve = 0
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(y)
            n += len(y)

        train_loss = running / max(n, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        print(f"Epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        torch.save(model.state_dict(), ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    (ckpt_dir / "train_meta.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

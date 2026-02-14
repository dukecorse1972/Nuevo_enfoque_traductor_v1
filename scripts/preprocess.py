#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.preprocess.pipeline import PreprocessConfig, preprocess_sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess keypoints and build trainable dataset")
    p.add_argument("--input-dir", type=Path, required=True, help="Directory with raw sample .npz files")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target-len", type=int, default=64)
    p.add_argument("--max-gap", type=int, default=5)
    p.add_argument("--background-dir", type=Path, default=None, help="Optional raw background samples")
    p.add_argument("--background-label", type=int, default=0)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def process_one(path: Path, out_root: Path, cfg: PreprocessConfig, force_label: int | None = None):
    data = np.load(path, allow_pickle=True)
    feats, low_q = preprocess_sequence(data["keypoints"], cfg)
    label = int(force_label if force_label is not None else data["label"])
    rel = path.with_suffix(".npz").name
    out_path = out_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=feats, label=label, low_quality=np.int8(low_q), source=data["source"])
    return {"path": str(out_path), "label": label, "low_quality": bool(low_q)}


def split_items(items, train_ratio: float, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    items = [items[i] for i in idx]
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]


def save_manifest(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, items=np.array(items, dtype=object))


def main() -> None:
    args = parse_args()
    cfg = PreprocessConfig(max_gap=args.max_gap, target_len=args.target_len)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = sorted(args.input_dir.rglob("*.npz"))
    items = [process_one(p, args.output_dir / "processed", cfg) for p in tqdm(raw, desc="Main data")]

    if args.background_dir and args.background_dir.exists():
        bg_raw = sorted(args.background_dir.rglob("*.npz"))
        bg_items = [
            process_one(p, args.output_dir / "processed_bg", cfg, force_label=args.background_label)
            for p in tqdm(bg_raw, desc="Background")
        ]
        items.extend(bg_items)

    train, val, test = split_items(items, args.train_ratio, args.val_ratio, args.seed)
    save_manifest(args.output_dir / "manifests" / "train.npz", train)
    save_manifest(args.output_dir / "manifests" / "val.npz", val)
    save_manifest(args.output_dir / "manifests" / "test.npz", test)

    summary = {
        "total": len(items),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "n_low_quality": int(sum(x["low_quality"] for x in items)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.formats import GestureSample
from src.data.mediapipe_extractor import extract_video_keypoints


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract MediaPipe Holistic keypoints from videos")
    p.add_argument("--videos-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--label-map", type=Path, default=None, help="JSON mapping class_name->int")
    p.add_argument("--source", type=str, default="lse_sign")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_map = json.loads(args.label_map.read_text()) if args.label_map else {}
    failures = []

    videos = sorted([p for p in args.videos_dir.rglob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}])
    for vp in tqdm(videos, desc="Extracting"):
        cls_name = vp.parent.name
        label = int(label_map.get(cls_name, -1))
        try:
            result = extract_video_keypoints(vp)
            sample = GestureSample(
                keypoints=result.keypoints,
                label=label,
                source=args.source,
                fps=result.fps,
                meta={"video": str(vp), "invalid_frames": result.invalid_frames, "class_name": cls_name},
            )
            out_name = vp.relative_to(args.videos_dir).with_suffix(".npz")
            sample.save_npz(args.output_dir / out_name)
        except Exception as exc:  # noqa: BLE001
            failures.append({"video": str(vp), "error": str(exc)})

    (args.output_dir / "extract_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Done. videos={len(videos)} failures={len(failures)}")


if __name__ == "__main__":
    main()

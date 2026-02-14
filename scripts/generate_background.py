#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record idle/background clips from webcam")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=180)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--fps", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "background_idle.mp4"

    cap = cv2.VideoCapture(args.camera)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read webcam")

    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    max_frames = args.seconds * args.fps
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            continue
        writer.write(frame)
        n += 1
        cv2.putText(frame, f"Recording idle {n}/{max_frames}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Background recorder", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

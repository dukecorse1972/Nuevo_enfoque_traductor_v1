#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from src.models.tcn import TCNClassifier
from src.preprocess.pipeline import PreprocessConfig, preprocess_sequence
from src.utils.io import load_config

POSE_IDS = [11, 12, 13, 14, 15, 16]


def _lm_to_np(landmarks, ids: list[int] | None = None) -> np.ndarray:
    if landmarks is None:
        n = len(ids) if ids else 21
        return np.full((n, 3), np.nan, dtype=np.float32)
    if ids is None:
        pts = [[lm.x, lm.y, 1.0] for lm in landmarks.landmark]
    else:
        pts = [[landmarks.landmark[i].x, landmarks.landmark[i].y, 1.0] for i in ids]
    return np.asarray(pts, dtype=np.float32)


def motion_energy(seq: np.ndarray) -> float:
    if len(seq) < 2:
        return 0.0
    # wrists in reduced pose are idx 4 and 5
    w = seq[:, [4, 5], :2]
    d = np.diff(w, axis=0)
    return float(np.linalg.norm(d, axis=-1).mean())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time webcam demo for ISLR")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--camera", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    labels = cfg["labels"]

    model = TCNClassifier(
        input_dim=cfg["input_dim"],
        num_classes=len(labels),
        hidden_dim=cfg.get("hidden_dim", 128),
        dropout=cfg.get("dropout", 0.2),
        tiny=cfg.get("tiny", True),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    T = cfg.get("sequence_length", 64)
    energy_th = cfg.get("energy_threshold", 0.006)
    conf_th = cfg.get("confidence_threshold", 0.55)
    smooth_k = cfg.get("smoothing_k", 5)

    buffer = collections.deque(maxlen=T)
    smooth_logits = collections.deque(maxlen=smooth_k)
    control_mode = False
    recording = True

    cap = cv2.VideoCapture(args.camera)
    mp_holistic = mp.solutions.holistic
    draw = mp.solutions.drawing_utils
    preprocess_cfg = PreprocessConfig(target_len=T, max_gap=5)

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5) as holistic:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Warning: webcam frame read failed")
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)

            pose = _lm_to_np(res.pose_landmarks, ids=POSE_IDS)
            lh = _lm_to_np(res.left_hand_landmarks)
            rh = _lm_to_np(res.right_hand_landmarks)
            kpts = np.concatenate([pose, lh, rh], axis=0)

            if recording:
                buffer.append(kpts)

            state = "Idle"
            text = "Idle"
            conf = 0.0

            if len(buffer) >= max(8, T // 2):
                seq = np.stack(buffer, axis=0)
                energy = motion_energy(seq)
                if energy >= energy_th:
                    state = "Detecting"
                    feats, _ = preprocess_sequence(seq, preprocess_cfg)
                    x = torch.from_numpy(feats[None, ...])
                    logits = model(x)
                    smooth_logits.append(logits.detach().numpy()[0])
                    avg = np.mean(np.stack(smooth_logits, axis=0), axis=0)
                    pred = int(avg.argmax())
                    conf = float(torch.softmax(torch.tensor(avg), dim=0)[pred])
                    text = labels[pred] if conf >= conf_th else "background"

            if res.pose_landmarks:
                draw.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if res.left_hand_landmarks:
                draw.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if res.right_hand_landmarks:
                draw.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            cv2.putText(frame, f"State: {state}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Pred: {text}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {conf:.2f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Control mode: {control_mode}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow("ISLR Demo (LSE)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                control_mode = not control_mode
            if control_mode and key == ord("s"):
                recording = not recording

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

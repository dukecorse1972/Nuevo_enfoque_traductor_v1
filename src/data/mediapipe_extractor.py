from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

POSE_IDS = [11, 12, 13, 14, 15, 16]


@dataclass
class ExtractResult:
    keypoints: np.ndarray
    fps: int
    invalid_frames: int


def _landmarks_to_np(landmarks, ids: list[int] | None = None) -> np.ndarray:
    if landmarks is None:
        if ids is None:
            return np.full((0, 3), np.nan, dtype=np.float32)
        return np.full((len(ids), 3), np.nan, dtype=np.float32)
    if ids is None:
        pts = [[lm.x, lm.y, lm.visibility if hasattr(lm, "visibility") else 1.0] for lm in landmarks.landmark]
    else:
        pts = []
        for idx in ids:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                pts.append([lm.x, lm.y, lm.visibility if hasattr(lm, "visibility") else 1.0])
            else:
                pts.append([np.nan, np.nan, np.nan])
    return np.asarray(pts, dtype=np.float32)


def extract_video_keypoints(video_path: str | Path, model_complexity: int = 1) -> ExtractResult:
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
    frames: list[np.ndarray] = []
    invalid = 0

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            pose = _landmarks_to_np(result.pose_landmarks, ids=POSE_IDS)
            left_hand = _landmarks_to_np(result.left_hand_landmarks)
            right_hand = _landmarks_to_np(result.right_hand_landmarks)

            if pose.size == 0 or left_hand.size == 0 or right_hand.size == 0:
                invalid += 1
            if left_hand.shape[0] == 0:
                left_hand = np.full((21, 3), np.nan, dtype=np.float32)
            if right_hand.shape[0] == 0:
                right_hand = np.full((21, 3), np.nan, dtype=np.float32)

            frame_kpts = np.concatenate([pose, left_hand, right_hand], axis=0)
            frames.append(frame_kpts)

    cap.release()
    return ExtractResult(keypoints=np.asarray(frames, dtype=np.float32), fps=fps, invalid_frames=invalid)

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_IDS = [11, 12, 13, 14, 15, 16]


@dataclass
class PreprocessConfig:
    max_gap: int = 5
    target_len: int = 64
    include_face: bool = False


def _interp_1d(arr: np.ndarray, max_gap: int) -> np.ndarray:
    out = arr.copy()
    valid = ~np.isnan(out)
    if valid.sum() < 2:
        return out
    idx = np.arange(len(out))
    interp = np.interp(idx, idx[valid], out[valid])
    out[~valid] = interp[~valid]

    nan_runs = np.diff(np.where(np.concatenate(([True], valid, [True])))[0]) - 1
    starts = np.where(np.diff(np.concatenate(([True], valid, [True]))))[0][::2]
    for start, length in zip(starts, nan_runs):
        if length > max_gap:
            out[start : start + length] = np.nan
    return out


def interpolate_missing(kpts: np.ndarray, max_gap: int = 5) -> np.ndarray:
    out = kpts.copy()
    t, v, c = out.shape
    for vi in range(v):
        for ci in range(c):
            out[:, vi, ci] = _interp_1d(out[:, vi, ci], max_gap=max_gap)
    return out


def normalize_by_shoulders(kpts: np.ndarray) -> np.ndarray:
    out = kpts.copy()
    shoulders = out[:, :2, :2]
    chest = np.nanmean(shoulders, axis=1, keepdims=True)
    shoulder_dist = np.linalg.norm(shoulders[:, 0, :] - shoulders[:, 1, :], axis=1, keepdims=True)
    shoulder_dist = np.clip(shoulder_dist, 1e-3, None)
    out[:, :, :2] = (out[:, :, :2] - chest) / shoulder_dist[:, None, :]
    return out


def velocity_features(kpts_xy: np.ndarray) -> np.ndarray:
    vel = np.diff(kpts_xy, axis=0, prepend=kpts_xy[0:1])
    return np.concatenate([kpts_xy, vel], axis=-1)


def temporal_resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    t = arr.shape[0]
    if t == target_len:
        return arr
    idx = np.linspace(0, t - 1, target_len).astype(np.float32)
    left = np.floor(idx).astype(int)
    right = np.clip(left + 1, 0, t - 1)
    alpha = (idx - left)[:, None, None]
    return (1 - alpha) * arr[left] + alpha * arr[right]


def flatten_features(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(arr.shape[0], -1)


def preprocess_sequence(kpts: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, bool]:
    clean = interpolate_missing(kpts, max_gap=cfg.max_gap)
    low_quality = np.isnan(clean).mean() > 0.15
    if np.isnan(clean).any():
        clean = np.nan_to_num(clean, nan=0.0)

    clean = normalize_by_shoulders(clean)
    feats = velocity_features(clean[:, :, :2])
    feats = temporal_resample(feats, cfg.target_len)
    feats = flatten_features(feats)
    return feats.astype(np.float32), low_quality

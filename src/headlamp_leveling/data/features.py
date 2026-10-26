"""Feature engineering shared by training, evaluation and deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .. import CHANNELS

FEATURES = CHANNELS + ("a_wheel",)
IDX = {name: i for i, name in enumerate(FEATURES)}


def wheel_accel(v: np.ndarray, fs_hz: float, smooth_s: float = 0.5) -> np.ndarray:
    """Longitudinal acceleration from wheel speed: smoothed central difference."""
    n = max(1, int(round(smooth_s * fs_hz)))
    if n > 1:
        k = np.ones(n) / n
        v = np.convolve(np.pad(v, (n, n), mode="edge"), k, mode="same")[n:-n]
    return np.gradient(v, 1.0 / fs_hz)


def build_features(inputs: np.ndarray, fs_hz: float) -> np.ndarray:
    """(T, 7) raw channels -> (T, 8) model features (adds wheel-speed acceleration)."""
    a_w = wheel_accel(inputs[:, IDX["v_wheel"]].astype(np.float64), fs_hz)
    return np.concatenate([inputs, a_w[:, None]], axis=1).astype(np.float32)


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, feats: np.ndarray) -> "Normalizer":
        mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        # Keep the accelerometer x/z on a common scale so their ratio (= pitch) survives normalisation.
        acc_scale = float(np.mean(std[[IDX["ax"], IDX["az"]]]))
        std[IDX["ax"]] = std[IDX["az"]] = acc_scale
        std[IDX["a_wheel"]] = acc_scale
        return cls(mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32))

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        return (feats - self.mean) / self.std

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, list]) -> "Normalizer":
        return cls(np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32))

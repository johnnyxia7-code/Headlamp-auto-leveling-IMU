"""Sliding-window PyTorch dataset over a TripSet."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import IDX, Normalizer, build_features
from .tripset import TripSet


class WindowDataset(Dataset):
    """Yields (x, y, excitation).

    x           (C, W) normalised features of one window
    y           (2,) [load pitch at window end, road grade at window end] in degrees
    excitation  std of wheel-speed acceleration in the window (m/s^2); low values
                mean the window carries little information about the load pitch
    """

    def __init__(self, trips: TripSet, window: int, stride: int,
                 normalizer: Optional[Normalizer] = None, augment: bool = False, seed: int = 0):
        self.window = window
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.feats = np.concatenate([
            build_features(trips.trip(i)["inputs"], trips.fs_hz) for i in range(len(trips))
        ])
        self.normalizer = normalizer or Normalizer.fit(self.feats)
        self.load_pitch = trips.labels["load_pitch_deg"]
        self.grade = trips.labels["road_grade_deg"]

        ends, ids = [], []
        for i in range(len(trips)):
            a, b = int(trips.offsets[i]), int(trips.offsets[i + 1])
            if b - a >= window:
                e = np.arange(a + window - 1, b, stride)
                ends.append(e)
                ids.append(np.full(e.size, i))
        self.ends = np.concatenate(ends) if ends else np.zeros(0, np.int64)
        self.trip_ids = np.concatenate(ids) if ids else np.zeros(0, np.int64)
        keep = ~np.isnan(self.load_pitch[self.ends])
        self.ends, self.trip_ids = self.ends[keep], self.trip_ids[keep]

        a_w = self.feats[:, IDX["a_wheel"]]
        self.excitation = np.array([a_w[e - window + 1:e + 1].std() for e in self.ends], np.float32)

    def __len__(self) -> int:
        return len(self.ends)

    def raw_window(self, k: int) -> np.ndarray:
        e = int(self.ends[k])
        return self.feats[e - self.window + 1:e + 1]

    def __getitem__(self, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e = int(self.ends[k])
        x = self.raw_window(k).copy()
        if self.augment:
            # Extra sensor variation on top of the simulator: accel bias, gyro bias, wheel-speed scale.
            x[:, :3] += self.rng.normal(0.0, 0.08, size=3).astype(np.float32)
            x[:, 3:6] += self.rng.normal(0.0, 0.3, size=3).astype(np.float32)
            s = np.float32(1.0 + self.rng.normal(0.0, 0.01))
            x[:, IDX["v_wheel"]] *= s
            x[:, IDX["a_wheel"]] *= s
        x = self.normalizer(x).T
        y = np.array([self.load_pitch[e], np.nan_to_num(self.grade[e])], np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y), torch.tensor(self.excitation[k])

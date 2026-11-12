"""Storage of variable-length trips (simulated or recorded) in a single .npz file.

All trips are concatenated along time; ``offsets[i]:offsets[i+1]`` selects trip i.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

LABEL_KEYS = ("load_pitch_deg", "road_grade_deg", "body_pitch_deg", "speed_ms", "accel_ms2")


@dataclass
class TripSet:
    inputs: np.ndarray                         # (sum_T, C) float32, raw sensor channels
    labels: Dict[str, np.ndarray]              # key -> (sum_T,) float32 (NaN when unknown)
    offsets: np.ndarray                        # (N + 1,) int64
    fs_hz: float
    meta: List[Dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def trip(self, i: int) -> Dict[str, np.ndarray]:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        out = {"inputs": self.inputs[a:b]}
        out.update({k: v[a:b] for k, v in self.labels.items()})
        return out

    @classmethod
    def from_trips(cls, trips: Sequence[Dict[str, Any]], fs_hz: float) -> "TripSet":
        lengths = [t["inputs"].shape[0] for t in trips]
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        inputs = np.concatenate([t["inputs"] for t in trips]).astype(np.float32)
        labels = {}
        for k in LABEL_KEYS:
            labels[k] = np.concatenate([
                np.asarray(t[k], np.float32) if k in t else np.full(t["inputs"].shape[0], np.nan, np.float32)
                for t in trips
            ])
        return cls(inputs, labels, offsets, fs_hz, [t.get("meta", {}) for t in trips])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, inputs=self.inputs, offsets=self.offsets, fs_hz=np.float64(self.fs_hz),
            meta=np.array(json.dumps(self.meta)), **{f"label_{k}": v for k, v in self.labels.items()},
        )

    @classmethod
    def load(cls, path: str | Path) -> "TripSet":
        with np.load(path, allow_pickle=False) as z:
            labels = {k[len("label_"):]: z[k] for k in z.files if k.startswith("label_")}
            return cls(z["inputs"], labels, z["offsets"], float(z["fs_hz"]), json.loads(str(z["meta"])))

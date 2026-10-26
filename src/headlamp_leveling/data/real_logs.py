"""Import recorded vehicle logs (CSV) as a TripSet for fine-tuning / validation.

Expected CSV columns (one file = one trip, any sample rate >= model rate):

    t                 time in seconds
    ax, ay, az        accelerometer, m/s^2, ISO 8855 vehicle frame (x fwd, y left, z up)
    gx, gy, gz        gyro, deg/s, right-handed about the same axes
    v_wheel           vehicle speed from wheel speeds, m/s

plus ONE of the following label sources:

    load_pitch_deg                 static load pitch directly (e.g. from a measurement rig), or
    h_front_mm, h_rear_mm          ride-height sensors (front / rear axle, larger = higher body)

Ride heights are converted to pitch relative to the reference state
(unladen + driver, given by ``h_ref_front_mm`` / ``h_ref_rear_mm``) and
low-pass filtered so the label reflects the static load pitch rather than
the dynamic squat / dive.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from .. import CHANNELS
from .tripset import TripSet


def _read_csv(path: Path) -> Dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: empty log")
    return {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in rows[0]}


def _resample(t: np.ndarray, x: np.ndarray, fs_hz: float) -> np.ndarray:
    t_new = np.arange(t[0], t[-1], 1.0 / fs_hz)
    src_fs = 1.0 / np.median(np.diff(t))
    n = int(round(src_fs / fs_hz))
    if n > 1:  # boxcar anti-alias before interpolation
        k = np.ones(n) / n
        x = np.convolve(np.pad(x, (n, n), mode="edge"), k, mode="same")[n:-n]
    return np.interp(t_new, t, x)


def _moving_median(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, n), axis=1)


def ride_height_to_pitch_deg(h_front_mm: np.ndarray, h_rear_mm: np.ndarray, wheelbase_m: float,
                             h_ref_front_mm: float, h_ref_rear_mm: float) -> np.ndarray:
    """Body pitch (nose-up positive) relative to the reference ride heights."""
    dh_f = (h_front_mm - h_ref_front_mm) * 1e-3
    dh_r = (h_rear_mm - h_ref_rear_mm) * 1e-3
    return np.rad2deg(np.arctan2(dh_f - dh_r, wheelbase_m))


def load_log(path: str | Path, fs_hz: float, wheelbase_m: Optional[float] = None,
             h_ref_front_mm: Optional[float] = None, h_ref_rear_mm: Optional[float] = None,
             label_smooth_s: float = 20.0) -> Dict[str, np.ndarray]:
    path = Path(path)
    cols = _read_csv(path)
    missing = [c for c in ("t",) + CHANNELS if c not in cols]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}")
    t = cols["t"]
    inputs = np.stack([_resample(t, cols[c], fs_hz) for c in CHANNELS], axis=1).astype(np.float32)
    trip = {"inputs": inputs, "meta": {"source": str(path)}}

    if "load_pitch_deg" in cols:
        trip["load_pitch_deg"] = _resample(t, cols["load_pitch_deg"], fs_hz)
    elif "h_front_mm" in cols and "h_rear_mm" in cols:
        if wheelbase_m is None or h_ref_front_mm is None or h_ref_rear_mm is None:
            raise ValueError("ride-height labels need wheelbase_m, h_ref_front_mm and h_ref_rear_mm")
        pitch = ride_height_to_pitch_deg(_resample(t, cols["h_front_mm"], fs_hz),
                                         _resample(t, cols["h_rear_mm"], fs_hz),
                                         wheelbase_m, h_ref_front_mm, h_ref_rear_mm)
        trip["body_pitch_deg"] = pitch
        trip["load_pitch_deg"] = _moving_median(pitch, int(label_smooth_s * fs_hz))
    if "road_grade_deg" in cols:
        trip["road_grade_deg"] = _resample(t, cols["road_grade_deg"], fs_hz)
    trip["speed_ms"] = inputs[:, CHANNELS.index("v_wheel")]
    return trip


def import_logs(paths: Iterable[str | Path], fs_hz: float, **kwargs) -> TripSet:
    return TripSet.from_trips([load_log(p, fs_hz, **kwargs) for p in paths], fs_hz)

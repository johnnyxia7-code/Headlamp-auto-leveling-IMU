"""Sensor fault injection for robustness testing.

Faults are applied to the full-rate signals of a simulated trip
(``simulate_trip(..., raw=True)["raw"]``). A missing sample is NaN, which is
how a receive timeout / invalid-flag reaches the software.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from .. import CHANNELS

IMU = ("ax", "ay", "az", "gx", "gy", "gz")
_I = {c: i for i, c in enumerate(CHANNELS)}


@dataclass
class FaultWindow:
    channel: str
    kind: str
    t0: float
    t1: float


def _span(fs: float, t0: float, t1: float, n: int) -> slice:
    return slice(max(0, int(t0 * fs)), min(n, int(t1 * fs)))


def _moving_start(raw: Dict[str, np.ndarray], t_min: float, dur: float, speed: float = 8.0) -> float:
    """First time >= t_min from which the car keeps driving faster than ``speed`` for ``dur``."""
    fs, v = raw["fs_hz"], raw["speed_ms"]
    k, n, need = int(t_min * fs), v.size, int(dur * fs)
    step = int(fs)
    while k + need < n:
        if np.all(v[k:k + need:step] > speed):
            return k / fs
        k += step
    return t_min


def none(raw, rng) -> List[FaultWindow]:
    return []


def imu_dropouts(raw, rng) -> List[FaultWindow]:
    """Short bus dropouts: 25 gaps of 30-300 ms on all IMU channels."""
    fs, n = raw["fs_hz"], raw["inputs"].shape[0]
    out = []
    for t0 in rng.uniform(60.0, n / fs - 5.0, size=25):
        d = rng.uniform(0.03, 0.3)
        raw["inputs"][_span(fs, t0, t0 + d, n), :6] = np.nan
        out.append(FaultWindow("imu", "dropout", t0, t0 + d))
    return out


def spikes(raw, rng) -> List[FaultWindow]:
    """EMC-like glitches: 0.5 % of samples of ax, az, gy replaced by large values."""
    n = raw["inputs"].shape[0]
    for c, amp in (("ax", 15.0), ("az", 15.0), ("gy", 150.0)):
        k = rng.choice(n, size=int(0.005 * n), replace=False)
        raw["inputs"][k, _I[c]] += rng.choice([-1.0, 1.0], size=k.size) * rng.uniform(0.5, 1.0, size=k.size) * amp
    return [FaultWindow("ax,az,gy", "spikes", 0.0, n / raw["fs_hz"])]


def wheel_dropout(raw, rng) -> List[FaultWindow]:
    """CAN timeout of the wheel speed: three gaps of 4 s while driving."""
    fs, n = raw["fs_hz"], raw["inputs"].shape[0]
    out = []
    for t_min in (150.0, 300.0, 450.0):
        t0 = _moving_start(raw, t_min, 6.0)
        raw["inputs"][_span(fs, t0, t0 + 4.0, n), _I["v_wheel"]] = np.nan
        out.append(FaultWindow("v_wheel", "dropout", t0, t0 + 4.0))
    return out


def wheel_zero(raw, rng) -> List[FaultWindow]:
    """Wheel speed reads 0 for 20 s while the car is driving."""
    fs, n = raw["fs_hz"], raw["inputs"].shape[0]
    t0 = _moving_start(raw, 200.0, 22.0)
    raw["inputs"][_span(fs, t0, t0 + 20.0, n), _I["v_wheel"]] = 0.0
    return [FaultWindow("v_wheel", "stuck_zero", t0, t0 + 20.0)]


def gyro_stuck(raw, rng) -> List[FaultWindow]:
    """Pitch gyro frozen at its last value for 30 s."""
    fs, n = raw["fs_hz"], raw["inputs"].shape[0]
    s = _span(fs, 200.0, 230.0, n)
    raw["inputs"][s, _I["gy"]] = raw["inputs"][s.start, _I["gy"]]
    return [FaultWindow("gy", "stuck", 200.0, 230.0)]


def gyro_offset(raw, rng) -> List[FaultWindow]:
    """Pitch gyro offset of +20 deg/s from t = 200 s to the end of the trip."""
    fs, n = raw["fs_hz"], raw["inputs"].shape[0]
    raw["inputs"][_span(fs, 200.0, 1e9, n), _I["gy"]] += 20.0
    return [FaultWindow("gy", "offset", 200.0, n / fs)]


def accel_z_stuck(raw, rng) -> List[FaultWindow]:
    """Vertical accelerometer frozen for 30 s."""
    fs, n = raw["fs_hz"], raw["inputs"].shape[0]
    s = _span(fs, 200.0, 230.0, n)
    raw["inputs"][s, _I["az"]] = raw["inputs"][s.start, _I["az"]]
    return [FaultWindow("az", "stuck", 200.0, 230.0)]


def accel_z_offset(raw, rng) -> List[FaultWindow]:
    """Vertical accelerometer offset of +2 m/s^2 for 60 s."""
    fs, n = raw["fs_hz"], raw["inputs"].shape[0]
    raw["inputs"][_span(fs, 200.0, 260.0, n), _I["az"]] += 2.0
    return [FaultWindow("az", "offset", 200.0, 260.0)]


def accel_saturated(raw, rng) -> List[FaultWindow]:
    """Longitudinal accelerometer stuck at full scale for 20 s."""
    fs, n = raw["fs_hz"], raw["inputs"].shape[0]
    raw["inputs"][_span(fs, 200.0, 220.0, n), _I["ax"]] = 19.6
    return [FaultWindow("ax", "saturated", 200.0, 220.0)]


def imu_outage(raw, rng) -> List[FaultWindow]:
    """Complete loss of the IMU for 15 s, then it returns."""
    fs, n = raw["fs_hz"], raw["inputs"].shape[0]
    raw["inputs"][_span(fs, 200.0, 215.0, n), :6] = np.nan
    return [FaultWindow("imu", "outage", 200.0, 215.0)]


def camera_garbage(raw, rng) -> List[FaultWindow]:
    """Camera reports plausible-looking but wrong pitch (+1.2 deg bias, noisy) for 90 s."""
    fs, n = raw["fs_hz"], raw["inputs"].shape[0]
    s = _span(fs, 200.0, 290.0, n)
    cam = raw["camera_pitch_deg"]
    frames = np.arange(s.start, s.stop, max(1, int(fs / 10.0)))
    cam[frames] = raw["body_pitch_deg"][frames] + 1.2 + rng.normal(0.0, 0.4, size=frames.size)
    return [FaultWindow("camera", "garbage", 200.0, 290.0)]


def camera_frozen(raw, rng) -> List[FaultWindow]:
    """Camera repeats the same frame result for 90 s."""
    fs, n = raw["fs_hz"], raw["inputs"].shape[0]
    s = _span(fs, 200.0, 290.0, n)
    frames = np.arange(s.start, s.stop, max(1, int(fs / 10.0)))
    raw["camera_pitch_deg"][frames] = 1.5
    return [FaultWindow("camera", "frozen", 200.0, 290.0)]


def apply(raw: Dict[str, np.ndarray], channel: str, kind: str, t0: float, t1: float,
          value: float = 0.0, seed: int = 0) -> FaultWindow:
    """Apply one fault in place to ``raw`` (use on a copy; see :func:`copy_raw`).

    channel  one of the sensor channels, "imu" (all six IMU axes) or "camera"
    kind     "missing"  no data (NaN)
             "frozen"   repeats the value it had at t0
             "offset"   adds ``value``
             "value"    reads ``value`` constantly
             "glitch"   a fraction ``value`` of the samples gets a large random error
             "wrong"    camera only: plausible-looking pitch with a ``value`` deg bias and extra noise
    """
    fs, n = raw["fs_hz"], raw["inputs"].shape[0]
    s = _span(fs, t0, t1, n)
    rng = np.random.default_rng(seed)
    if channel == "camera":
        cam = raw["camera_pitch_deg"]
        frames = np.arange(s.start, s.stop, max(1, int(fs / 10.0)))
        if kind == "missing":
            cam[s] = np.nan
        elif kind == "frozen":
            cam[frames] = value
        elif kind == "wrong":
            cam[frames] = raw["body_pitch_deg"][frames] + value + rng.normal(0.0, 0.4, size=frames.size)
        else:
            raise ValueError(f"fault kind {kind!r} is not defined for the camera")
        return FaultWindow(channel, kind, t0, t1)
    cols = list(range(6)) if channel == "imu" else [_I[channel]]
    x = raw["inputs"]
    for c in cols:
        if kind == "missing":
            x[s, c] = np.nan
        elif kind == "frozen":
            x[s, c] = x[s.start, c]
        elif kind == "offset":
            x[s, c] += value
        elif kind == "value":
            x[s, c] = value
        elif kind == "glitch":
            k = s.start + rng.choice(s.stop - s.start, size=int(value * (s.stop - s.start)), replace=False)
            amp = 150.0 if CHANNELS[c].startswith("g") else 15.0
            x[k, c] += rng.choice([-1.0, 1.0], size=k.size) * rng.uniform(0.5, 1.0, size=k.size) * amp
        else:
            raise ValueError(f"unknown fault kind {kind!r}")
    return FaultWindow(channel, kind, t0, t1)


def copy_raw(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = dict(raw)
    out["inputs"] = raw["inputs"].astype(np.float64).copy()
    out["camera_pitch_deg"] = raw["camera_pitch_deg"].astype(np.float64).copy()
    return out


SCENARIOS: Dict[str, Callable] = {
    "none": none,
    "imu_dropouts": imu_dropouts,
    "spikes": spikes,
    "wheel_dropout": wheel_dropout,
    "wheel_zero": wheel_zero,
    "gyro_stuck": gyro_stuck,
    "gyro_offset": gyro_offset,
    "accel_z_stuck": accel_z_stuck,
    "accel_z_offset": accel_z_offset,
    "accel_saturated": accel_saturated,
    "imu_outage": imu_outage,
    "camera_garbage": camera_garbage,
    "camera_frozen": camera_frozen,
}


def inject(raw: Dict[str, np.ndarray], scenario: str, seed: int = 0) -> Tuple[Dict[str, np.ndarray], List[FaultWindow]]:
    """Return a copy of ``raw`` with the scenario's faults applied, plus the fault windows."""
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown fault scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    out = dict(raw)
    out["inputs"] = raw["inputs"].astype(np.float64).copy()
    out["camera_pitch_deg"] = raw["camera_pitch_deg"].astype(np.float64).copy()
    windows = SCENARIOS[scenario](out, np.random.default_rng(seed))
    return out, windows

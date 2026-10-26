"""Front camera as an auxiliary pitch sensor.

A forward camera (lane / road-plane detection) can estimate its own pitch
relative to the road surface ahead: the vanishing point of the lane markings
lies on the horizon of the road plane at the look-ahead distance. With the
camera rigidly mounted on the body this measures

    theta_cam = theta - (gamma(s + d_la) - gamma(s)) + b_cam + noise

i.e. body pitch relative to the road, corrupted by the grade change between
the vehicle and the look-ahead point, a mounting bias and detection noise.
It is only available when lane markings are detected (daylight, marked
roads, above a minimum speed) and occasionally produces outliers.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..config import CameraConfig


def camera_availability(rng: np.random.Generator, cfg: CameraConfig, n: int, fs_hz: float,
                        speed: np.ndarray) -> np.ndarray:
    """Two-state Markov detection availability (mean on / off durations), gated by speed."""
    avail = np.zeros(n, bool)
    i, on = 0, bool(rng.random() < cfg.mean_on_s / (cfg.mean_on_s + cfg.mean_off_s))
    while i < n:
        dur = int(rng.exponential(cfg.mean_on_s if on else cfg.mean_off_s) * fs_hz) + 1
        avail[i:i + dur] = on
        i += dur
        on = not on
    return avail & (speed >= cfg.min_speed_ms)


def camera_pitch_measurements(rng: np.random.Generator, cfg: CameraConfig, fs_hz: float,
                              theta: np.ndarray, gamma: np.ndarray, gamma_ahead: np.ndarray,
                              speed: np.ndarray) -> Tuple[np.ndarray, float]:
    """Camera pitch frames (deg) on the simulation time base; NaN where no frame / no detection.

    Returns (measurements, mounting bias in deg).
    """
    n = theta.size
    bias = float(rng.normal(0.0, cfg.bias_deg))
    out = np.full(n, np.nan)
    if not cfg.enabled:
        return out, bias
    step = max(1, int(round(fs_hz / cfg.rate_hz)))
    frames = np.arange(0, n, step)
    avail = camera_availability(rng, cfg, n, fs_hz, speed)[frames]
    true = np.rad2deg(theta[frames] - (gamma_ahead[frames] - gamma[frames]))
    meas = true + bias + rng.normal(0.0, cfg.noise_deg, size=frames.size)
    outlier = rng.random(frames.size) < cfg.outlier_prob
    meas[outlier] += rng.normal(0.0, cfg.outlier_deg, size=int(outlier.sum()))
    out[frames[avail]] = meas[avail]
    return out, bias

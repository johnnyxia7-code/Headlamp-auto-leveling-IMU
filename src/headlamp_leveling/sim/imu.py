"""6-axis MEMS IMU and wheel-speed sensor error models."""

from __future__ import annotations

import numpy as np

from .. import G
from ..config import ImuConfig


def _rot_small(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix for small mounting misalignment (body -> sensor)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])   # nose-up positive
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _quantize(x: np.ndarray, lsb: float) -> np.ndarray:
    return np.round(x / lsb) * lsb if lsb > 0 else x


def imu_measurements(rng: np.random.Generator, cfg: ImuConfig, fs_hz: float,
                     f_body: np.ndarray, w_body: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Corrupt true specific force (m/s^2) and angular rate (deg/s), shape (n, 3) each.

    Returns (n, 6) [ax, ay, az, gx, gy, gz] as the sensor would report them.
    Errors: per-trip bias + bias random walk, scale factor, residual
    misalignment, white noise, speed-dependent body vibration and quantization.
    """
    n = f_body.shape[0]
    dt = 1.0 / fs_hz
    bw = fs_hz / 2.0

    mis = np.deg2rad(rng.normal(0.0, cfg.misalign_deg, size=3))
    R = _rot_small(*mis)
    f = f_body @ R.T
    w = w_body @ R.T

    bias0 = rng.normal(0.0, cfg.accel_bias_mg * 1e-3 * G, size=3)
    walk = np.cumsum(rng.normal(0.0, cfg.accel_bias_walk_mg_s * 1e-3 * G * np.sqrt(dt), size=(n, 3)), axis=0)
    scale = 1.0 + rng.normal(0.0, cfg.accel_scale_err, size=3)
    noise = rng.normal(0.0, cfg.accel_noise_ug_rthz * 1e-6 * G * np.sqrt(bw), size=(n, 3))
    vib_gain = cfg.vibration_ms2 * np.clip(speed / 27.8, 0.0, 1.5)[:, None]
    vib = rng.normal(0.0, 1.0, size=(n, 3)) * vib_gain * np.array([0.4, 0.6, 1.0])
    acc = f * scale + bias0 + walk + noise + vib

    gbias = rng.normal(0.0, cfg.gyro_bias_dps, size=3)
    gnoise = rng.normal(0.0, cfg.gyro_noise_dps_rthz * np.sqrt(bw), size=(n, 3))
    gyr = w + gbias + gnoise

    acc = _quantize(acc, cfg.accel_lsb_mg * 1e-3 * G)
    gyr = _quantize(gyr, cfg.gyro_lsb_dps)
    return np.concatenate([acc, gyr], axis=1)


def wheel_speed_measurement(rng: np.random.Generator, cfg: ImuConfig, speed: np.ndarray,
                            accel: np.ndarray) -> np.ndarray:
    """Vehicle speed as derived from wheel-speed sensors (m/s).

    Includes a per-trip rolling-radius error, longitudinal slip under
    traction / braking and white noise. Reads exactly 0 at standstill.
    """
    radius_err = rng.normal(0.0, cfg.tyre_radius_err)
    v = speed * (1.0 + radius_err) * (1.0 + cfg.slip_per_ms2 * accel)
    v = v + rng.normal(0.0, cfg.wheel_speed_noise_ms, size=v.shape)
    v[speed <= 0.0] = 0.0
    return np.maximum(v, 0.0)

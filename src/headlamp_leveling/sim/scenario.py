"""End-to-end trip simulation: vehicle + load + road + driver -> IMU log with labels.

Kinematics (pitch only, body frame x forward / z up, angles nose-up positive)::

    gamma  road grade angle (uphill positive)
    theta  body pitch relative to the road = static load pitch + dynamic pitch
    phi    body pitch relative to the horizon = gamma + theta
    a      acceleration along the road

    f_x =  a cos(theta) + g sin(phi)
    f_z = -a sin(theta) + g cos(phi) + heave acceleration

At constant grade the change of the specific-force vector caused by a change
of ``a`` points along the road, so the angle of (df_x, df_z) in the body frame
reveals ``theta`` independently of the grade and of the accelerometer bias.
That observability is what both the physics baseline and the learned models
exploit.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple

import numpy as np

from .. import G
from ..config import Config
from .driving import drive_cycle
from .imu import imu_measurements, wheel_speed_measurement
from .road import curvature_profile, grade_profile, iso8608_profile
from .vehicle import sample_load_case, sample_vehicle, static_load_pitch_rad


def second_order(u: np.ndarray, wn: float, zeta: float, dt: float,
                 base_excitation: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate y'' = wn^2 (u - y) + 2 zeta wn (c u' - y'), c = 1 if base excited else 0.

    Returns (y, y''). Semi-implicit Euler, starts at rest at y = u[0].
    """
    n = u.size
    y = np.empty(n)
    ydd = np.empty(n)
    du = np.gradient(u, dt) if base_excitation else np.zeros(n)
    yi, vi = float(u[0]), 0.0
    w2, c = wn * wn, 2.0 * zeta * wn
    for i in range(n):
        acc = w2 * (u[i] - yi) + c * (du[i] - vi)
        vi += acc * dt
        yi += vi * dt
        y[i], ydd[i] = yi, acc
    return y, ydd


def _pick(rng: np.random.Generator, classes) -> Tuple[str, float]:
    names, vals, probs = zip(*classes)
    k = rng.choice(len(names), p=np.asarray(probs) / np.sum(probs))
    return names[k], float(vals[k])


def _decimate(x: np.ndarray, q: int) -> np.ndarray:
    """Boxcar anti-alias + downsample along axis 0."""
    n = (x.shape[0] // q) * q
    return x[:n].reshape(n // q, q, *x.shape[1:]).mean(axis=1)


def simulate_trip(cfg: Config, seed: int | np.random.SeedSequence) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    sc = cfg.sim
    fs, dt = sc.fs_hz, 1.0 / sc.fs_hz
    n = int(sc.trip_duration_s * fs)

    veh = sample_vehicle(rng, cfg.vehicle)
    L = veh.wheelbase_m
    load_a = sample_load_case(rng, veh, cfg.vehicle)
    cyc = drive_cycle(rng, n, dt, sc.max_speed_kmh)

    # Optional load change during one of the stops (passengers get in/out).
    theta_static = np.full(n, static_load_pitch_rad(veh, load_a))
    load_b = None
    later_stops = [s for s in cyc.stops[1:] if s[1] - s[0] > int(2.0 * fs)]
    if later_stops and rng.random() < sc.load_change_prob:
        s0, s1 = later_stops[int(rng.integers(len(later_stops)))]
        load_b = sample_load_case(rng, veh, cfg.vehicle)
        theta_static[(s0 + s1) // 2:] = static_load_pitch_rad(veh, load_b)

    # Road.
    length = cyc.distance[-1] + L + 100.0
    grade_cls, grade_max = _pick(rng, sc.grade_classes)
    rough_cls, gd_n0 = _pick(rng, sc.roughness_classes)
    ds_macro, ds_rough = 1.0, 0.05
    s_macro = np.arange(int(np.ceil(length / ds_macro)) + 1) * ds_macro
    grade_s = grade_profile(rng, length, ds_macro, grade_max)
    kappa_s = curvature_profile(rng, length, ds_macro)
    bank = np.arctan(rng.uniform(-0.025, 0.025))
    z_s = iso8608_profile(rng, length, ds_rough, gd_n0)
    s_rough = np.arange(z_s.size) * ds_rough

    pos = cyc.distance + L                       # front axle position along the road
    gamma = np.interp(pos - L / 2.0, s_macro, grade_s)
    kappa = np.interp(pos - L / 2.0, s_macro, kappa_s)
    z_front = np.interp(pos, s_rough, z_s)
    z_rear = np.interp(pos - L, s_rough, z_s)

    a, v = cyc.accel, cyc.speed
    # Drivers slow down for curves; equivalently cap the lateral acceleration they accept.
    ay_max = rng.uniform(2.5, 6.0)
    kappa = np.clip(kappa, -ay_max / np.maximum(v * v, 1e-3), ay_max / np.maximum(v * v, 1e-3))

    # Pitch relative to road: static load + inertial pitch + road-induced pitch.
    wn_p = 2.0 * np.pi * veh.pitch_freq_hz
    pitch_in = theta_static + veh.pitch_gradient_rad_per_ms2 * a
    theta, _ = second_order(pitch_in, wn_p, veh.pitch_damping, dt)
    theta_road, _ = second_order(np.arctan2(z_front - z_rear, L), wn_p, veh.pitch_damping, dt,
                                 base_excitation=True)
    theta = theta + theta_road
    # Body heave from road roughness (vertical acceleration only).
    _, heave_acc = second_order(0.5 * (z_front + z_rear), 2.0 * np.pi * veh.heave_freq_hz,
                                veh.pitch_damping, dt, base_excitation=True)

    phi = gamma + theta
    a_lat = v * v * kappa
    roll = bank - veh.roll_gradient_rad_per_ms2 * a_lat     # body rolls out of the curve

    f_body = np.stack([
        a * np.cos(theta) + G * np.sin(phi),
        a_lat * np.cos(roll) + G * np.sin(roll) * np.cos(phi),
        -a * np.sin(theta) + G * np.cos(phi) * np.cos(roll) + heave_acc,
    ], axis=1)
    # Gyro: right-handed about ISO 8855 axes (x fwd, y left, z up) -> nose-up is negative gy.
    w_body = np.rad2deg(np.stack([
        np.gradient(roll, dt),
        -np.gradient(phi, dt),
        v * kappa,
    ], axis=1))

    imu = imu_measurements(rng, cfg.imu, fs, f_body, w_body, v)
    v_wheel = wheel_speed_measurement(rng, cfg.imu, v, a)
    raw = np.concatenate([imu, v_wheel[:, None]], axis=1)

    q = sc.decimation
    x = _decimate(raw, q).astype(np.float32)
    # Labels are sampled (not averaged) at the end of each decimation block so a
    # load step is not smeared across the block.
    idx = np.arange(q - 1, (n // q) * q, q)
    meta = {
        "vehicle": {k: float(val) for k, val in asdict(veh).items()},
        "load_case": load_a.name,
        "load_case_after_change": load_b.name if load_b else None,
        "grade_class": grade_cls,
        "roughness_class": rough_cls,
        "load_pitch_deg_initial": float(np.rad2deg(theta_static[0])),
    }
    return {
        "inputs": x,
        "load_pitch_deg": np.rad2deg(theta_static[idx]).astype(np.float32),
        "road_grade_deg": np.rad2deg(gamma[idx]).astype(np.float32),
        "body_pitch_deg": np.rad2deg(theta[idx]).astype(np.float32),
        "speed_ms": v[idx].astype(np.float32),
        "accel_ms2": a[idx].astype(np.float32),
        "meta": meta,
    }

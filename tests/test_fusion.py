import copy
import dataclasses

import numpy as np
import pytest

from headlamp_leveling import G
from headlamp_leveling.config import Config, ControlConfig, FusionConfig, ImuConfig
from headlamp_leveling.control import LevelingController
from headlamp_leveling.fusion import PitchEKF
from headlamp_leveling.fusion.ekf import (A, D2R, GAMR, N_STATES, THS, V, h_acc_x, h_acc_z, h_camera, h_gyro,
                                          h_wheel, propagate)
from headlamp_leveling.fusion.run import run_ekf
from headlamp_leveling.sim import simulate_trip


def _random_state(rng):
    x = rng.normal(0.0, 0.02, N_STATES)
    x[V], x[A], x[GAMR] = 15.0, 1.2, 0.01
    x[7] = 0.004          # pitch gradient
    x[11] = 1.01          # wheel scale
    return x


def _num_jac(f, x, eps=1e-6):
    f0 = np.atleast_1d(f(x))
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        d = np.zeros_like(x)
        d[i] = eps
        J[:, i] = (np.atleast_1d(f(x + d)) - np.atleast_1d(f(x - d))) / (2 * eps)
    return J


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_jacobians_match_finite_differences(seed):
    x = _random_state(np.random.default_rng(seed))
    _, F = propagate(x, 0.02, 2 * np.pi * 1.4, 0.4, 5.0)
    np.testing.assert_allclose(F, _num_jac(lambda z: propagate(z, 0.02, 2 * np.pi * 1.4, 0.4, 5.0)[0], x),
                               atol=1e-7)
    for h in (h_acc_x, lambda z: h_acc_z(z, 0.05), h_gyro, h_wheel, lambda z: h_camera(z, 30.0)):
        np.testing.assert_allclose(h(x)[1], _num_jac(lambda z: h(z)[0], x)[0], atol=1e-6)


def _ideal_cfg(trip_s=360.0):
    cfg = Config()
    cfg.imu = ImuConfig(**{f.name: 0 for f in dataclasses.fields(ImuConfig)})
    cfg.sim.roughness_classes = [("smooth", 1e-12, 1.0)]
    cfg.sim.trip_duration_s = trip_s
    cfg.sim.load_change_prob = 0.0
    cfg.camera.noise_deg = cfg.camera.bias_deg = cfg.camera.outlier_prob = 0.0
    return cfg


@pytest.mark.parametrize("seed", [3, 4])
def test_ekf_converges_with_ideal_sensors(seed):
    cfg = _ideal_cfg()
    raw = simulate_trip(cfg, seed, raw=True)["raw"]
    res = run_ekf(raw, cfg, use_camera=False)
    after = np.arange(res["load_pitch_deg"].size) / res["fs_hz"] >= 120.0
    err = np.abs(res["load_pitch_deg"] - res["true_load_pitch_deg"])[after]
    assert np.median(err) < 0.08
    assert np.all(res["load_pitch_std_deg"][after] < 0.3)


def test_camera_speeds_up_convergence():
    cfg = _ideal_cfg(trip_s=200.0)
    raw = simulate_trip(cfg, 5, raw=True)["raw"]
    no_cam = run_ekf(raw, cfg, use_camera=False)
    cam = run_ekf(raw, cfg, use_camera=True)
    assert cam["load_pitch_std_deg"][-1] < no_cam["load_pitch_std_deg"][-1]


def _drive(ekf, n, ax, az, v_wheel, gy=0.0, dt=0.02):
    for _ in range(n):
        out = ekf.step(dt, ax, az, gy, v_wheel)
    return out


def test_load_change_at_standstill_is_tracked():
    cfg = _ideal_cfg(trip_s=400.0)
    cfg.sim.load_change_prob = 1.0
    # First seed whose trip has a load change >= 3 s into a stop. Changes earlier than
    # standstill_settle_s + load_change_delay_s are left to the driving update by design.
    for seed in range(40):
        raw = simulate_trip(cfg, seed, raw=True)["raw"]
        steps = np.flatnonzero(np.abs(np.diff(raw["load_pitch_deg"])) > 0.05)
        if steps.size:
            moving = np.flatnonzero(raw["speed_ms"][:steps[0]] > 0.0)
            stop_start = moving[-1] + 1 if moving.size else 0
            if (steps[0] - stop_start) / raw["fs_hz"] >= 3.0:
                break
    res = run_ekf(raw, cfg, use_camera=False)
    fs = res["fs_hz"]
    k_change = int(steps[0] / raw["fs_hz"] * fs)
    true_jump = raw["load_pitch_deg"][steps[0] + 1] - raw["load_pitch_deg"][steps[0]]
    # Compare 1 s before the change with 5 s after it (still standing or just driving off).
    est_jump = res["load_pitch_deg"][k_change + int(5 * fs)] - res["load_pitch_deg"][k_change - int(fs)]
    assert est_jump == pytest.approx(true_jump, abs=0.1)
    assert res["road_grade_deg"][k_change + int(5 * fs)] == pytest.approx(
        res["road_grade_deg"][k_change - int(fs)], abs=0.1)


def test_drive_away_is_not_a_load_change():
    ekf = PitchEKF(FusionConfig())
    _drive(ekf, 300, 0.0, G, v_wheel=0.0)
    before = ekf.output().load_pitch_deg
    # The car starts accelerating at 2 m/s^2 before the wheel speed exceeds the standstill threshold.
    _drive(ekf, 5, 2.0, G, v_wheel=0.0)
    v = 0.0
    for _ in range(100):
        v += 2.0 * 0.02
        ekf.step(0.02, 2.0, G, 0.0, v)
    assert abs(ekf.output().load_pitch_deg - before) < 0.1


def test_camera_outlier_is_gated():
    cfg = FusionConfig()
    ekf = PitchEKF(cfg)
    ekf.x[V] = 20.0
    ekf.P[THS, THS] = (0.1 * D2R) ** 2
    assert not ekf.update_camera(5.0)          # 5 deg: far outside the innovation gate
    assert ekf.update_camera(0.05)


def test_sensor_fault_is_flagged():
    cfg = Config()
    cfg.sim.trip_duration_s = 120.0
    raw = copy.deepcopy(simulate_trip(cfg, 6, raw=True)["raw"])
    # Gyro y-axis fails stuck-at +20 deg/s halfway through the trip.
    raw["inputs"][raw["inputs"].shape[0] // 2:, 4] = 20.0
    res = run_ekf(raw, cfg, use_camera=False)
    half = res["fault"].size // 2
    assert not res["fault"][:half].any()
    assert res["fault"][half:].any()


def test_controller_uses_ekf_sigma():
    c = LevelingController(ControlConfig(max_rate_deg_s=100.0, deadband_deg=0.0, max_estimate_std_deg=0.3))
    out = c.update(1.0, 1.0, estimate_std_deg=0.5)      # too uncertain: ignored
    assert out.correction_deg == 0.0
    out = c.update(1.0, 1.0, estimate_std_deg=0.1)
    assert out.correction_deg == pytest.approx(1.0)

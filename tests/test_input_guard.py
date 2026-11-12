import copy

import numpy as np
import pytest

from headlamp_leveling import G
from headlamp_leveling.config import Config, ControlConfig, FusionConfig, InputGuardConfig
from headlamp_leveling.control import LevelingController
from headlamp_leveling.fusion import PitchEKF
from headlamp_leveling.fusion.ekf import THS, D2R, h_acc_z
from headlamp_leveling.fusion.input_guard import (CHANNELS, IDX, MISSING, RANGE, SPIKE, STUCK, Health,
                                                  SensorInputGuard)
from headlamp_leveling.fusion.run import closed_loop_series, run_ekf
from headlamp_leveling.sim import simulate_trip
from headlamp_leveling.sim.faults import SCENARIOS, inject

DT = 0.01
NAN = float("nan")


def _sample(rng, speed=0.0, **override):
    """One plausible raw sample: standstill (quiet) or driving (road vibration)."""
    vib = 0.01 if speed == 0.0 else 0.3
    s = {"ax": rng.normal(0.0, vib * 0.4), "ay": rng.normal(0.0, vib * 0.6), "az": G + rng.normal(0.0, vib),
         "gx": rng.normal(0.0, 0.06), "gy": rng.normal(0.0, 0.06), "gz": rng.normal(0.0, 0.06),
         "v_wheel": 0.0 if speed == 0.0 else speed + rng.normal(0.0, 0.02), "camera": NAN}
    s.update(override)
    return [s[c] for c in CHANNELS]


def _run(guard, rng, seconds, speed=0.0, **override):
    out = None
    for _ in range(int(seconds / DT)):
        out = guard.process(_sample(rng, speed, **override), DT)
    return out


def test_clean_signals_pass_unchanged():
    rng = np.random.default_rng(0)
    for speed in (0.0, 20.0):
        g = SensorInputGuard(InputGuardConfig())
        for _ in range(500):
            s = _sample(rng, speed)
            out = g.process(s, DT)
            assert out.values[:7] == s[:7]
            assert not out.critical_failed and not out.failsafe
        assert all(h == Health.OK for h in out.health.values())
        assert all(not v for k, v in g.summary().items() if k != "camera")


def test_missing_samples_degrade_then_fail_then_heal():
    rng = np.random.default_rng(1)
    cfg = InputGuardConfig()
    g = SensorInputGuard(cfg)
    _run(g, rng, 2.0, speed=20.0)
    out = g.process(_sample(rng, 20.0, az=NAN), DT)
    assert np.isnan(out.value("az")) and out.reasons["az"] == MISSING
    assert out.health["az"] == Health.DEGRADED and not out.critical_failed
    out = _run(g, rng, cfg.qualify_s + 0.05, speed=20.0, az=NAN)
    assert out.health["az"] == Health.FAILED and out.critical_failed and not out.failsafe
    out = _run(g, rng, cfg.failsafe_after_s, speed=20.0, az=NAN)
    assert out.failsafe
    # Data returns: still withheld until it has been good for heal_s, then released.
    out = _run(g, rng, cfg.heal_s - 0.2, speed=20.0)
    assert out.health["az"] == Health.FAILED and np.isnan(out.value("az"))
    out = _run(g, rng, 0.4, speed=20.0)
    assert out.health["az"] != Health.FAILED and np.isfinite(out.value("az")) and not out.failsafe


def test_saturated_sample_is_rejected():
    rng = np.random.default_rng(2)
    g = SensorInputGuard(InputGuardConfig())
    _run(g, rng, 1.0, speed=20.0)
    out = g.process(_sample(rng, 20.0, ax=19.6, gy=250.0), DT)
    assert out.reasons["ax"] == RANGE and out.reasons["gy"] == RANGE
    assert np.isnan(out.value("ax")) and np.isnan(out.value("gy"))


def test_spike_is_rejected_but_a_real_step_is_followed():
    rng = np.random.default_rng(3)
    g = SensorInputGuard(InputGuardConfig())
    _run(g, rng, 1.0, speed=20.0)
    out = g.process(_sample(rng, 20.0, az=G + 8.0), DT)
    assert out.reasons["az"] == SPIKE and np.isnan(out.value("az"))
    out = g.process(_sample(rng, 20.0), DT)
    assert np.isfinite(out.value("az"))
    # A sustained change of level is accepted after a few samples (it is not a spike).
    accepted = [np.isfinite(g.process(_sample(rng, 20.0, ax=6.0 + rng.normal(0, 0.1)), DT).value("ax"))
                for _ in range(10)]
    assert not accepted[0] and all(accepted[4:])


def test_frozen_imu_channel_fails_but_zero_wheel_speed_is_fine():
    rng = np.random.default_rng(4)
    cfg = InputGuardConfig()
    g = SensorInputGuard(cfg)
    _run(g, rng, 3.0, speed=0.0)                       # v_wheel is exactly 0 all the time
    out = _run(g, rng, cfg.stuck_s + cfg.qualify_s + 0.1, speed=0.0, gy=0.1234)
    assert out.health["gy"] == Health.FAILED and out.reasons["gy"] & STUCK
    assert out.health["v_wheel"] == Health.OK
    assert not out.critical_failed                      # the pitch gyro is not critical


def test_wheel_speed_dropping_to_zero_while_driving_never_reaches_the_filter():
    rng = np.random.default_rng(5)
    g = SensorInputGuard(InputGuardConfig())
    _run(g, rng, 3.0, speed=20.0)
    zeros_passed = 0
    for _ in range(int(20.0 / DT)):                    # body still shakes like a moving car
        out = g.process(_sample(rng, 20.0, v_wheel=0.0), DT)
        zeros_passed += int(out.value("v_wheel") == 0.0)
    assert zeros_passed == 0
    assert out.health["v_wheel"] == Health.FAILED and out.failsafe
    out = _run(g, rng, 8.0, speed=20.0)                # sensor back
    assert out.health["v_wheel"] == Health.OK and out.value("v_wheel") > 19.0 and not out.failsafe


def test_gyro_offset_is_caught_by_plausibility_without_failsafe():
    rng = np.random.default_rng(6)
    g = SensorInputGuard(InputGuardConfig())
    _run(g, rng, 3.0, speed=20.0)
    out = None
    for _ in range(int(10.0 / DT)):
        out = g.process(_sample(rng, 20.0, gy=20.0 + rng.normal(0.0, 0.06)), DT)
    assert out.health["gy"] == Health.FAILED and np.isnan(out.value("gy"))
    assert not out.critical_failed and not out.failsafe and out.degraded
    assert g.summary()["plausibility_rules"] == {"gyro_mean_y": 1}


def test_accelerometer_offset_at_standstill_violates_gravity_norm():
    rng = np.random.default_rng(7)
    g = SensorInputGuard(InputGuardConfig())
    _run(g, rng, 3.0, speed=0.0)
    out = None
    for _ in range(int(6.0 / DT)):
        out = g.process(_sample(rng, 0.0, az=G + 1.5 + rng.normal(0.0, 0.01)), DT)
    assert out.health["az"] == Health.FAILED and out.critical_failed


def test_camera_absence_is_normal_but_a_frozen_camera_is_rejected():
    rng = np.random.default_rng(8)
    g = SensorInputGuard(InputGuardConfig())
    out = _run(g, rng, 5.0, speed=20.0)                # no frames at all
    assert out.health["camera"] == Health.OK
    used = 0
    for k in range(int(5.0 / DT)):
        cam = 1.5 if k % 10 == 0 else NAN              # same frame result at 10 Hz
        out = g.process(_sample(rng, 20.0, camera=cam), DT)
        used += int(np.isfinite(out.value("camera")))
    assert used <= InputGuardConfig().camera_stuck_frames
    assert out.health["camera"] == Health.FAILED and not out.critical_failed


def test_ekf_coasts_on_missing_inputs_and_stays_finite():
    ekf = PitchEKF(FusionConfig())
    v = 15.0
    for k in range(6000):                              # flat road, level car, accelerate / brake
        a = 1.5 if (k // 250) % 2 == 0 else -1.5
        v += a * 0.02
        ax = a if k % 7 else NAN                       # 14 % of accel samples missing
        gy = NAN if 600 <= k < 900 else 0.0            # gyro gone for 6 s
        vw = NAN if 1300 <= k < 1400 else v            # wheel speed gone for 2 s
        out = ekf.step(0.02, ax, G, gy, vw)
        assert np.isfinite(out.load_pitch_deg)
    assert np.all(np.isfinite(ekf.x)) and np.all(np.isfinite(ekf.P))
    assert abs(out.load_pitch_deg) < 0.2 and ekf.numeric_resets == 0


def test_ekf_long_gap_is_predicted_in_substeps():
    a, b = PitchEKF(FusionConfig()), PitchEKF(FusionConfig())
    for ekf in (a, b):
        for _ in range(100):
            ekf.step(0.02, 0.5, G, 0.0, 5.0)
    a.step(0.5, NAN, NAN, NAN, NAN)                    # one call spanning 0.5 s
    for _ in range(25):
        b.step(0.02, NAN, NAN, NAN, NAN)               # the same time in nominal steps
    np.testing.assert_allclose(a.x, b.x, rtol=0.05, atol=1e-3)


def test_ekf_recovers_from_numerical_corruption():
    ekf = PitchEKF(FusionConfig())
    for _ in range(200):
        ekf.step(0.02, 0.0, G, 0.0, 0.0)
    ekf.x[THS] = 0.4 * D2R
    ekf.step(0.02, 0.0, G, 0.0, 0.0)
    good = ekf.output().load_pitch_deg
    ekf.P[3, 3] = np.nan
    out = ekf.step(0.02, 0.0, G, 0.0, 0.0)
    assert ekf.numeric_resets == 1 and np.all(np.isfinite(ekf.P))
    assert out.load_pitch_deg == pytest.approx(good, abs=0.05)
    assert out.load_pitch_std_deg == pytest.approx(FusionConfig().recover_load_pitch_std_deg, rel=0.2)


def test_huber_weighting_limits_the_pull_of_an_outlier():
    def pull(k):
        cfg = FusionConfig(huber_k=k, gate_imu=1e9)
        ekf = PitchEKF(cfg)
        for _ in range(300):
            ekf.step(0.02, 0.0, G, 0.0, 0.0)
        before = ekf.x.copy()
        h, H = h_acc_z(ekf.x)
        ekf.update_scalar("acc_z", h + 1.0, h, H, 0.03 ** 2, huber=True)
        return float(np.linalg.norm(ekf.x - before))
    assert pull(3.0) < 0.5 * pull(0.0)


def test_controller_failsafe_request_does_not_latch():
    c = LevelingController(ControlConfig(max_rate_deg_s=100.0, deadband_deg=0.0))
    assert c.update(1.0, 1.0, estimate_std_deg=0.1).correction_deg == pytest.approx(1.0)
    out = c.update(1.0, 0.2, estimate_std_deg=0.1, failsafe_request=True)
    assert out.fault and out.correction_deg == pytest.approx(1.0)       # held, not raised
    out = c.update(1.0, 0.2, estimate_std_deg=0.1)
    assert not out.fault and out.correction_deg == pytest.approx(0.2)   # leveling resumes


def test_inject_does_not_modify_the_original_trip():
    cfg = Config()
    cfg.sim.trip_duration_s = 260.0
    raw = simulate_trip(cfg, 3, raw=True)["raw"]
    before = raw["inputs"].copy()
    for name in SCENARIOS:
        faulty, _ = inject(raw, name, seed=1)
        assert np.array_equal(raw["inputs"], before)
        if name != "none" and not name.startswith("camera"):
            assert not np.array_equal(faulty["inputs"], before, equal_nan=True)
    with pytest.raises(KeyError):
        inject(raw, "no_such_fault")


@pytest.fixture(scope="module")
def trip_raw():
    cfg = Config()
    cfg.sim.trip_duration_s = 330.0
    return cfg, simulate_trip(cfg, 6, raw=True)["raw"]


def test_guard_changes_nothing_on_a_healthy_trip(trip_raw):
    cfg, raw = trip_raw
    res = run_ekf(raw, cfg, use_camera=True, guard=True)
    assert not res["failsafe_request"].any() and not res["fault"].any()
    assert not res["guard"]["summary"]["plausibility_rules"]
    ref = run_ekf(raw, cfg, use_camera=True, guard=False)
    assert np.abs(res["load_pitch_deg"] - ref["load_pitch_deg"])[-1] < 0.05


@pytest.mark.parametrize("scenario", ["gyro_stuck", "gyro_offset", "spikes"])
def test_guarded_filter_keeps_leveling_through_faults(trip_raw, scenario):
    cfg, raw = trip_raw
    faulty, _ = inject(raw, scenario, seed=0)
    clean = closed_loop_series(run_ekf(raw, cfg, use_camera=True, guard=True), cfg)
    res = run_ekf(faulty, cfg, use_camera=True, guard=True)
    cl = closed_loop_series(res, cfg)
    assert np.all(np.isfinite(res["load_pitch_deg"]))
    assert not cl["failsafe"].any()
    late = cl["t"] >= 240.0
    assert np.abs(cl["err"][late]).mean() < np.abs(clean["err"][late]).mean() + 0.1


def test_critical_outage_goes_to_failsafe_and_comes_back(trip_raw):
    cfg, raw = trip_raw
    faulty, windows = inject(raw, "imu_outage", seed=0)
    res = run_ekf(faulty, cfg, use_camera=True, guard=True)
    cl = closed_loop_series(res, cfg)
    during = (cl["t"] >= windows[0].t0 + 5.0) & (cl["t"] <= windows[0].t1)
    assert cl["failsafe"][during].all()
    assert not cl["failsafe"][cl["t"] >= windows[0].t1 + 10.0].any()
    assert np.all(np.isfinite(res["load_pitch_deg"])) and res["numeric_resets"] == 0

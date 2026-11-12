import numpy as np
import pytest

from headlamp_leveling import G
from headlamp_leveling.config import Config, FusionConfig
from headlamp_leveling.fusion import PitchEKF
from headlamp_leveling.sim.halfcar import (LOADS, HalfCarParams, ScriptedDrive, simulate_scripted,
                                           slope_pitch_per_moment, speed_profile, static_equilibrium,
                                           static_load_pitch_deg)
from headlamp_leveling.verify import SCENARIOS, run_scenario
from headlamp_leveling.verify.criteria import (BeamNotRaised, Converges, LampQuiet, LoadStepTracked, RunData,
                                               SteadyAccuracy)
from headlamp_leveling.verify.scenarios import Script


# ----------------------------------------------------------------- plant physics

def test_static_trim_signs_and_magnitudes():
    p = HalfCarParams()
    assert static_load_pitch_deg(p, "driver_only") == pytest.approx(0.0, abs=1e-9)
    assert static_load_pitch_deg(p, "front_passenger") < 0.0          # ahead of the CG: nose down
    assert 0.8 < static_load_pitch_deg(p, "full_cabin_trunk") < 1.8
    assert static_load_pitch_deg(p, "max_rear") > static_load_pitch_deg(p, "full_cabin_trunk")
    # A load exactly at the CG only lowers the car.
    x = static_equilibrium(p, 100.0, 0.0)
    assert x[0] < 0.0 and abs(np.degrees(x[1])) < 0.12


def test_static_trim_matches_rigid_tyre_hand_calculation():
    p = HalfCarParams(k_tf=1e12, k_tr=1e12)
    m, x = 150.0, p.x_trunk                                            # load behind the rear axle
    f_rear = m * G * x / p.wheelbase
    f_front = m * G - f_rear
    expected = np.degrees((f_rear / p.k_r - f_front / p.k_f) / p.wheelbase)
    assert static_load_pitch_deg(p, "trunk_heavy") == pytest.approx(expected, rel=1e-3)


def test_speed_profile_respects_targets_and_jerk():
    ends = []
    a, v = speed_profile([("hold", 2), ("to", 72, 2.0), ("hold", 3), ("to", 0, 3.0), ("hold", 2)], 1e-3, 4.0, ends)
    assert len(ends) == 5 and ends == sorted(ends)
    assert v.max() == pytest.approx(20.0, abs=0.3) and v[-1] == 0.0 and v.min() >= 0.0
    assert a.max() <= 2.0 + 1e-6 and a.min() >= -3.0 - 1e-6
    assert np.abs(np.diff(a)).max() / 1e-3 <= 4.0 + 1e-6 or v[np.argmax(np.abs(np.diff(a)))] < 0.1


def _steady(drive_kw, t_from, **veh):
    cfg = Config()
    drive = ScriptedDrive(roughness=None, vehicle=HalfCarParams(f_fric=0.0, **veh), **drive_kw)
    raw = simulate_scripted(drive, cfg, ideal_sensors=True)
    k = int(t_from * raw["fs_hz"])
    return raw, k


def test_plant_squat_matches_the_analytic_pitch_gradient():
    p = HalfCarParams(f_fric=0.0)
    raw, k = _steady(dict(manoeuvres=[("hold", 2), ("to", 150, 2.0)]), 6.0)
    sl = slice(k, k + 300)
    assert raw["accel_ms2"][sl].mean() == pytest.approx(2.0, abs=0.01)
    expected = np.degrees(slope_pitch_per_moment(p, braking=False) * p.m_s * 2.0 * p.h)
    assert raw["body_pitch_deg"][sl].mean() == pytest.approx(expected, rel=0.03)
    # The accelerometer sees the acceleration plus gravity leaking in through the squat.
    th = np.deg2rad(raw["body_pitch_deg"][sl].mean())
    assert raw["inputs"][sl, 0].mean() == pytest.approx(2.0 * np.cos(th) + G * np.sin(th), abs=0.02)


def test_plant_standing_on_a_slope():
    grade = [(0, 10.0), (1000, 10.0)]
    raw, k = _steady(dict(manoeuvres=[("hold", 8)], grade=grade), 5.0)
    gamma = np.arctan(0.10)
    assert raw["road_grade_deg"][k] == pytest.approx(np.degrees(gamma), abs=1e-6)
    th = np.deg2rad(raw["body_pitch_deg"][k:].mean())
    assert th > 0.0                                                    # load transfer to the rear: nose up
    assert np.degrees(th) == pytest.approx(raw["aim_pitch_deg"][k], abs=0.01)
    assert raw["inputs"][k:, 0].mean() == pytest.approx(G * np.sin(gamma + th), abs=0.01)
    assert np.hypot(raw["inputs"][k:, 0].mean(), raw["inputs"][k:, 2].mean()) == pytest.approx(G, rel=1e-3)


def test_load_change_must_be_at_standstill():
    drive = ScriptedDrive([("hold", 2), ("to", 50, 2.0), ("hold", 5)], loads=[(0.0, "driver_only"), (6.0, "trailer")])
    with pytest.raises(ValueError):
        simulate_scripted(drive, Config())
    with pytest.raises(KeyError):
        simulate_scripted(ScriptedDrive([("hold", 2)], loads=[(0.0, "elephant")]), Config())


# ----------------------------------------------------------------- lever arm

def test_lever_arm_compensation_removes_pitch_acceleration():
    """A pitching body with the IMU 1 m ahead of the axis: az must come out flat."""
    ekf = PitchEKF(FusionConfig(imu_lever_x_m=1.0, imu_lever_z_m=0.0))
    dt, w = 0.02, 2 * np.pi * 1.2
    out = []
    for k in range(200):
        t = k * dt
        th_rate = 0.02 * w * np.cos(w * t)                 # rad/s
        th_acc = -0.02 * w * w * np.sin(w * t)
        s = ekf._remove_lever_arm(dt, 0.0, G + 1.0 * th_acc, -np.rad2deg(th_rate), 10.0, float("nan"),
                                  0.0, 0.0)
        if s is not None:
            out.append(s[1])
    assert np.std(np.array(out[5:]) - G) < 0.02 * (0.02 * w * w)       # < 2 % of the injected amplitude


# ----------------------------------------------------------------- criteria

def _data(err, correction=None, failsafe=None, load_times=(), faults=()):
    n = len(err)
    t = np.arange(n, dtype=float)
    corr = np.zeros(n) if correction is None else np.asarray(correction, float)
    return RunData(t=t, err=np.asarray(err, float), correction=corr, truth=np.zeros(n),
                   failsafe=np.zeros(n, bool) if failsafe is None else np.asarray(failsafe, bool),
                   t_est=t, est=np.zeros(n), est_std=np.ones(n), est_truth=np.zeros(n), t_raw=t,
                   speed=np.ones(n), guard_failed=np.zeros(n, bool), t_drive=0.0, load_times=load_times,
                   fault_windows=faults, numeric_resets=0)


def test_criteria_pass_and_fail_as_specified():
    err = np.r_[np.full(40, 1.0), np.full(160, 0.05)]
    assert Converges(within_s=60).evaluate(_data(err)).passed
    assert not Converges(within_s=30).evaluate(_data(err)).passed
    assert SteadyAccuracy(converge_s=60).evaluate(_data(err)).passed
    assert not SteadyAccuracy(converge_s=60).evaluate(_data(np.full(200, 0.2))).passed

    step = np.r_[np.zeros(100), np.full(8, 2.0), np.zeros(92)]
    assert LoadStepTracked(100.0, within_s=12).evaluate(_data(step, load_times=(100.0,))).passed
    assert not LoadStepTracked(100.0, within_s=5).evaluate(_data(step, load_times=(100.0,))).passed
    # The grace period after a load change keeps the step out of the steady-state numbers.
    assert SteadyAccuracy(converge_s=0).evaluate(_data(step, load_times=(100.0,))).passed

    corr = np.r_[np.full(100, 1.2), np.full(50, 0.6), np.full(50, 1.2)]
    assert not BeamNotRaised(100.0, 150.0).evaluate(_data(np.zeros(200), correction=corr)).passed
    assert BeamNotRaised(100.0, 150.0).evaluate(_data(np.zeros(200), correction=np.full(200, 1.2))).passed
    assert not LampQuiet(0.0, 199.0, 0.1).evaluate(_data(np.zeros(200), correction=corr)).passed


def test_script_reports_step_times():
    s = Script().hold(5, "wait").to(36, 2.0, "launch").hold(4, "cruise")
    t = s.times()
    assert t["wait"] == (0.0, pytest.approx(5.0))
    assert t["launch"][0] == pytest.approx(5.0) and t["cruise"][1] == pytest.approx(t["_end"][0])
    assert 4.5 < t["launch"][1] - t["launch"][0] < 7.0                # 10 m/s at 2 m/s^2 plus jerk ramps


# ----------------------------------------------------------------- end to end

def test_scenarios_are_well_formed():
    assert len(SCENARIOS) >= 15
    for name, make in SCENARIOS.items():
        sc = make()
        assert sc.name == name and sc.criteria and sc.purpose
        assert all(load in LOADS for _, load in sc.drive.loads)


@pytest.mark.parametrize("name", ["parked_on_slope_loading", "fault_imu_outage"])
def test_scenario_runs_and_passes(name):
    r = run_scenario(SCENARIOS[name](), Config())
    failed = [f"{c.name}: {c.measured}" for c in r["checks"] if not c.passed]
    assert r["passed"], failed


# ----------------------------------------------------------------- turning and IMU location

def _steady_turn(x_imu: float, radius: float = 20.0):
    p = HalfCarParams(x_imu=x_imu)
    drive = ScriptedDrive([("hold", 1), ("to", 25, 2.0), ("hold", 25)], curvature=[(0, 0), (40, 1 / radius), (400, 1 / radius)],
                          roughness=None, vehicle=p)
    raw = simulate_scripted(drive, Config(), ideal_sensors=True)
    sl = slice(-500, None)                                           # last 5 s: steady circle at constant speed
    return p, raw["inputs"][sl], raw["speed_ms"][sl]


def test_plant_turning_terms_depend_on_the_distance_from_the_rear_axle():
    p, at_axle, v = _steady_turn(x_imu=-HalfCarParams().b)          # IMU on the rear axle
    _, ahead, _ = _steady_turn(x_imu=0.9)
    yaw = np.deg2rad(at_axle[:, 5]).mean()
    a_lat = float(v.mean()) * yaw
    assert yaw == pytest.approx(v.mean() / 20.0, rel=1e-3)
    # On the rear axle only the side-slip term is left; it does not depend on the turn direction.
    assert at_axle[:, 0].mean() == pytest.approx(p.rear_slip_gradient() * a_lat ** 2, rel=0.03)
    # 2.4 m further forward the centripetal term is added.
    assert (ahead[:, 0] - at_axle[:, 0]).mean() == pytest.approx(-yaw ** 2 * (0.9 + p.b), rel=0.02)


def test_filter_removes_the_turning_terms():
    cfg = FusionConfig(imu_rear_axle_x_m=2.4, rear_slip_rad_per_ms2=0.007)
    ekf = PitchEKF(cfg)
    v, yaw_dps = 6.0, 20.0
    yaw = np.deg2rad(yaw_dps)
    ax_in = 0.007 * (v * yaw) ** 2 - yaw ** 2 * 2.4                 # what the plant adds for this turn
    ax, az, *_ = ekf._remove_turn_terms((ax_in, 9.81, 0.0, v, float("nan"), 0.0, yaw_dps))
    assert ax == pytest.approx(0.0, abs=1e-9) and az == 9.81
    assert ekf._turn_var > 0.0                                       # acc_x is trusted less while turning
    # Without a yaw rate nothing is changed.
    assert ekf._remove_turn_terms((ax_in, 9.81, 0.0, v, float("nan"), 0.0, float("nan")))[0] == ax_in
    assert ekf._turn_var == 0.0


def test_mounting_study_runs(tmp_path):
    from headlamp_leveling.verify.mounting import LOCATIONS, mounting_study
    veh = HalfCarParams()
    x, z = LOCATIONS["overhead_console"].lever(veh)
    assert abs(x) < 0.3 and z > 0.5                                  # roughly above the CG, high up
    rep = mounting_study(Config(), ["floor_centre", "headlamp"], ["urban_turns"], n_seeds=1, out_dir=str(tmp_path),
                         sensitivity=False, plots=False, workers=1, verbose=False)
    assert [e["name"] for e in rep["locations"]] == ["floor_centre", "headlamp"]
    assert rep["locations"][0]["variants"]["known"]["runs"] == 1
    assert (tmp_path / "report.md").read_text().startswith("# IMU mounting-location study")
    # A second call reuses the stored runs.
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 6
    mounting_study(Config(), ["floor_centre", "headlamp"], ["urban_turns"], n_seeds=1, out_dir=str(tmp_path),
                   sensitivity=False, plots=False, workers=1, verbose=False)
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 6

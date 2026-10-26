import dataclasses

import numpy as np
import pytest

from headlamp_leveling import G
from headlamp_leveling.config import Config, ImuConfig
from headlamp_leveling.data.features import IDX, build_features
from headlamp_leveling.models.physics import PhysicsEstimator
from headlamp_leveling.sim import LoadCase, LoadItem, VehicleParams, simulate_trip, static_load_pitch_rad
from headlamp_leveling.sim.road import iso8608_profile


def _vehicle(**kw):
    base = dict(wheelbase_m=2.7, front_axle_rate_n_m=50e3, rear_axle_rate_n_m=50e3,
                pitch_gradient_rad_per_ms2=0.005, pitch_freq_hz=1.4, pitch_damping=0.4,
                heave_freq_hz=1.3, roll_gradient_rad_per_ms2=0.008)
    base.update(kw)
    return VehicleParams(**base)


def _ideal_cfg(**sim):
    cfg = Config()
    cfg.imu = ImuConfig(**{f.name: 0 for f in dataclasses.fields(ImuConfig)})
    cfg.sim.roughness_classes = [("smooth", 1e-12, 1.0)]
    cfg.sim.load_change_prob = 0.0
    cfg.vehicle.pitch_gradient_deg_per_g = (0.0, 0.0)
    for k, v in sim.items():
        setattr(cfg.sim, k, v)
    return cfg


def test_load_pitch_signs():
    veh = _vehicle()
    assert static_load_pitch_rad(veh, LoadCase("none")) == 0.0
    trunk = LoadCase("trunk", [LoadItem("trunk", 100.0, 3.3)])
    front = LoadCase("front", [LoadItem("bag", 100.0, 0.2)])
    assert static_load_pitch_rad(veh, trunk) > 0.0          # rear squats -> nose up
    assert static_load_pitch_rad(veh, front) < 0.0          # front squats -> nose down
    # Load exactly between the axles of a symmetric car keeps the body level.
    mid = LoadCase("mid", [LoadItem("x", 100.0, 1.35)])
    assert static_load_pitch_rad(veh, mid) == pytest.approx(0.0, abs=1e-12)


def test_trunk_load_magnitude_is_plausible():
    veh = _vehicle(front_axle_rate_n_m=45e3, rear_axle_rate_n_m=45e3)
    full = LoadCase("full", [LoadItem("fp", 75, 1.0)] + [LoadItem("r", 75, 2.15)] * 3 + [LoadItem("t", 120, 3.3)])
    deg = np.rad2deg(static_load_pitch_rad(veh, full))
    assert 0.5 < deg < 2.0


def test_iso8608_variance_matches_psd():
    rng = np.random.default_rng(0)
    gd = 64e-6
    z = iso8608_profile(rng, 20000.0, 0.05, gd)
    # integral of Gd(n0) (n/n0)^-2 over 0.01..10 cycles/m
    expected = gd * 0.1 ** 2 * (1 / 0.01 - 1 / 10.0)
    assert z.var() == pytest.approx(expected, rel=0.35)


def test_trip_shapes_and_labels():
    cfg = Config()
    cfg.sim.trip_duration_s = 60.0
    trip = simulate_trip(cfg, 0)
    T = int(60.0 * cfg.model_fs_hz)
    assert trip["inputs"].shape == (T, 7)
    for k in ("load_pitch_deg", "road_grade_deg", "body_pitch_deg", "speed_ms"):
        assert trip[k].shape == (T,)
        assert np.all(np.isfinite(trip[k]))
    assert np.all(trip["inputs"][:, IDX["v_wheel"]] >= 0.0)


def test_standstill_accelerometer_sees_grade_plus_load_pitch():
    cfg = _ideal_cfg(trip_duration_s=30.0)
    trip = simulate_trip(cfg, 3)
    x = trip["inputs"]
    k = 5  # first samples: vehicle starts at standstill (>= 2 s)
    tilt = np.rad2deg(np.arctan2(x[:k, IDX["ax"]].mean(), x[:k, IDX["az"]].mean()))
    expected = trip["road_grade_deg"][0] + trip["load_pitch_deg"][0]
    assert tilt == pytest.approx(expected, abs=0.02)
    assert np.linalg.norm(x[:k, :3].mean(axis=0)) == pytest.approx(G, rel=1e-3)


@pytest.mark.parametrize("seed", [1, 2])
def test_physics_estimator_recovers_pitch_with_ideal_sensors(seed):
    cfg = _ideal_cfg(trip_duration_s=400.0)
    trip = simulate_trip(cfg, seed)
    feats = build_features(trip["inputs"], cfg.model_fs_hz)
    est = PhysicsEstimator(cfg.model_fs_hz, min_excitation_ms2=0.3)
    W = int(120 * cfg.model_fs_hz)
    errs = []
    for e in range(W - 1, len(feats), W // 2):
        p = est.estimate(feats[e - W + 1:e + 1])
        if np.isfinite(p):
            errs.append(p - trip["load_pitch_deg"][e])
    assert errs, "no excited windows"
    assert np.median(np.abs(errs)) < 0.1

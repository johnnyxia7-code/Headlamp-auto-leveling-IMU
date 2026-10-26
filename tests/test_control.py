import numpy as np
import pytest

from headlamp_leveling.config import ControlConfig
from headlamp_leveling.control import LevelingController


def _ctrl(**kw):
    return LevelingController(ControlConfig(**kw))


def test_converges_to_estimate_and_respects_slew_rate():
    c = _ctrl(max_rate_deg_s=0.5)
    positions = [c.update(1.0, 1.2, excitation=1.0).correction_deg for _ in range(60)]
    assert positions[-1] == pytest.approx(1.2, abs=0.06)
    assert np.max(np.abs(np.diff([0.0] + positions))) <= 0.5 + 1e-9


def test_ignores_estimates_without_excitation():
    c = _ctrl()
    out = c.update(1.0, 2.0, excitation=0.0)
    assert out.estimate_deg == 0.0 and out.correction_deg == 0.0
    out = c.update(1.0, float("nan"), excitation=5.0)
    assert out.estimate_deg == 0.0


def test_deadband_holds_small_changes():
    c = _ctrl(deadband_deg=0.1, filter_tau_s=0.0)
    c.update(1.0, 0.05, excitation=1.0)
    assert c.target == 0.0
    c.update(1.0, 0.5, excitation=1.0)
    assert c.target > 0.1


def test_range_clamp():
    c = _ctrl(range_deg=(-1.0, 2.0), max_rate_deg_s=100.0)
    for _ in range(50):
        out = c.update(1.0, 5.0, excitation=1.0)
    assert out.correction_deg == pytest.approx(2.0)


def test_standstill_tracks_load_change_from_tilt():
    c = _ctrl(max_rate_deg_s=100.0, deadband_deg=0.0)
    for _ in range(30):
        c.update(1.0, 0.4, excitation=1.0)
    est_before = c.estimate
    # Stop on a 3 deg slope; then 0.6 deg of extra nose-up from loading the trunk.
    c.update(1.0, standstill=True, tilt_deg=3.4)
    out = c.update(1.0, standstill=True, tilt_deg=4.0)
    assert out.estimate_deg == pytest.approx(est_before + 0.6)
    assert out.correction_deg == pytest.approx(est_before + 0.6)


def test_fault_moves_to_failsafe_and_latches():
    c = _ctrl(failsafe_deg=0.0, max_rate_deg_s=100.0)
    for _ in range(20):
        c.update(1.0, 1.5, excitation=1.0)
    out = c.update(1.0, fault=True)
    assert out.fault and out.correction_deg == 0.0
    out = c.update(1.0, 1.5, excitation=1.0)
    assert out.fault and out.correction_deg == 0.0
    assert out.steps == 0

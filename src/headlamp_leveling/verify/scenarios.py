"""Verification scenarios: scripted drives with explicit acceptance criteria.

Every scenario is deterministic (fixed script, fixed noise seed) and states
what it is meant to prove. The limits in the criteria are engineering targets
chosen before looking at the results; they are documented in
``docs/verification.md`` and are meant to be replaced by the project's real
requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..sim.halfcar import HalfCarParams, ScriptedDrive, speed_profile
from .criteria import (ActuatorTravel, AimWithin, BeamNotRaised, Consistent, Converges, Criterion, FaultDetected, LampQuiet,
                       LevelingContinues, LoadStepTracked, NoFailsafe, NoGlare, NumericallySound, Recovers,
                       SteadyAccuracy)

Fault = Tuple[str, str, float, float, float]      # channel, kind, t0, t1, value


@dataclass
class VerifyScenario:
    name: str
    title: str
    purpose: str
    drive: ScriptedDrive
    criteria: List[Criterion]
    faults: List[Fault] = field(default_factory=list)
    use_camera: bool = True
    group: str = "nominal"
    lever_arm_scale: float = 1.0      # the filter is told this multiple of the true IMU lever arm


class Script:
    """Builds a manoeuvre list and remembers when each named step starts and ends."""

    def __init__(self, jerk: float = 4.0):
        self.manoeuvres: List[tuple] = []
        self.names: List[Optional[str]] = []
        self.jerk = jerk

    def hold(self, seconds: float, name: Optional[str] = None) -> "Script":
        self.manoeuvres.append(("hold", seconds))
        self.names.append(name)
        return self

    def to(self, kmh: float, accel: float, name: Optional[str] = None) -> "Script":
        self.manoeuvres.append(("to", kmh, accel))
        self.names.append(name)
        return self

    def times(self) -> Dict[str, Tuple[float, float]]:
        ends: List[float] = []
        speed_profile(self.manoeuvres, 1e-3, self.jerk, ends)
        starts = [0.0] + ends[:-1]
        out = {n: (s, e) for n, s, e in zip(self.names, starts, ends) if n}
        out["_end"] = (ends[-1], ends[-1])
        return out

    def distances(self) -> Dict[str, Tuple[float, float]]:
        """Distance travelled (m) at the start and end of each named step."""
        dt = 1e-3
        _, v = speed_profile(self.manoeuvres, dt, self.jerk)
        s = np.concatenate([[0.0], np.cumsum(v) * dt])
        return {n: (float(s[min(int(t0 / dt), s.size - 1)]), float(s[min(int(t1 / dt), s.size - 1)]))
                for n, (t0, t1) in self.times().items()}


def _city(s: Optional[Script] = None) -> Script:
    """About 4.5 minutes of urban stop-and-go."""
    s = s or Script()
    return (s.hold(20, "start").to(50, 2.0).hold(15).to(0, 2.5).hold(8, "stop1")
            .to(70, 1.8).hold(20).to(30, 2.0).hold(10).to(60, 1.5).hold(15, "cruise")
            .to(0, 3.0).hold(10, "stop2").to(50, 2.0).hold(20).to(80, 1.5).hold(25, "cruise2")
            .to(40, 2.0).hold(15).to(0, 2.0).hold(8, "stop3").to(60, 2.0).hold(30, "cruise3").to(0, 2.5).hold(5))


def _standard(converge_s: float = 90.0, mae: float = 0.15, p95: float = 0.30) -> List[Criterion]:
    return [Converges(within_s=converge_s), SteadyAccuracy(mae_deg=mae, p95_deg=p95, converge_s=converge_s),
            NoGlare(converge_s=converge_s), ActuatorTravel(converge_s=converge_s), Consistent(), NumericallySound()]


def _nominal(converge_s: float = 90.0, **kw) -> List[Criterion]:
    return _standard(converge_s, **kw) + [NoFailsafe()]


# --------------------------------------------------------------------------- nominal

def city_loaded() -> VerifyScenario:
    s = _city()
    t = s.times()
    return VerifyScenario(
        "city_loaded", "City drive, fully loaded, load unknown at start",
        "Baseline. The car was loaded while switched off (4 passengers + 100 kg luggage, +1.19°). "
        "The filter starts at 0° and must find the load from normal stop-and-go driving. "
        "The lamp must not move during the initial standstill, where the load is not observable.",
        ScriptedDrive(s.manoeuvres, loads=[(0.0, "full_cabin_trunk")]),
        _nominal() + [LampQuiet(0.0, t["start"][1], 0.05, "the initial standstill")])


def city_no_camera() -> VerifyScenario:
    sc = city_loaded()
    return replace(sc, name="city_no_camera", title="City drive, fully loaded, without camera",
                   purpose="Same drive with the camera unavailable (night, unmarked road). IMU and wheel "
                           "speed alone must converge, with a longer time allowed.",
                   criteria=_nominal(converge_s=150.0), use_camera=False)


def motorway_cruise() -> VerifyScenario:
    s = Script().hold(10).to(120, 1.5, "launch").hold(200, "cruise").to(80, 1.5).hold(30).to(0, 2.0).hold(5)
    t = s.times()
    return VerifyScenario(
        "motorway_cruise", "Motorway: one launch, then 200 s at constant speed",
        "Almost no excitation after the launch. The estimate must be found during the launch and must "
        "not drift while nothing can be observed.",
        ScriptedDrive(s.manoeuvres, loads=[(0.0, "full_cabin")], roughness="B"),
        _nominal() + [LampQuiet(t["cruise"][0] + 60.0, t["cruise"][1], 0.10, "constant-speed cruise")])


def hill_route() -> VerifyScenario:
    s = (Script().hold(8).to(60, 2.0).hold(25, "flat").to(40, 1.5).hold(20, "climb").to(60, 1.5).hold(20)
         .to(30, 2.5, "brake_downhill").hold(15).to(60, 2.0).hold(25).to(0, 2.0).hold(5))
    t = s.times()
    grade = [(0, 0), (400, 0), (500, 8), (900, 8), (1000, 0), (1100, -8), (1500, -8), (1600, 0), (4000, 0)]
    return VerifyScenario(
        "hill_route", "8 % climb and descent, driver only",
        "Grade handling. A slope tilts the accelerometer exactly as a load would, and the filter must not "
        "mistake one for the other. A slope also really pitches the body relative to the road (load "
        "transfer, about ±0.15° here), and that part the lamp should follow. The aim error against the "
        "true pitch to the road must stay small while the car climbs, crests, descends and brakes.",
        ScriptedDrive(s.manoeuvres, grade=grade, loads=[(0.0, "driver_only")]),
        _nominal() + [AimWithin(30.0, t["_end"][0], 0.25, "the whole hill route")])


def parked_on_slope_loading() -> VerifyScenario:
    s = (Script().hold(8).to(60, 2.0).hold(15).to(30, 2.0).hold(10).to(60, 2.0).hold(20).to(0, 2.0)
         .hold(30, "parked").to(60, 1.8).hold(20).to(30, 2.0).hold(10).to(70, 1.5).hold(30).to(0, 2.0).hold(5))
    t = s.times()
    t_load = t["parked"][0] + 10.0
    grade = [(0, 0), (600, 0), (700, 10), (5000, 10)]
    return VerifyScenario(
        "parked_on_slope_loading", "Loading the car while parked on a 10 % slope",
        "The hardest static case: the car stops on a slope and is then loaded (4 passengers + luggage). "
        "The filter must attribute the tilt change to the load, not to the road, follow it while still "
        "parked, and keep the result when driving off.",
        ScriptedDrive(s.manoeuvres, grade=grade, loads=[(0.0, "driver_only"), (t_load, "full_cabin_trunk")]),
        _nominal() + [LoadStepTracked(t_load)])


def unload_at_stop() -> VerifyScenario:
    s = (Script().hold(8).to(60, 2.0).hold(15).to(20, 2.5).hold(8).to(70, 1.8).hold(20).to(30, 2.0).hold(10)
         .to(60, 2.0).hold(20).to(0, 2.0).hold(25, "stop").to(60, 2.0).hold(20).to(20, 2.0).hold(10)
         .to(60, 1.5).hold(25).to(0, 2.0).hold(5))
    t = s.times()
    t_unload = t["stop"][0] + 8.0
    return VerifyScenario(
        "unload_at_stop", "Maximum rear load removed at a stop",
        "Largest step (2.2°). After unloading the beam is too low; the lamp must come back up promptly so "
        "the driver does not lose range. The lamp's own slew rate (0.5°/s) sets the floor at about 5 s.",
        ScriptedDrive(s.manoeuvres, loads=[(0.0, "max_rear"), (t_unload, "driver_only")]),
        _nominal() + [LoadStepTracked(t_unload)])


def sporty_driving() -> VerifyScenario:
    s = Script().hold(8).to(60, 2.0).hold(15).to(20, 2.5).hold(8).to(60, 2.0).hold(15)
    s.to(30, 2.0, "block_start")
    for _ in range(5):
        s.to(90, 3.5).to(30, 6.0)
    s.hold(10, "block_end").to(60, 2.0).hold(20).to(0, 2.0).hold(5)
    t = s.times()
    return VerifyScenario(
        "sporty_driving", "Repeated hard acceleration and braking",
        "Dynamic immunity. Squat and dive reach ±1° here. A static leveling system must not chase them: "
        "the lamp has to stay put through five full-throttle / hard-braking cycles.",
        ScriptedDrive(s.manoeuvres, loads=[(0.0, "rear_passengers")]),
        _nominal() + [LampQuiet(t["block_start"][0], t["block_end"][1], 0.15, "the acceleration / braking block")])


def rough_road() -> VerifyScenario:
    s = _city()
    return VerifyScenario(
        "rough_road", "City drive on a rough road (ISO 8608 class C)",
        "Noise robustness. Vertical vibration is roughly ten times that of a good road and far above the "
        "load signal. Accuracy may degrade (relaxed limits) but the estimate must stay stable and honest.",
        ScriptedDrive(s.manoeuvres, loads=[(0.0, "full_cabin")], roughness="C"),
        _nominal(converge_s=120.0, mae=0.25, p95=0.45))


def curvy_road() -> VerifyScenario:
    s = Script().hold(8).to(50, 2.0).hold(20).to(30, 2.0).hold(10).to(60, 1.8).hold(25).to(25, 2.5).hold(10)
    s.to(55, 1.8).hold(25).to(0, 2.0).hold(8).to(50, 2.0).hold(30).to(0, 2.0).hold(5)
    curv = [(0, 0)]
    for i in range(30):
        curv += [(150 + 120 * i, (1 if i % 2 else -1) / 90.0), (150 + 120 * i + 60, 0.0)]
    return VerifyScenario(
        "curvy_road", "Winding road with left and right curves of 90 m radius",
        "Lateral dynamics. Lateral acceleration reaches 0.3 g, the body rolls and the yaw rate is high. "
        "None of that may leak into the pitch estimate.",
        ScriptedDrive(s.manoeuvres, curvature=curv, loads=[(0.0, "trunk_heavy")], roughness="B"), _nominal())


def urban_turns() -> VerifyScenario:
    s = Script().hold(15, "start").to(40, 2.0).hold(8)
    radii = {}
    for i, (kmh, radius, secs) in enumerate([(22, 15, 9), (25, -20, 12), (20, -13, 8), (22, 15, 9), (28, 25, 14),
                                             (20, -13, 8)]):
        s.to(kmh, 2.0).hold(secs, f"turn{i}").to(50 if i % 2 else 60, 2.0).hold(10)
        radii[f"turn{i}"] = radius
        if i == 2:
            s.to(0, 2.5).hold(8, "stop")
    s.to(0, 2.5).hold(5)
    d = s.distances()
    curv = [(0.0, 0.0)]
    for name, radius in radii.items():
        d0, d1 = d[name]
        # The curve starts while still braking and ends while already accelerating.
        curv += [(d0 - 12.0, 0.0), (d0 - 2.0, 1.0 / radius), (d1 + 6.0, 1.0 / radius), (d1 + 16.0, 0.0)]
    return VerifyScenario(
        "urban_turns", "City drive with junction turns and roundabouts (13 to 25 m radius)",
        "Tight turns at low speed. The yaw rate reaches 25 °/s, so the centripetal acceleration at the "
        "IMU (its distance from the rear axle times the yaw rate squared) and the rear tyres' side-slip "
        "appear in the longitudinal accelerometer, always with the same sign. The turns overlap with "
        "braking and accelerating. None of that may be read as load.",
        ScriptedDrive(s.manoeuvres, curvature=curv, loads=[(0.0, "full_cabin_trunk")]), _nominal())


def soft_vehicle() -> VerifyScenario:
    s = _city()
    veh = HalfCarParams(k_f=33e3, k_r=36e3, c_f=2.6e3, c_r=2.8e3, h=0.68, anti_dive=0.3, anti_squat=0.1,
                        m_s=1750.0, i_y=2900.0)
    return VerifyScenario(
        "soft_vehicle", "Different vehicle: soft, tall SUV, same filter tuning",
        "Parameter robustness. The plant is softer and taller than the filter's nominal model: pitch "
        f"frequency {veh.pitch_freq_hz:.2f} Hz instead of 1.4 Hz, squat {veh.pitch_gradient_deg_per_g():.1f}°/g "
        "instead of 2.5°/g. The filter is not retuned.",
        ScriptedDrive(s.manoeuvres, loads=[(0.0, "full_cabin_trunk")], vehicle=veh), _nominal())


def lever_arm_miscalibrated() -> VerifyScenario:
    sc = city_loaded()
    return replace(sc, name="lever_arm_miscalibrated", title="City drive, IMU position known only to 25 %",
                   purpose="Calibration sensitivity. The filter removes the effect of the IMU's distance from "
                           "the pitch axis (0.9 m here). In this run it is told 0.68 m instead.",
                   criteria=_nominal(), lever_arm_scale=0.75)


# --------------------------------------------------------------------------- faults

def _fault(name: str, title: str, purpose: str, faults_fn: Callable[[Dict], List[Fault]],
           criteria_fn: Callable[[List[Fault]], List[Criterion]]) -> VerifyScenario:
    s = _city()
    faults = faults_fn(s.times())
    return VerifyScenario(name, title, purpose, ScriptedDrive(s.manoeuvres, loads=[(0.0, "full_cabin_trunk")]),
                          criteria_fn(faults), faults=faults, group="fault")


def _isolate_hold_recover(detect_s: float) -> Callable[[List[Fault]], List[Criterion]]:
    """A critical sensor fails for a while: isolate it, hold the beam, resume afterwards."""
    def build(faults: List[Fault]) -> List[Criterion]:
        _, _, t0, t1, _ = faults[0]
        return _standard() + [FaultDetected(t0, detect_s), BeamNotRaised(t0, t1 + 10.0), Recovers(t1)]
    return build


def _ride_through(faults: List[Fault]) -> List[Criterion]:
    """The fault must not matter: accuracy is required during it as well."""
    return [Converges(), SteadyAccuracy(recover_s=None), NoGlare(recover_s=None), ActuatorTravel(recover_s=None),
            LevelingContinues(), NumericallySound()]


def fault_wheel_speed_zero() -> VerifyScenario:
    return _fault(
        "fault_wheel_speed_zero", "Wheel speed reads zero for 20 s while driving",
        "The most dangerous input fault: a filter that believes the car stands still applies zero-velocity "
        "updates and reads road slope as load. The fault must be isolated, the beam must not rise, and "
        "leveling must resume.",
        lambda t: [("v_wheel", "value", t["cruise2"][0] + 2.0, t["cruise2"][0] + 22.0, 0.0)],
        _isolate_hold_recover(2.0))


def fault_imu_outage() -> VerifyScenario:
    return _fault(
        "fault_imu_outage", "IMU delivers nothing for 15 s",
        "Loss of the critical sensor. The lamp must hold (never rise) and leveling must resume by itself.",
        lambda t: [("imu", "missing", t["cruise2"][0] + 2.0, t["cruise2"][0] + 17.0, 0.0)],
        _isolate_hold_recover(2.0))


def fault_accel_z_frozen() -> VerifyScenario:
    return _fault(
        "fault_accel_z_frozen", "Vertical accelerometer frozen for 30 s",
        "A frozen channel still looks like a valid number. It must be recognised by its missing noise.",
        lambda t: [("az", "frozen", t["stop2"][0] - 10.0, t["stop2"][0] + 20.0, 0.0)],
        _isolate_hold_recover(2.0))


def fault_gyro_offset() -> VerifyScenario:
    return _fault(
        "fault_gyro_offset", "Pitch gyro offset of 20 °/s for the rest of the trip",
        "Degraded mode. The gyro is not needed for the load pitch, so leveling must simply continue "
        "without it: no fail-safe, no loss of accuracy.",
        lambda t: [("gy", "offset", t["stop1"][1] + 5.0, 1e9, 20.0)],
        lambda f: _ride_through(f) + [FaultDetected(f[0][2], 6.0)])


def fault_glitches() -> VerifyScenario:
    return _fault(
        "fault_glitches", "Random glitches on 0.5 % of the IMU samples, whole trip",
        "EMC-like disturbance. Single wrong samples must be removed without any effect on leveling.",
        lambda t: [(ch, "glitch", 0.0, 1e9, 0.005) for ch in ("ax", "az", "gy")], _ride_through)


def fault_camera_wrong() -> VerifyScenario:
    return _fault(
        "fault_camera_wrong", "Camera reports a pitch that is 1.2° wrong for 60 s",
        "A plausible-looking but wrong auxiliary input. No simple check can reject it; the filter must "
        "notice that it contradicts the IMU and stop using it.",
        lambda t: [("camera", "wrong", t["cruise"][0], t["cruise"][0] + 60.0, 1.2)], _ride_through)


SCENARIOS: Dict[str, Callable[[], VerifyScenario]] = {f.__name__: f for f in (
    city_loaded, city_no_camera, motorway_cruise, hill_route, parked_on_slope_loading, unload_at_stop,
    sporty_driving, rough_road, curvy_road, urban_turns, soft_vehicle, lever_arm_miscalibrated,
    fault_wheel_speed_zero, fault_imu_outage, fault_accel_z_frozen, fault_gyro_offset, fault_glitches,
    fault_camera_wrong)}

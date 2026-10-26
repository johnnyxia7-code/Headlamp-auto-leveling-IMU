"""Headlamp leveling controller (quasi-static automatic leveling).

Inputs every ``update_period_s``:
    * the window estimate of the static load pitch from a model (deg, NaN if none)
    * the excitation of that window (std of longitudinal acceleration)
    * standstill flag and the accelerometer tilt atan2(f_x, f_z) while stopped
    * a sensor-fault flag

Output: headlamp correction angle (deg, beam lowered by this amount) and
the equivalent stepper position.

Strategy
    driving     accept estimates only with enough excitation; fuse them with
                a first-order filter whose gain starts high (fast initial
                convergence) and settles to ``dt / filter_tau_s``
    standstill  the road grade cannot change while stopped, so any change in
                accelerometer tilt is a load change (doors, passengers,
                luggage): track it directly relative to the tilt at stop entry
    actuator    deadband + slew-rate limit + mechanical range clamp
    fault       move to the fail-safe position and hold
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import ControlConfig


@dataclass
class ControllerOutput:
    correction_deg: float       # actual actuator position (beam lowered by this angle)
    target_deg: float           # requested position after deadband
    estimate_deg: float         # fused load-pitch estimate
    steps: int                  # stepper motor position
    fault: bool


class LevelingController:
    def __init__(self, cfg: ControlConfig, initial_estimate_deg: float = 0.0):
        self.cfg = cfg
        self.reset(initial_estimate_deg)

    def reset(self, initial_estimate_deg: float = 0.0) -> None:
        self.estimate = float(initial_estimate_deg)
        self.target = float(np.clip(initial_estimate_deg, *self.cfg.range_deg))
        self.position = self.target
        self.n_accepted = 0
        self._stop_ref_tilt: Optional[float] = None
        self._stop_ref_estimate = self.estimate
        self.fault = False

    def _accept(self, est: float, weight: float, dt: float) -> None:
        self.n_accepted += 1
        alpha = max(dt / (self.cfg.filter_tau_s + dt), 1.0 / (self.n_accepted + 1)) * weight
        self.estimate += min(alpha, 1.0) * (est - self.estimate)

    def update(self, dt: float, estimate_deg: float = float("nan"), excitation: float = 0.0,
               standstill: bool = False, tilt_deg: float = float("nan"), fault: bool = False,
               estimate_std_deg: float = float("nan")) -> ControllerOutput:
        """One controller cycle.

        Window estimators pass ``excitation``; the EKF passes ``estimate_std_deg``
        instead, in which case its (already filtered) estimate is used directly
        whenever its 1-sigma is below ``max_estimate_std_deg``.
        """
        c = self.cfg
        self.fault = self.fault or fault
        if self.fault:
            self.target = c.failsafe_deg
        else:
            if np.isfinite(estimate_std_deg):
                if np.isfinite(estimate_deg) and estimate_std_deg <= c.max_estimate_std_deg:
                    self.estimate = float(estimate_deg)
            elif standstill and np.isfinite(tilt_deg):
                if self._stop_ref_tilt is None:
                    self._stop_ref_tilt = tilt_deg
                    self._stop_ref_estimate = self.estimate
                self.estimate = self._stop_ref_estimate + (tilt_deg - self._stop_ref_tilt)
            else:
                self._stop_ref_tilt = None
                if np.isfinite(estimate_deg) and excitation >= c.min_excitation_ms2:
                    weight = min(1.0, excitation / (2.0 * c.min_excitation_ms2))
                    self._accept(float(estimate_deg), weight, dt)
            desired = float(np.clip(self.estimate, *c.range_deg))
            if abs(desired - self.target) > c.deadband_deg:
                self.target = desired

        step = c.max_rate_deg_s * dt
        self.position += float(np.clip(self.target - self.position, -step, step))
        return ControllerOutput(self.position, self.target, self.estimate,
                                int(round(self.position * c.steps_per_deg)), self.fault)

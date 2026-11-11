"""Acceptance criteria for the verification scenarios.

Each criterion states a requirement on the behaviour of the whole chain
(input layer -> EKF -> controller -> lamp) and measures it on one run.

Sign convention of the aim error: ``err = true static load pitch - applied
correction``. Positive means the beam points higher than intended (glare),
negative means it points lower (reduced range).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RunData:
    """Everything a criterion may look at."""
    t: np.ndarray               # controller time base (s)
    err: np.ndarray             # aim error (deg)
    correction: np.ndarray      # lamp correction (deg, beam lowered by this angle)
    truth: np.ndarray           # true static load pitch (deg)
    failsafe: np.ndarray        # lamp in fail-safe
    t_est: np.ndarray           # estimator time base (s)
    est: np.ndarray             # estimated load pitch (deg)
    est_std: np.ndarray         # its 1-sigma (deg)
    est_truth: np.ndarray
    t_raw: np.ndarray           # sensor time base (s)
    speed: np.ndarray           # true speed (m/s)
    guard_failed: np.ndarray    # any channel FAILED (sensor time base)
    t_drive: float              # first time the car moves
    load_times: Sequence[float]
    fault_windows: Sequence[Tuple[float, float]]
    numeric_resets: int

    def settled(self, converge_s: float, step_grace_s: float, recover_s: Optional[float]) -> np.ndarray:
        """Controller samples at which steady-state accuracy is required.

        ``recover_s`` is the time allowed after a fault ends; ``None`` means the
        requirement also holds while a fault is present (ride-through).
        """
        m = self.t >= self.t_drive + converge_s
        for tc in self.load_times:
            m &= ~((self.t >= tc) & (self.t < tc + step_grace_s))
        if recover_s is not None:
            for t0, t1 in self.fault_windows:
                m &= ~((self.t >= t0) & (self.t < t1 + recover_s))
        return m


@dataclass
class CheckResult:
    name: str
    requirement: str
    measured: str
    passed: bool


class Criterion:
    name = "criterion"

    def evaluate(self, d: RunData) -> CheckResult:   # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class Converges(Criterion):
    """From an unknown load, the aim error must come within tolerance and stay there."""
    within_s: float = 90.0
    tol_deg: float = 0.25
    name = "convergence"

    def evaluate(self, d: RunData) -> CheckResult:
        t_end = min([tc for tc in d.load_times if tc > d.t_drive] + [t0 for t0, _ in d.fault_windows]
                    + [float(d.t[-1]) + 1.0])
        m = (d.t >= d.t_drive) & (d.t < t_end)
        bad = np.flatnonzero(m & (np.abs(d.err) > self.tol_deg))
        t_conv = 0.0 if bad.size == 0 else float(d.t[min(bad[-1] + 1, d.t.size - 1)] - d.t_drive)
        ok = t_conv <= self.within_s and (bad.size == 0 or d.t[bad[-1]] < t_end - 1.5)
        return CheckResult(self.name, f"|aim error| ≤ {self.tol_deg}° within {self.within_s:.0f} s of driving off, "
                           "and stays there", f"{t_conv:.0f} s", bool(ok))


@dataclass
class SteadyAccuracy(Criterion):
    mae_deg: float = 0.15
    p95_deg: float = 0.30
    converge_s: float = 90.0
    step_grace_s: float = 15.0
    recover_s: Optional[float] = 60.0
    name = "steady-state accuracy"

    def evaluate(self, d: RunData) -> CheckResult:
        m = d.settled(self.converge_s, self.step_grace_s, self.recover_s)
        if not m.any():
            return CheckResult(self.name, "n/a", "no settled samples", False)
        e = np.abs(d.err[m])
        mae, p95 = float(e.mean()), float(np.percentile(e, 95))
        return CheckResult(self.name, f"aim error MAE ≤ {self.mae_deg}° and P95 ≤ {self.p95_deg}° once settled",
                           f"MAE {mae:.3f}°, P95 {p95:.3f}°", mae <= self.mae_deg and p95 <= self.p95_deg)


@dataclass
class NoGlare(Criterion):
    """One-sided: the beam must never sit clearly above its intended aim."""
    limit_deg: float = 0.35
    converge_s: float = 90.0
    step_grace_s: float = 15.0
    recover_s: Optional[float] = 60.0
    name = "no glare"

    def evaluate(self, d: RunData) -> CheckResult:
        m = d.settled(self.converge_s, self.step_grace_s, self.recover_s)
        worst = float(d.err[m].max()) if m.any() else float("nan")
        return CheckResult(self.name, f"beam never more than {self.limit_deg}° above target once settled",
                           f"max +{max(worst, 0.0):.3f}°", bool(worst <= self.limit_deg))


@dataclass
class LampQuiet(Criterion):
    """The lamp must not move during a phase in which the load does not change."""
    t0: float
    t1: float
    max_move_deg: float = 0.10
    what: str = "phase"
    name = "lamp steady"

    def evaluate(self, d: RunData) -> CheckResult:
        m = (d.t >= self.t0) & (d.t <= self.t1)
        move = float(np.ptp(d.correction[m])) if m.any() else float("nan")
        return CheckResult(f"{self.name} ({self.what})",
                           f"lamp moves ≤ {self.max_move_deg}° during {self.what} ({self.t0:.0f}–{self.t1:.0f} s)",
                           f"{move:.3f}°", bool(move <= self.max_move_deg))


@dataclass
class LoadStepTracked(Criterion):
    t_event: float
    within_s: float = 12.0
    tol_deg: float = 0.25
    hold_s: float = 20.0
    name = "load change tracked"

    def evaluate(self, d: RunData) -> CheckResult:
        m = (d.t >= self.t_event) & (d.t <= self.t_event + self.within_s + self.hold_s)
        bad = np.flatnonzero(m & (np.abs(d.err) > self.tol_deg))
        t_track = 0.0 if bad.size == 0 else float(d.t[min(bad[-1] + 1, d.t.size - 1)] - self.t_event)
        return CheckResult(self.name, f"|aim error| ≤ {self.tol_deg}° within {self.within_s:.0f} s of a load change "
                           f"at standstill (t = {self.t_event:.0f} s)", f"{t_track:.0f} s", t_track <= self.within_s)


@dataclass
class NoFailsafe(Criterion):
    name = "no false alarm"

    def evaluate(self, d: RunData) -> CheckResult:
        n_fs, n_g = int(d.failsafe.sum()), int(d.guard_failed.sum())
        return CheckResult(self.name, "no fail-safe and no sensor disqualified on healthy data",
                           "none" if n_fs + n_g == 0 else f"fail-safe {n_fs} s, sensor failed {n_g / 100:.1f} s",
                           n_fs + n_g == 0)


@dataclass
class LevelingContinues(Criterion):
    """Degraded mode: the fault must not put the lamp into fail-safe at all."""
    name = "leveling continues"

    def evaluate(self, d: RunData) -> CheckResult:
        n = int(d.failsafe.sum())
        return CheckResult(self.name, "lamp never enters fail-safe", "never" if n == 0 else f"{n} s", n == 0)


@dataclass
class FaultDetected(Criterion):
    t0: float
    within_s: float = 5.0
    name = "fault isolated"

    def evaluate(self, d: RunData) -> CheckResult:
        k = np.flatnonzero(d.guard_failed & (d.t_raw >= self.t0))
        lat = float(d.t_raw[k[0]] - self.t0) if k.size else float("inf")
        return CheckResult(self.name, f"faulty sensor disqualified within {self.within_s:.0f} s",
                           f"{lat:.1f} s" if np.isfinite(lat) else "not detected", lat <= self.within_s)


@dataclass
class BeamNotRaised(Criterion):
    """During a fault the beam must not be raised relative to where it was."""
    t0: float
    t1: float
    limit_deg: float = 0.10
    name = "beam not raised during fault"

    def evaluate(self, d: RunData) -> CheckResult:
        before = d.correction[d.t < self.t0]
        ref = float(before[-1]) if before.size else 0.0
        m = (d.t >= self.t0) & (d.t <= self.t1)
        rise = float(ref - d.correction[m].min()) if m.any() else 0.0
        return CheckResult(self.name, f"correction drops by ≤ {self.limit_deg}° while the fault is present",
                           f"{max(rise, 0.0):.3f}°", rise <= self.limit_deg)


@dataclass
class Recovers(Criterion):
    t_end: float
    within_s: float = 60.0
    tol_deg: float = 0.25
    name = "recovery after fault"

    def evaluate(self, d: RunData) -> CheckResult:
        m = d.t >= self.t_end
        bad = np.flatnonzero(m & ((np.abs(d.err) > self.tol_deg) | d.failsafe))
        t_rec = 0.0 if bad.size == 0 else float(d.t[min(bad[-1] + 1, d.t.size - 1)] - self.t_end)
        ok = t_rec <= self.within_s and not bool(d.failsafe[-1]) and (bad.size == 0 or bad[-1] < d.t.size - 1)
        return CheckResult(self.name, f"leveling back within {self.tol_deg}° and out of fail-safe within "
                           f"{self.within_s:.0f} s after the fault ends", f"{t_rec:.0f} s", bool(ok))


@dataclass
class Consistent(Criterion):
    """The filter's own uncertainty must be honest: errors inside 3 sigma."""
    k_sigma: float = 3.0
    min_fraction: float = 0.95
    after_s: float = 30.0
    name = "filter consistency"

    def evaluate(self, d: RunData) -> CheckResult:
        m = d.t_est >= d.t_drive + self.after_s
        for t0, t1 in d.fault_windows:
            m &= ~((d.t_est >= t0) & (d.t_est < t1 + 30.0))
        if not m.any():
            return CheckResult(self.name, "n/a", "no samples", False)
        frac = float(np.mean(np.abs(d.est[m] - d.est_truth[m]) <= self.k_sigma * d.est_std[m]))
        return CheckResult(self.name, f"estimate error within {self.k_sigma:.0f}σ at least "
                           f"{100 * self.min_fraction:.0f} % of the time", f"{100 * frac:.1f} %",
                           frac >= self.min_fraction)


@dataclass
class ActuatorTravel(Criterion):
    """No hunting: total lamp travel per minute while the load is constant."""
    max_deg_per_min: float = 0.5
    converge_s: float = 90.0
    step_grace_s: float = 15.0
    recover_s: Optional[float] = 60.0
    name = "actuator activity"

    def evaluate(self, d: RunData) -> CheckResult:
        m = d.settled(self.converge_s, self.step_grace_s, self.recover_s)
        both = m[1:] & m[:-1]
        if not both.any():
            return CheckResult(self.name, "n/a", "no settled samples", False)
        travel = float(np.abs(np.diff(d.correction))[both].sum())
        rate = travel / (both.sum() * float(np.median(np.diff(d.t))) / 60.0)
        return CheckResult(self.name, f"lamp travel ≤ {self.max_deg_per_min}°/min while the load is constant",
                           f"{rate:.2f}°/min", rate <= self.max_deg_per_min)


@dataclass
class NumericallySound(Criterion):
    name = "numerically sound"

    def evaluate(self, d: RunData) -> CheckResult:
        ok = bool(np.all(np.isfinite(d.est)) and np.all(np.isfinite(d.correction)) and d.numeric_resets == 0)
        return CheckResult(self.name, "all outputs finite, no filter reset", "ok" if ok else
                           f"{d.numeric_resets} resets" if d.numeric_resets else "non-finite output", ok)


def evaluate_all(criteria: List[Criterion], d: RunData) -> List[CheckResult]:
    return [c.evaluate(d) for c in criteria]

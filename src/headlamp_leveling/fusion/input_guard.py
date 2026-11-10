"""Resilient sensor-input layer in front of the EKF.

Every raw sample of every channel passes through the same pipeline before the
filter may use it (details and rationale: ``docs/input_guard.md``)::

    raw sample
      1. validity      finite?  inside the sensor's physical range (not saturated)?
      2. stuck         does it still change at all? (a live MEMS signal always has noise)
      3. spike         causal Hampel test (median / MAD of the last valid samples);
                       wheel speed: physically possible change since the last valid sample
      4. plausibility  cross-sensor physics: |f| = g at standstill, no rotation at
                       standstill, no sustained pitch rate, accel vs. wheel-speed
                       derivative, vibration while the wheel speed reads zero, ...
      5. debounce      per-channel health OK -> DEGRADED -> FAILED, with healing
      6. reaction      invalid samples become NaN (the EKF skips that update and
                       coasts on the remaining sensors); a critical channel FAILED
                       for longer than ``failsafe_after_s`` requests the lamp fail-safe

The guard never invents data: it does not hold or extrapolate values. A missing
measurement is simply not used, which is the statistically correct treatment in
a Kalman filter.

It is written sample-by-sample with scalar state only (no arrays growing over
time), so it ports directly to the embedded target.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from .. import G
from ..config import InputGuardConfig

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz", "v_wheel", "camera")
IDX = {c: i for i, c in enumerate(CHANNELS)}
# Without these the load pitch is not observable. The pitch gyro is deliberately not
# in the list: the EKF keeps its accuracy on accelerometer + wheel speed alone (the
# gyro mainly speeds up tracking of grade changes), so its loss is a degraded mode.
CRITICAL = ("ax", "az", "v_wheel")
NAN = float("nan")


class Health(IntEnum):
    OK = 0
    DEGRADED = 1      # samples currently rejected, fault not (yet) qualified
    FAILED = 2        # qualified fault: channel not used until healed


# Reasons a sample / channel was rejected (bit flags, so they can be combined and counted).
MISSING, RANGE, STUCK, SPIKE, PLAUSIBILITY = 1, 2, 4, 8, 16
REASON_NAMES = {MISSING: "missing", RANGE: "range", STUCK: "stuck", SPIKE: "spike",
                PLAUSIBILITY: "plausibility"}


def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


class _Debounce:
    """Time-based fault qualification and healing."""

    __slots__ = ("qualify_s", "heal_s", "bad_t", "good_t", "failed")

    def __init__(self, qualify_s: float, heal_s: float):
        self.qualify_s, self.heal_s = qualify_s, heal_s
        self.bad_t = self.good_t = 0.0
        self.failed = False

    def update(self, bad: bool, dt: float) -> bool:
        if bad:
            self.bad_t += dt
            self.good_t = 0.0
            if self.bad_t >= self.qualify_s:
                self.failed = True
        else:
            self.good_t += dt
            # Bad time decays slower than it builds up, so an intermittent fault
            # (e.g. 50 % of samples bad) still qualifies.
            self.bad_t = max(0.0, self.bad_t - dt * self.qualify_s / self.heal_s)
            if self.failed and self.good_t >= self.heal_s:
                self.failed = False
                self.bad_t = 0.0
        return self.failed


class _Channel:
    """Per-channel sample checks (validity, stuck, spike) and health."""

    def __init__(self, name: str, cfg: InputGuardConfig):
        self.name = name
        self.cfg = cfg
        kind = "accel" if name[0] == "a" else "gyro" if name[0] == "g" else name
        self.kind = kind
        if kind == "accel":
            self.lo, self.hi = -cfg.accel_range_ms2, cfg.accel_range_ms2
            self.spike_min, self.stuck_eps = cfg.spike_min_accel_ms2, cfg.stuck_eps_accel_ms2
        elif kind == "gyro":
            self.lo, self.hi = -cfg.gyro_range_dps, cfg.gyro_range_dps
            self.spike_min, self.stuck_eps = cfg.spike_min_gyro_dps, cfg.stuck_eps_gyro_dps
        elif kind == "v_wheel":
            self.lo, self.hi = cfg.speed_range_ms
            self.spike_min, self.stuck_eps = 0.3, cfg.stuck_eps_speed_ms
        else:  # camera
            self.lo, self.hi = -cfg.camera_range_deg, cfg.camera_range_deg
            self.spike_min, self.stuck_eps = 0.0, 1e-9
        self.window: Deque[float] = deque(maxlen=max(3, cfg.hampel_window))
        self.last_raw = NAN
        self.same_t = 0.0
        self.same_n = 0
        self.spike_run = 0
        self.last_valid = NAN
        self.since_valid = 0.0
        self.debounce = _Debounce(cfg.qualify_s, cfg.wheel_heal_s if kind == "v_wheel" else cfg.heal_s)
        self.plaus_failed = False
        self.health = Health.OK
        self.reason = 0
        self.counts: Dict[int, int] = {r: 0 for r in REASON_NAMES}

    # ------------------------------------------------------------ sample check
    def check(self, x: float, dt: float) -> Tuple[float, int]:
        """Returns (valid value or NaN, reason flags of this sample)."""
        self.since_valid += dt
        if not math.isfinite(x):
            return NAN, MISSING
        if x < self.lo or x > self.hi:
            return NAN, RANGE

        # Stuck: identical readings although sensor noise must be present. A wheel
        # speed of exactly zero is a legitimate constant (standstill).
        if abs(x - self.last_raw) <= self.stuck_eps:
            self.same_t += dt
            self.same_n += 1
        else:
            self.same_t, self.same_n = 0.0, 0
        self.last_raw = x
        if self.kind == "camera":
            if self.same_n >= self.cfg.camera_stuck_frames:
                return NAN, STUCK
        elif self.same_t >= self.cfg.stuck_s and not (self.kind == "v_wheel" and x == 0.0):
            return NAN, STUCK

        if self.kind == "v_wheel":
            # Physically possible change since the last valid sample. The allowance
            # grows with elapsed time, so the signal is re-accepted after a real gap.
            if math.isfinite(self.last_valid):
                allow = self.cfg.wheel_max_accel_ms2 * self.since_valid + self.spike_min
                if abs(x - self.last_valid) > allow:
                    return NAN, SPIKE
        elif self.kind != "camera" and len(self.window) >= 3:
            med = _median(self.window)
            mad = _median([abs(v - med) for v in self.window])
            thr = max(self.cfg.hampel_k * 1.4826 * mad, self.spike_min)
            if abs(x - med) > thr:
                self.spike_run += 1
                if self.spike_run <= 3:
                    return NAN, SPIKE
                self.window.clear()          # persistent: a real step, not a spike
            self.spike_run = 0

        self.window.append(x)
        self.last_valid = x
        self.since_valid = 0.0
        return x, 0

    # ------------------------------------------------------------ health
    def finish(self, reason: int, dt: float, absence_is_fault: bool = True) -> None:
        sample_bad = bool(reason & ~PLAUSIBILITY) and (absence_is_fault or reason != MISSING)
        failed = self.debounce.update(sample_bad, dt)
        if reason:
            for r in REASON_NAMES:
                if reason & r:
                    self.counts[r] += 1
        self.reason = reason | (PLAUSIBILITY if self.plaus_failed else 0)
        if failed or self.plaus_failed:
            self.health = Health.FAILED
        elif sample_bad or self.debounce.bad_t > 0.0:
            self.health = Health.DEGRADED
        else:
            self.health = Health.OK


class _Lowpass:
    __slots__ = ("tau", "y")

    def __init__(self, tau: float):
        self.tau, self.y = tau, NAN

    def update(self, x: float, dt: float) -> float:
        if math.isfinite(x):
            self.y = x if not math.isfinite(self.y) else self.y + min(1.0, dt / self.tau) * (x - self.y)
        return self.y


@dataclass
class GuardOutput:
    values: List[float]                     # CHANNELS order; NaN = do not use
    health: Dict[str, Health]
    reasons: Dict[str, int]
    degraded: bool                          # any channel not OK
    critical_failed: bool                   # a channel the estimate depends on is FAILED
    failsafe: bool                          # ... for longer than failsafe_after_s
    newly_failed: Tuple[str, ...] = ()      # channels disqualified in this sample
    recovered: Tuple[str, ...] = ()         # channels that healed in this sample
    outage_s: float = 0.0                   # duration of the outage that just ended

    def value(self, name: str) -> float:
        return self.values[IDX[name]]


class SensorInputGuard:
    def __init__(self, cfg: InputGuardConfig):
        self.cfg = cfg
        self.ch: Dict[str, _Channel] = {c: _Channel(c, cfg) for c in CHANNELS}
        tau = cfg.plaus_lp_tau_s
        self._lp = {k: _Lowpass(tau) for k in ("ax", "az", "gx", "gy", "v")}
        self._lp_fast = {k: _Lowpass(0.5) for k in ("gx", "gy", "gz")}
        self._ax_fast = _Lowpass(0.3)
        self._prev_v_lp = NAN
        self._prev_az = NAN
        self._az_noise_var = 0.0
        self._still_t = 0.0
        self._ax_still_ref = NAN
        # One debounce per plausibility rule; value = channels it disqualifies.
        q, h = cfg.plaus_qualify_s, cfg.heal_s
        self._rules: Dict[str, Tuple[_Debounce, Tuple[str, ...]]] = {
            "still_norm": (_Debounce(q, h), ("ax", "az")),
            "still_gyro_x": (_Debounce(q, h), ("gx",)),
            "still_gyro_y": (_Debounce(q, h), ("gy",)),
            "still_gyro_z": (_Debounce(q, h), ("gz",)),
            "moving_at_zero_speed": (_Debounce(min(q, 1.0), h), ("v_wheel",)),
            "az_level": (_Debounce(q, h), ("az",)),
            "gyro_mean_x": (_Debounce(q, h), ("gx",)),
            "gyro_mean_y": (_Debounce(q, h), ("gy",)),
            "accel_vs_wheel": (_Debounce(q, h), ("ax", "v_wheel")),
        }
        self.rule_counts: Dict[str, int] = {r: 0 for r in self._rules}
        self._critical_failed_t = 0.0
        self._cam_gap = 0.0
        self._outage_t: Dict[str, float] = {c: 0.0 for c in CHANNELS}
        self.time = 0.0

    # ------------------------------------------------------------ plausibility
    def _plausibility(self, v: Dict[str, float], dt: float) -> None:
        """Cross-sensor rules on the sample-valid values; sets ``plaus_failed`` per channel."""
        c = self.cfg
        fin = math.isfinite
        ax, ay, az, gx, gy, gz, vw = (v[k] for k in ("ax", "ay", "az", "gx", "gy", "gz", "v_wheel"))

        ax_lp = self._lp["ax"].update(ax, dt)
        az_lp = self._lp["az"].update(az, dt)
        gx_lp = self._lp["gx"].update(gx, dt)
        gy_lp = self._lp["gy"].update(gy, dt)
        v_lp = self._lp["v"].update(vw, dt)
        gf = {k: self._lp_fast[k].update(v[k], dt) for k in ("gx", "gy", "gz")}
        ax_fast = self._ax_fast.update(ax, dt)
        a_wheel = (v_lp - self._prev_v_lp) / dt if fin(v_lp) and fin(self._prev_v_lp) else NAN
        self._prev_v_lp = v_lp
        still = fin(vw) and vw < 0.05
        self._still_t = self._still_t + dt if still else 0.0
        settled = self._still_t >= 1.5            # brake dive and its rebound are over
        moving = fin(vw) and vw > 3.0
        # Vibration (white-noise level of az) measured only from 1 s into the stop, so
        # the road vibration from before the stop does not leak into it.
        if self._still_t < 1.0:
            self._az_noise_var = 0.0
        elif fin(az) and fin(self._prev_az):
            self._az_noise_var += min(1.0, dt / 0.5) * (0.5 * (az - self._prev_az) ** 2 - self._az_noise_var)
        self._prev_az = az
        if settled and not fin(self._ax_still_ref):
            self._ax_still_ref = ax_fast
        if not still:
            self._ax_still_ref = NAN

        bad: Dict[str, Optional[bool]] = {r: None for r in self._rules}   # None = rule not applicable
        if settled:
            if fin(ax) and fin(ay) and fin(az):
                bad["still_norm"] = abs(math.sqrt(ax * ax + ay * ay + az * az) - G) > c.still_norm_tol_ms2
            for k, r in (("gx", "still_gyro_x"), ("gy", "still_gyro_y"), ("gz", "still_gyro_z")):
                if fin(gf[k]):
                    bad[r] = abs(gf[k]) > c.still_gyro_max_dps
            # The wheel speed reads zero, but the body accelerates or shakes like a moving car.
            moving_signs = []
            if fin(ax_fast) and fin(self._ax_still_ref):
                moving_signs.append(abs(ax_fast - self._ax_still_ref) > c.still_ax_change_ms2)
            if fin(az):
                moving_signs.append(math.sqrt(self._az_noise_var) > c.still_vibration_max_ms2)
            if moving_signs:
                bad["moving_at_zero_speed"] = any(moving_signs)
        elif fin(vw) and vw > 0.5:
            bad["moving_at_zero_speed"] = False   # the sensor reports motion again: it is alive
        if moving:
            if fin(az_lp):
                bad["az_level"] = abs(az_lp - G) > c.az_dev_max_ms2
            if fin(gx_lp):
                bad["gyro_mean_x"] = abs(gx_lp) > c.gyro_mean_max_dps
            if fin(gy_lp):
                bad["gyro_mean_y"] = abs(gy_lp) > c.gyro_mean_max_dps
            if fin(ax_lp) and fin(a_wheel):
                bad["accel_vs_wheel"] = abs(ax_lp - a_wheel) > c.accel_wheel_mismatch_ms2

        failed_channels = set()
        for rule, (deb, channels) in self._rules.items():
            b = bad[rule]
            if b is None:
                # Not applicable now: keep a qualified fault (it can only heal while the
                # rule is being checked), but do not accumulate towards a new one.
                if not deb.failed:
                    deb.bad_t = 0.0
            else:
                was = deb.failed
                deb.update(b, dt)
                if deb.failed and not was:
                    self.rule_counts[rule] += 1
            if deb.failed:
                failed_channels.update(channels)
        for name, ch in self.ch.items():
            ch.plaus_failed = name in failed_channels

    # ------------------------------------------------------------ main entry
    def process(self, sample: Sequence[float], dt: float) -> GuardOutput:
        """Check one raw sample (CHANNELS order, NaN = not received this cycle)."""
        c = self.cfg
        self.time += dt
        valid: Dict[str, float] = {}
        reasons: Dict[str, int] = {}
        for i, name in enumerate(CHANNELS):
            valid[name], reasons[name] = self.ch[name].check(float(sample[i]), dt)

        self._plausibility(valid, dt)

        values: List[float] = []
        recovered: List[str] = []
        newly_failed: List[str] = []
        outage = 0.0
        for name in CHANNELS:
            ch = self.ch[name]
            was_failed = ch.health == Health.FAILED
            if name == "camera" and c.camera_timeout_s <= 0.0:
                # Frames are sparse and their absence is normal (no lane markings), so the
                # camera's health is judged per received frame, at the nominal frame period.
                self._cam_gap += dt
                if reasons[name] == MISSING:
                    ch.reason = PLAUSIBILITY if ch.plaus_failed else 0
                else:
                    ch.finish(reasons[name], min(self._cam_gap, 0.2))
                    self._cam_gap = 0.0
            else:
                ch.finish(reasons[name], dt)
            failed = ch.health == Health.FAILED
            values.append(NAN if failed else valid[name])
            if failed:
                if not was_failed:
                    newly_failed.append(name)
                self._outage_t[name] += dt
            elif was_failed:
                recovered.append(name)
                outage = max(outage, self._outage_t[name])
                self._outage_t[name] = 0.0

        critical_failed = any(self.ch[n].health == Health.FAILED for n in CRITICAL)
        self._critical_failed_t = self._critical_failed_t + dt if critical_failed else 0.0
        return GuardOutput(
            values=values,
            health={n: self.ch[n].health for n in CHANNELS},
            reasons={n: self.ch[n].reason for n in CHANNELS},
            degraded=any(self.ch[n].health != Health.OK for n in CHANNELS),
            critical_failed=critical_failed,
            failsafe=self._critical_failed_t >= c.failsafe_after_s,
            newly_failed=tuple(newly_failed),
            recovered=tuple(recovered),
            outage_s=outage,
        )

    def summary(self) -> Dict[str, Dict[str, int]]:
        """Rejected-sample counts per channel and reason, plus qualified plausibility faults."""
        out = {n: {REASON_NAMES[r]: k for r, k in ch.counts.items() if k} for n, ch in self.ch.items()}
        out["plausibility_rules"] = {r: k for r, k in self.rule_counts.items() if k}
        return out

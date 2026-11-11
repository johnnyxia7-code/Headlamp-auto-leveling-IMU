"""Extended Kalman filter for headlamp leveling: IMU + wheel speed + camera (+ optional NN).

The full derivation, tuning rationale and embedded notes are in
``docs/ekf_design.md``. Units inside the filter are SI (m, s, rad).

State vector (13)::

    0  v       vehicle speed along the road                    m/s
    1  a       acceleration along the road                     m/s^2
    2  gam     road grade angle (uphill +)                     rad
    3  gamr    road grade rate d(gam)/dt                       rad/s
    4  ths     static load pitch rel. to road (nose-up +)      rad     <- leveling output
    5  thd     dynamic pitch (squat / dive)                    rad
    6  thdr    dynamic pitch rate                              rad/s
    7  kp      pitch gradient (thd steady state per m/s^2)     rad/(m/s^2)
    8  bax     accelerometer x bias                            m/s^2
    9  baz     accelerometer z bias                            m/s^2
    10 bgy     gyro y bias                                     rad/s
    11 sw      wheel-speed scale factor (tyre radius)          -
    12 bcam    camera pitch extrinsic bias                     rad

Process model (Euler-discretised, dt = 1 / fs)::

    v'    = v + a dt
    a'    = a                                   + w_jerk
    gam'  = gam + gamr dt
    gamr' = gamr (1 - dt / tau)                 + w_curv   (variance ~ distance travelled)
    ths'  = ths                                 + w_load   (large only at standstill)
    thd'  = thd + thdr dt
    thdr' = thdr + (wn^2 (kp a - thd) - 2 zeta wn thdr) dt + w_road
    kp, bax, baz, bgy, sw, bcam: random walks

Measurement models (theta = ths + thd, phi = gam + theta)::

    acc_x  = a cos(theta) + g sin(phi) + bax
    acc_z  = -a sin(theta) + g cos(phi) cos(rho) + baz
    gyr_y  = -(gamr + thdr) + bgy                     (right-handed y-left axis: nose-up < 0)
    v_whl  = sw v
    cam    = ths + thd - gamr * d_la / v + bcam       (pitch to road plane at look-ahead d_la)
    nn     = ths                                      (optional learned window estimate)

rho is the roll angle, not a state: it is computed from the lateral
accelerometer minus the centripetal acceleration v * yaw_rate.

Measurement noise is adaptive: road-induced vibration varies ~10x between
smooth and rough roads, so R for acc_x, acc_z and gyro is estimated online
from the variance of first differences of the raw signal (a model-free
white-noise estimate), floored and inflated for residual correlation.

Standstill (wheel speed ~0): zero-velocity pseudo-measurements v = a = gamr = 0
and the grade random walk is frozen. The road cannot change while the car
stands still, so a tilt change seen by the accelerometer is a load change.
The static-pitch process noise is raised only when such a change is
detected (event-triggered): after the suspension has settled from braking,
the low-pass accelerometer tilt (minus the estimated dynamic pitch) is
compared with a reference; a deviation
above ``load_change_tilt_deg`` releases ``ths`` and moves the reference.
While a tilt change is pending (current tilt off the reference, no decision
yet) the accelerometer updates are held, so the change is not absorbed by
the grade or biases. The decision uses the tilt from ``load_change_delay_s``
earlier and is only taken while the car is still standing, so the onset of a drive-away (which
starts before the wheel-speed sensor reports motion) is never mistaken for
a load change. Releasing ``ths`` during every stop would instead let the
brake-dive rebound leak into the static pitch.

All updates are sequential scalar updates (no matrix inversion), with
chi-square innovation gating and a per-sensor NIS monitor for fault detection.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np

from .. import G
from ..config import FusionConfig

N_STATES = 13
V, A, GAM, GAMR, THS, THD, THDR, KP, BAX, BAZ, BGY, SW, BCAM = range(N_STATES)
STATE_NAMES = ("v", "a", "gam", "gamr", "ths", "thd", "thdr", "kp", "bax", "baz", "bgy", "sw", "bcam")
SENSORS = ("acc_x", "acc_z", "gyro", "wheel", "camera", "nn")
CRITICAL_SENSORS = ("acc_x", "acc_z", "gyro", "wheel")   # a fault here -> lamp fail-safe
AUX_SENSORS = ("camera", "nn")                            # a fault here -> input disabled

D2R = np.pi / 180.0
MG = 1e-3 * G


# --------------------------------------------------------------------------- models
# Pure functions so the Jacobians can be verified against finite differences
# (tests/test_fusion.py) and ported 1:1 to the embedded implementation.

def propagate(x: np.ndarray, dt: float, wn: float, zeta: float, grade_tau: float):
    """Process model f(x) and its Jacobian F = df/dx."""
    w2, c2 = wn * wn, 2.0 * zeta * wn
    v, a, gamr, thd, thdr, kp = x[V], x[A], x[GAMR], x[THD], x[THDR], x[KP]
    xn = x.copy()
    xn[V] = v + a * dt
    xn[GAM] = x[GAM] + gamr * dt
    xn[GAMR] = gamr * (1.0 - dt / grade_tau)
    xn[THD] = thd + thdr * dt
    xn[THDR] = thdr + (w2 * (kp * a - thd) - c2 * thdr) * dt
    F = np.eye(N_STATES)
    F[V, A] = dt
    F[GAM, GAMR] = dt
    F[GAMR, GAMR] = 1.0 - dt / grade_tau
    F[THD, THDR] = dt
    F[THDR, A] = w2 * kp * dt
    F[THDR, THD] = -w2 * dt
    F[THDR, THDR] = 1.0 - c2 * dt
    F[THDR, KP] = w2 * a * dt
    return xn, F


def h_acc_x(x: np.ndarray):
    a, th = x[A], x[THS] + x[THD]
    phi = x[GAM] + th
    H = np.zeros(N_STATES)
    H[A] = np.cos(th)
    H[THS] = H[THD] = -a * np.sin(th) + G * np.cos(phi)
    H[GAM] = G * np.cos(phi)
    H[BAX] = 1.0
    return a * np.cos(th) + G * np.sin(phi) + x[BAX], H


def h_acc_z(x: np.ndarray, roll: float = 0.0):
    a, th = x[A], x[THS] + x[THD]
    phi = x[GAM] + th
    c_ro = np.cos(roll)
    H = np.zeros(N_STATES)
    H[A] = -np.sin(th)
    H[THS] = H[THD] = -a * np.cos(th) - G * np.sin(phi) * c_ro
    H[GAM] = -G * np.sin(phi) * c_ro
    H[BAZ] = 1.0
    return -a * np.sin(th) + G * np.cos(phi) * c_ro + x[BAZ], H


def h_gyro(x: np.ndarray):
    H = np.zeros(N_STATES)
    H[GAMR] = H[THDR] = -1.0
    H[BGY] = 1.0
    return -(x[GAMR] + x[THDR]) + x[BGY], H


def h_wheel(x: np.ndarray):
    H = np.zeros(N_STATES)
    H[V] = x[SW]
    H[SW] = x[V]
    return x[SW] * x[V], H


def h_camera(x: np.ndarray, lookahead_m: float):
    v, d = x[V], lookahead_m
    H = np.zeros(N_STATES)
    H[THS] = H[THD] = 1.0
    H[GAMR] = -d / v
    H[V] = x[GAMR] * d / (v * v)
    H[BCAM] = 1.0
    return x[THS] + x[THD] - x[GAMR] * d / v + x[BCAM], H


@dataclass
class EkfOutput:
    load_pitch_deg: float
    load_pitch_std_deg: float
    road_grade_deg: float
    road_grade_std_deg: float
    body_pitch_deg: float          # ths + thd, relative to the road
    standstill: bool
    faults: Dict[str, bool] = field(default_factory=dict)

    @property
    def critical_fault(self) -> bool:
        """IMU or wheel-speed fault: the estimate cannot be trusted, use the fail-safe aim."""
        return any(self.faults.get(s, False) for s in CRITICAL_SENSORS)


class PitchEKF:
    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.wn = 2.0 * np.pi * cfg.pitch_freq_hz
        self.zeta = cfg.pitch_damping
        self.kp_nominal = cfg.pitch_gradient_deg_per_g * D2R / G
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, load_pitch_deg: Optional[float] = None, load_pitch_std_deg: Optional[float] = None) -> None:
        """(Re)start the filter, e.g. at ignition-on with the value stored in NVM."""
        c = self.cfg
        lp = c.init_load_pitch_deg if load_pitch_deg is None else load_pitch_deg
        lp_std = c.init_load_pitch_std_deg if load_pitch_std_deg is None else load_pitch_std_deg
        self.x = np.zeros(N_STATES)
        self.x[THS] = lp * D2R
        self.x[KP] = self.kp_nominal
        self.x[SW] = 1.0
        sd = np.zeros(N_STATES)
        sd[V], sd[A] = 0.5, 0.5
        sd[GAM], sd[GAMR] = 8.0 * D2R, 0.01
        sd[THS] = lp_std * D2R
        sd[THD], sd[THDR] = 0.2 * D2R, 0.5 * D2R
        sd[KP] = c.pitch_gradient_rel_std * self.kp_nominal
        sd[BAX] = sd[BAZ] = c.init_accel_bias_mg * MG
        sd[BGY] = c.init_gyro_bias_dps * D2R
        sd[SW] = c.init_wheel_scale_std
        sd[BCAM] = c.init_camera_bias_deg * D2R
        self.P = np.diag(sd ** 2)
        self.initialized = False
        self.standstill = False
        self._still_time = 0.0
        self.roll = 0.0
        self._tilt_lp: Optional[float] = None
        self._tilt_hist: Deque[float] = deque()
        self._tilt_ref: Optional[float] = None
        self._load_event = False
        self._tilt_pending = False
        self._prev_meas: Optional[np.ndarray] = None
        self._noise_var = np.array([c.r_accel_x, c.r_accel_z, c.r_gyro_dps * D2R]) ** 2
        self.nis_ema = {s: 1.0 for s in SENSORS}
        self.faults = {s: False for s in SENSORS}
        self._time = 0.0
        self.numeric_resets = 0
        self.gate_lockouts = 0
        self._last_good_load_pitch_deg = lp
        self._updated: set = set()
        self._imu_gap_s = 0.0
        self._wheel_gap_s = 0.0
        self._prev_gy = self._prev_gy2 = float("nan")
        self._gate_run = {s: 0 for s in CRITICAL_SENSORS}

    def _initialize(self, ax: float, az: float) -> None:
        # Linearisation point for the grade from the first accelerometer sample.
        self.x[GAM] = np.arctan2(ax, az) - self.x[THS]
        self.initialized = True

    # -------------------------------------------------------------- prediction
    def _on_drive_away(self) -> None:
        """Re-open speed / acceleration uncertainty when the vehicle starts moving.

        At standstill v and a are pinned to zero with small variance. Without
        this, the launch acceleration (seen by acc_x before the wheel-speed
        derivative catches up) would be explained as body pitch.
        """
        sd = {V: 0.3, A: self.cfg.drive_away_accel_std}
        for i, s in sd.items():
            self.P[i, :] = 0.0
            self.P[:, i] = 0.0
            self.P[i, i] = s * s

    def _detect_load_change(self, ax: float, az: float, dt: float) -> None:
        c = self.cfg
        # Tilt with the filter's dynamic pitch removed, so the brake-dive rebound
        # (observed by the gyro through thd/thdr) does not look like a load change.
        tilt = float(np.arctan2(ax, az)) - self.x[THD]
        alpha = min(1.0, dt / c.tilt_lp_tau_s)
        self._tilt_lp = tilt if self._tilt_lp is None else self._tilt_lp + alpha * (tilt - self._tilt_lp)
        self._load_event = False
        self._tilt_pending = False
        if not self.standstill:
            self._tilt_ref = None
            self._tilt_hist.clear()
            return
        # A tilt change not yet decided on (load change or drive-away onset): hold the
        # accelerometer updates so the change is not absorbed by grade / biases meanwhile.
        self._tilt_pending = (self._tilt_ref is not None and
                              abs(self._tilt_lp - self._tilt_ref) > c.load_change_tilt_deg * D2R)
        self._tilt_hist.append(self._tilt_lp)
        n_delay = max(1, int(round(c.load_change_delay_s / dt)))
        if len(self._tilt_hist) <= n_delay:
            return
        delayed = self._tilt_hist.popleft()            # tilt load_change_delay_s ago
        if self._still_time < c.standstill_settle_s + c.load_change_delay_s:
            return
        if self._tilt_ref is None:
            self._tilt_ref = delayed
            return
        change = abs(delayed - self._tilt_ref)
        if c.load_change_tilt_deg * D2R < change < c.load_change_max_deg * D2R:
            self._load_event = True
            self._tilt_ref = delayed

    def predict(self, dt: float) -> None:
        c = self.cfg
        speed = abs(self.x[V])
        self.x, F = propagate(self.x, dt, self.wn, self.zeta, c.grade_rate_tau_s)

        q = np.zeros(N_STATES)
        q[V] = 0.01 ** 2
        q[A] = c.q_jerk ** 2
        if not self.standstill:
            q[GAMR] = c.q_grade_curv ** 2 * max(speed, 0.5)       # per metre travelled
        q[THS] = (c.q_load_pitch_stop_deg if self._load_event else c.q_load_pitch_drive_deg) ** 2 * D2R ** 2
        q[THDR] = (c.q_pitch_dyn_rate * max(speed, 1.0)) ** 2
        q[KP] = (1e-3 * self.kp_nominal) ** 2
        q[BAX] = q[BAZ] = (c.q_accel_bias_mg * MG) ** 2
        q[BGY] = (c.q_gyro_bias_dps * D2R) ** 2
        q[SW] = 1e-5 ** 2
        q[BCAM] = (1e-4 * D2R) ** 2
        self.P = F @ self.P @ F.T + np.diag(q * dt)

    # ------------------------------------------------------------------ update
    def update_scalar(self, sensor: str, z: float, h: float, H: np.ndarray, R: float,
                      gate: Optional[float] = None, huber: bool = False,
                      monitor_var: Optional[float] = None) -> bool:
        """Sequential scalar EKF update with chi-square gating. Returns True if applied.

        With ``huber`` the update is a Huber M-estimate: an innovation beyond
        ``huber_k`` sigma keeps its direction but its gain is scaled by
        ``huber_k / |normalised innovation|`` (equivalent to inflating R for that
        sample), so a moderate outlier nudges the state instead of pulling it.
        """
        PHt = self.P @ H
        S = float(H @ PHt) + R
        nu = z - h
        nis = nu * nu / S
        self._updated.add(sensor)
        if sensor in self.nis_ema and self._time >= self.cfg.fault_warmup_s:
            # Capped samples: a short transient (drive-away, pothole, outlier frame) cannot
            # trip the monitor on its own; a persistent inconsistency can.
            # monitor_var: a tighter innovation variance for the monitor only, so a biased
            # input cannot hide behind the uncertainty of a weakly observed state.
            nis_mon = nis if monitor_var is None else nu * nu / monitor_var
            self._monitor(sensor, min(nis_mon, self.cfg.fault_nis_cap))
        if sensor in AUX_SENSORS and self.faults[sensor]:
            return False     # monitor only: a faulted auxiliary input is watched, not used
        if gate is not None and nis > gate:
            if sensor in self._gate_run:
                self._gate_run[sensor] += 1
            return False
        if sensor in self._gate_run:
            self._gate_run[sensor] = 0
        k = self.cfg.huber_k
        if huber and k > 0.0 and nis > k * k:
            S *= np.sqrt(nis) / k
        K = PHt / S
        self.x += K * nu
        self.P -= np.outer(K, PHt)
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def _monitor(self, sensor: str, nis: float) -> None:
        """Residual monitor: EMA of the normalised innovation squared, with hysteresis."""
        c = self.cfg
        if sensor in AUX_SENSORS:
            alpha = 1.0 / max(1, c.fault_window_aux_updates)
        else:
            alpha = min(1.0, 1.0 / (c.fault_window_s * c.fs_hz))
        self.nis_ema[sensor] += alpha * (nis - self.nis_ema[sensor])
        if self.nis_ema[sensor] > c.fault_nis:
            self.faults[sensor] = True
        elif self.nis_ema[sensor] < c.fault_nis * c.fault_clear_ratio:
            self.faults[sensor] = False

    def _pseudo(self, idx: int, value: float, sigma: float) -> None:
        H = np.zeros(N_STATES)
        H[idx] = 1.0
        self.update_scalar("pseudo", value, self.x[idx], H, sigma ** 2)

    def _r(self, i: int) -> float:
        """Adaptive measurement variance for acc_x (0), acc_z (1), gyro (2)."""
        c = self.cfg
        floor = (c.r_accel_x, c.r_accel_z, c.r_gyro_dps * D2R)[i]
        return max(floor ** 2, self._noise_var[i]) * c.r_colored_inflation

    def _track_noise(self, ax: float, az: float, gy_dps: float, dt: float) -> None:
        """EMA of half the squared first difference: white-noise variance of each channel.

        A channel without a sample this cycle (NaN) keeps its estimate; its
        difference chain restarts at the next valid sample.
        """
        m = np.array([ax, az, gy_dps * D2R])
        if self._prev_meas is not None:
            alpha = min(1.0, dt / self.cfg.r_adapt_tau_s)
            d = m - self._prev_meas
            ok = np.isfinite(d)
            self._noise_var[ok] += alpha * (0.5 * d[ok] ** 2 - self._noise_var[ok])
        self._prev_meas = m

    def update_accel(self, ax: float, az: float) -> None:
        c = self.cfg
        if np.isfinite(ax):
            h, H = h_acc_x(self.x)
            self.update_scalar("acc_x", ax, h, H, self._r(0), c.gate_imu, huber=True)
        if np.isfinite(az):
            h, H = h_acc_z(self.x, self.roll)        # re-linearised after the x update
            self.update_scalar("acc_z", az, h, H, self._r(1), c.gate_imu, huber=True)

    def update_gyro(self, gy_dps: float) -> None:
        h, H = h_gyro(self.x)
        self.update_scalar("gyro", gy_dps * D2R, h, H, self._r(2), self.cfg.gate_imu, huber=True)

    def update_wheel(self, v_wheel: float) -> None:
        c = self.cfg
        h, H = h_wheel(self.x)
        r = c.r_wheel_ms + c.r_wheel_slip * abs(self.x[A])
        self.update_scalar("wheel", v_wheel, h, H, r * r, c.gate_imu, huber=True)

    def update_camera(self, pitch_deg: float) -> bool:
        c = self.cfg
        if self.x[V] < c.camera_min_speed_ms:   # look-ahead geometry and lane detection need speed
            return False
        h, H = h_camera(self.x, c.camera_lookahead_m)
        R = (c.r_camera_deg * D2R) ** 2
        # The monitor compares the frame with the pitch states only. The full innovation
        # variance also holds the grade-rate uncertainty (times d / v), which would let a
        # constant camera offset pass as "grade change ahead".
        P = self.P
        mon = R + P[THS, THS] + P[THD, THD] + P[BCAM, BCAM] + 2.0 * (P[THS, THD] + P[THS, BCAM] + P[THD, BCAM])
        return self.update_scalar("camera", pitch_deg * D2R, h, H, R, c.gate_camera, monitor_var=max(mon, R))

    def update_load_pitch(self, pitch_deg: float, std_deg: float, sensor: str = "nn") -> bool:
        """Direct measurement of the static load pitch, e.g. from the neural window estimator."""
        H = np.zeros(N_STATES)
        H[THS] = 1.0
        return self.update_scalar(sensor, pitch_deg * D2R, self.x[THS], H, (std_deg * D2R) ** 2,
                                  self.cfg.gate_camera)

    # -------------------------------------------------------------- full step
    def step(self, dt: float, ax: float, az: float, gy_dps: float, v_wheel: float,
             camera_pitch_deg: float = float("nan"), ay: float = float("nan"),
             gz_dps: float = float("nan")) -> EkfOutput:
        """One filter cycle: standstill detection, prediction and all available updates.

        ``ay`` / ``gz_dps`` (lateral accel, yaw rate) are optional; when given they
        provide the roll angle used in the vertical-accelerometer model.

        Any input may be NaN ("no valid sample this cycle", as delivered by the
        input guard). The corresponding update is skipped and the filter coasts on
        its model and the remaining sensors; its covariance grows accordingly.
        """
        c = self.cfg
        self._time += dt
        ax, az = self._remove_lever_arm(ax, az, gy_dps, dt)
        acc_ok = bool(np.isfinite(ax) and np.isfinite(az))
        wheel_ok = bool(np.isfinite(v_wheel))
        if np.isfinite(ay) and np.isfinite(gz_dps) and wheel_ok:
            a_lat = v_wheel * gz_dps * D2R
            self.roll = float(np.arcsin(np.clip((ay - a_lat) / G, -0.3, 0.3)))
        if not self.initialized:
            if not acc_ok:
                return self.output()             # nothing to linearise around yet
            self._initialize(ax, az)

        # Without a wheel speed, "standstill" is unknown: no zero-velocity updates and
        # no load-change logic (both would be wrong if the car is in fact moving).
        self._still_time = self._still_time + dt if wheel_ok and v_wheel < c.standstill_speed_ms else 0.0
        was_still = self.standstill
        self.standstill = self._still_time >= c.standstill_min_s
        if was_still and not self.standstill:
            self._on_drive_away()
        if acc_ok:
            self._detect_load_change(ax, az, dt)
        else:
            self._load_event = self._tilt_pending = False
            self._tilt_ref = None
            self._tilt_hist.clear()

        # First IMU sample after a gap: the fast states (squat / dive, grade rate) moved
        # by an unknown amount meanwhile. Re-open them so the returning data corrects
        # them, not the slowly varying load pitch.
        imu_now = acc_ok and bool(np.isfinite(gy_dps))
        if imu_now and self._imu_gap_s > 0.0:
            t = self._imu_gap_s
            self._reopen(THD, 0.2 * D2R)
            self._reopen(THDR, 0.5 * D2R)
            self._reopen(GAMR, 0.01)
            self.P[GAM, GAM] += (0.01 * t) ** 2
            self._prev_meas = None
        self._imu_gap_s = 0.0 if imu_now else self._imu_gap_s + dt
        # Same for the speed after a wheel-speed gap: it was integrated from an
        # acceleration that is only known up to the grade / bias ambiguity.
        if wheel_ok and self._wheel_gap_s > 0.0:
            self.P[V, V] += (c.drive_away_accel_std * self._wheel_gap_s) ** 2
        self._wheel_gap_s = 0.0 if wheel_ok else self._wheel_gap_s + dt
        self._updated = set()
        n_sub = max(1, int(np.ceil(dt / c.max_step_s - 1e-9)))     # long gap: predict in sub-steps
        for _ in range(n_sub):
            self.predict(dt / n_sub)
        self._track_noise(ax, az, gy_dps, dt)
        if self.standstill:
            self._pseudo(V, 0.0, 0.01)
            self._pseudo(A, 0.0, 0.05)
            self._pseudo(GAMR, 0.0, 1e-4)
        elif wheel_ok:
            self.update_wheel(v_wheel)
        if np.isfinite(gy_dps):
            self.update_gyro(gy_dps)
        if not (self._tilt_pending and not self._load_event):
            self.update_accel(ax, az)
        if c.use_camera and np.isfinite(camera_pitch_deg):
            self.update_camera(camera_pitch_deg)
        # A critical sensor that delivered nothing this cycle gives no evidence either way:
        # its monitor relaxes towards "consistent" instead of freezing at its last value.
        for s in CRITICAL_SENSORS:
            if s not in self._updated and not (s == "wheel" and self.standstill):
                self._monitor(s, 1.0)
        self._break_gate_lockout()
        self._sanitize()
        return self.output()

    def _remove_lever_arm(self, ax: float, az: float, gy_dps: float, dt: float):
        """Refer the accelerometer to the pitch axis.

        An IMU mounted ahead of the pitch axis sees ``x * pitch acceleration`` in its
        vertical channel (and ``-z *`` it in the longitudinal one). During every
        acceleration change that term is correlated with the acceleration itself, so
        uncompensated it biases the load pitch (about +0.2 deg for a 0.9 m lever arm
        on the verification plant). The pitch acceleration is taken from the gyro
        (model-free); without a gyro sample it falls back to the filter's own
        squat / dive model.
        """
        c = self.cfg
        if c.imu_lever_x_m == 0.0 and c.imu_lever_z_m == 0.0:
            return ax, az
        if np.isfinite(gy_dps) and np.isfinite(self._prev_gy) and np.isfinite(self._prev_gy2) and dt > 0.0:
            # Second-order backward difference: the derivative at the current sample, not
            # half a sample earlier, so it lines up with the accelerometer sample.
            thdd = -(3.0 * gy_dps - 4.0 * self._prev_gy + self._prev_gy2) * D2R / (2.0 * dt)
        elif np.isfinite(gy_dps) and np.isfinite(self._prev_gy) and dt > 0.0:
            thdd = -(gy_dps - self._prev_gy) * D2R / dt          # nose-up positive
        elif self.initialized:
            x = self.x
            thdd = self.wn ** 2 * (x[KP] * x[A] - x[THD]) - 2.0 * self.zeta * self.wn * x[THDR]
        else:
            thdd = 0.0
        self._prev_gy2, self._prev_gy = self._prev_gy, gy_dps
        return ax + c.imu_lever_z_m * thdd, az - c.imu_lever_x_m * thdd

    # ------------------------------------------------------------ resilience
    def _reopen(self, idx: int, sd: float) -> None:
        """Forget what the filter believed about one state (variance reset, correlations cleared)."""
        self.P[idx, :] = 0.0
        self.P[:, idx] = 0.0
        self.P[idx, idx] = sd * sd

    def notify_recovered(self, channels, outage_s: float) -> None:
        """A sensor is usable again after ``outage_s`` without it.

        States that sensor was observing have drifted by an unknown amount, so
        their uncertainty is re-opened in proportion to the outage. The static
        load pitch keeps its estimate (it does not change while driving) with a
        modestly inflated variance.
        """
        t = max(0.0, outage_s)
        channels = set(channels)
        if "v_wheel" in channels:
            self._reopen(V, min(40.0, 0.5 + 3.0 * t))
            self._reopen(A, self.cfg.drive_away_accel_std)
        if channels & {"ax", "az"}:
            self._reopen(GAM, min(8.0, 0.2 + 0.5 * t) * D2R)
            self._reopen(A, self.cfg.drive_away_accel_std)
            self._tilt_ref = None
            self._tilt_hist.clear()
            self._prev_meas = None
        if channels & {"ax", "az", "gy"}:
            self._reopen(GAMR, 0.01)
            self._reopen(THD, 0.2 * D2R)
            self._reopen(THDR, 0.5 * D2R)
            self.P[THS, THS] += (min(0.3, 0.02 * t) * D2R) ** 2
        for s in CRITICAL_SENSORS:                 # the monitors restart from a clean slate
            self.nis_ema[s] = 1.0
            self.faults[s] = False

    def _break_gate_lockout(self) -> None:
        """A sensor whose samples have all been gated out for ``gate_lockout_s`` is either
        faulty (then the monitor / input guard deals with it) or the filter has drifted
        away from it and would reject it forever. Re-opening the states it observes lets
        the filter re-acquire; the monitor keeps running, so a real fault still shows."""
        c = self.cfg
        limit = max(1, int(round(c.gate_lockout_s * c.fs_hz)))
        reopen = {
            "wheel": ((V, 5.0), (A, c.drive_away_accel_std)),
            "acc_x": ((GAM, 2.0 * D2R), (A, c.drive_away_accel_std)),
            "acc_z": ((THD, 0.2 * D2R), (BAZ, c.init_accel_bias_mg * MG)),
            "gyro": ((GAMR, 0.01), (THDR, 0.5 * D2R), (BGY, c.init_gyro_bias_dps * D2R)),
        }
        for s, run in self._gate_run.items():
            if run >= limit:
                for idx, sd in reopen[s]:
                    self._reopen(idx, sd)
                self._gate_run[s] = 0
                self.gate_lockouts += 1

    def notify_isolated(self, channels) -> None:
        """The input guard has just disqualified these channels.

        Their last samples before isolation were already wrong, so the states they
        act on most directly are re-opened and their residual monitors restart.
        Otherwise the monitor would still report the (now removed) fault and latch
        the fail-safe.
        """
        c = self.cfg
        channels = set(channels)
        if "gy" in channels:
            self._reopen(BGY, c.init_gyro_bias_dps * D2R)
            self._reopen(GAMR, 0.01)
            self._reopen(THDR, 0.5 * D2R)
        if "az" in channels:
            self._reopen(BAZ, c.init_accel_bias_mg * MG)
        if "ax" in channels:
            self._reopen(BAX, c.init_accel_bias_mg * MG)
            self._reopen(GAM, 2.0 * D2R)
        if "v_wheel" in channels:
            self._reopen(V, 2.0)
            self._reopen(A, c.drive_away_accel_std)
        if channels & {"ax", "az", "gy", "v_wheel"}:
            for s in CRITICAL_SENSORS:
                self.nis_ema[s] = 1.0
                self.faults[s] = False

    def _sanitize(self) -> None:
        """Numerical safety net: never let a non-finite or indefinite state survive a cycle."""
        d = np.diag(self.P)
        if np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.P)) and np.all(d > 0.0):
            self._last_good_load_pitch_deg = float(self.x[THS] / D2R)
            return
        self.numeric_resets += 1
        keep_time, resets = self._time, self.numeric_resets
        self.reset(self._last_good_load_pitch_deg, self.cfg.recover_load_pitch_std_deg)
        self._time, self.numeric_resets = keep_time, resets

    def output(self) -> EkfOutput:
        x, P = self.x, self.P
        return EkfOutput(
            load_pitch_deg=float(x[THS] / D2R),
            load_pitch_std_deg=float(np.sqrt(P[THS, THS]) / D2R),
            road_grade_deg=float(x[GAM] / D2R),
            road_grade_std_deg=float(np.sqrt(P[GAM, GAM]) / D2R),
            body_pitch_deg=float((x[THS] + x[THD]) / D2R),
            standstill=self.standstill,
            faults=dict(self.faults),
        )

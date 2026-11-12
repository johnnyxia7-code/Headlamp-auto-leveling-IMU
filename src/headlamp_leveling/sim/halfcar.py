"""Half-car verification plant with scripted manoeuvres.

This is a second, independent vehicle model used to *verify* the estimator.
The training simulator (``sim/scenario.py``) and the EKF share the same
simplification: body pitch is a second-order response to the longitudinal
acceleration. Testing the filter only against that model would be an "inverse
crime". This plant is built from the mechanics instead:

    * 4 degrees of freedom: body heave and pitch, front and rear unsprung mass
    * separate front / rear springs, dampers (with friction) and tyre stiffness
    * loads as point masses on the body (they change mass, inertia and trim)
    * load transfer from the specific force a_x + g sin(grade) at CG height,
      with different anti-dive (braking) and anti-squat (traction) geometry
    * road roughness enters at the front axle and, one wheelbase later, the rear
    * vertical road curvature (crests / dips) and an IMU lever arm
    * in curves: the IMU's distance from the rear axle (centripetal term) and the
      rear tyres' side-slip, both of which appear in the longitudinal accelerometer
    * integrated at 1 kHz; sensors sample at 100 Hz

Frames: the plant works in a road-aligned frame at the vehicle position (x along
the road, z normal to it). ``th`` is the body pitch relative to the road, nose-up
positive; the reference (th = 0) is the unladen car with driver on a level road.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import G
from ..config import Config
from .camera import camera_pitch_measurements
from .imu import imu_measurements, wheel_speed_measurement
from .road import iso8608_profile

ROUGHNESS_GD = {"A": 16e-6, "B": 64e-6, "C": 256e-6}


@dataclass
class HalfCarParams:
    m_s: float = 1450.0            # sprung mass incl. driver (reference state), kg
    i_y: float = 2300.0            # pitch inertia about the CG, kg m^2
    a: float = 1.25                # CG to front axle, m
    b: float = 1.50                # CG to rear axle, m
    h: float = 0.55                # CG height above the road, m
    k_f: float = 44e3              # front axle wheel rate (both wheels), N/m
    k_r: float = 48e3
    c_f: float = 3.4e3             # front axle damping, N s/m
    c_r: float = 3.6e3
    f_fric: float = 60.0           # suspension friction per axle, N
    m_uf: float = 90.0             # unsprung mass per axle, kg
    m_ur: float = 80.0
    k_tf: float = 420e3            # tyre vertical stiffness per axle, N/m
    k_tr: float = 420e3
    anti_dive: float = 0.5         # share of braking load transfer carried by the links
    anti_squat: float = 0.3        # same for traction
    roll_gradient_deg_per_g: float = 4.5
    c_alpha_r: float = 110e3       # rear axle cornering stiffness (both tyres), N/rad
    x_imu: float = 0.9             # IMU position forward of the CG, m
    z_imu: float = -0.1            # IMU position above the CG, m
    # Load positions behind the front axle, m.
    x_front_row: float = 1.05
    x_rear_row: float = 2.20
    x_trunk: float = 3.30
    x_towball: float = 3.85

    @property
    def wheelbase(self) -> float:
        return self.a + self.b

    @property
    def pitch_stiffness(self) -> float:
        return self.k_f * self.a ** 2 + self.k_r * self.b ** 2

    @property
    def pitch_freq_hz(self) -> float:
        return math.sqrt(self.pitch_stiffness / self.i_y) / (2.0 * math.pi)

    def rear_slip_gradient(self, s0: float = 0.0, s1: float = 0.0) -> float:
        """Rear-axle side-slip angle per unit lateral acceleration, rad / (m/s^2).

        ``s0`` / ``s1`` are the load sums (mass, mass * position forward of the CG).
        """
        m_rear = (self.m_s * self.a + s0 * self.a - s1) / self.wheelbase + self.m_ur
        return m_rear / self.c_alpha_r

    def pitch_gradient_deg_per_g(self, braking: bool = False) -> float:
        eta = self.anti_dive if braking else self.anti_squat
        return math.degrees((1.0 - eta) * self.m_s * self.h * G / self.pitch_stiffness)


# Deterministic load cases relative to "driver only": lists of (mass kg, position name).
LOADS: Dict[str, List[Tuple[float, str]]] = {
    "driver_only": [],
    "front_passenger": [(75.0, "front_row")],
    "rear_passengers": [(75.0, "rear_row"), (75.0, "rear_row")],
    "full_cabin": [(75.0, "front_row"), (225.0, "rear_row")],
    "full_cabin_trunk": [(75.0, "front_row"), (225.0, "rear_row"), (100.0, "trunk")],
    "trunk_heavy": [(150.0, "trunk")],
    "trailer": [(75.0, "towball")],
    "max_rear": [(75.0, "front_row"), (225.0, "rear_row"), (150.0, "trunk"), (75.0, "towball")],
}


def load_sums(p: HalfCarParams, load: str) -> Tuple[float, float, float]:
    """(sum m, sum m d, sum m d^2) with d = position forward of the CG."""
    if load not in LOADS:
        raise KeyError(f"unknown load {load!r}; choose from {sorted(LOADS)}")
    s0 = s1 = s2 = 0.0
    for m, where in LOADS[load]:
        d = p.a - getattr(p, f"x_{where}")
        s0 += m
        s1 += m * d
        s2 += m * d * d
    return s0, s1, s2


def static_equilibrium(p: HalfCarParams, s0: float, s1: float) -> np.ndarray:
    """[z_s, th, z_uf, z_ur] at rest on a level road for a load (sum m, sum m d)."""
    # Unknowns x = [zs, th, zuf, zur]; suspension force on body F = -k (z_body - z_u).
    A = np.array([
        [-(p.k_f + p.k_r), -(p.k_f * p.a - p.k_r * p.b), p.k_f, p.k_r],
        [-(p.k_f * p.a - p.k_r * p.b), -(p.k_f * p.a ** 2 + p.k_r * p.b ** 2), p.k_f * p.a, -p.k_r * p.b],
        [p.k_f, p.k_f * p.a, -(p.k_f + p.k_tf), 0.0],
        [p.k_r, -p.k_r * p.b, 0.0, -(p.k_r + p.k_tr)],
    ])
    rhs = np.array([s0 * G, s1 * G, 0.0, 0.0])
    return np.linalg.solve(A, rhs)


def slope_pitch_per_moment(p: HalfCarParams, braking: bool) -> float:
    """Steady body pitch (rad) per unit load-transfer moment m * f * h (N m).

    ``f`` is the specific force along the road: acceleration, or g sin(grade) when
    standing or cruising on a slope. Includes the tyre deflection.
    """
    eta = p.anti_dive if braking else p.anti_squat
    A = np.array([
        [-(p.k_f + p.k_r), -(p.k_f * p.a - p.k_r * p.b), p.k_f, p.k_r],
        [-(p.k_f * p.a - p.k_r * p.b), -(p.k_f * p.a ** 2 + p.k_r * p.b ** 2), p.k_f * p.a, -p.k_r * p.b],
        [p.k_f, p.k_f * p.a, -(p.k_f + p.k_tf), 0.0],
        [p.k_r, -p.k_r * p.b, 0.0, -(p.k_r + p.k_tr)],
    ])
    rhs = np.array([0.0, -(1.0 - eta), -eta / p.wheelbase, eta / p.wheelbase])
    return float(np.linalg.solve(A, rhs)[1])


def static_load_pitch_deg(p: HalfCarParams, load: str) -> float:
    """Body pitch relative to a level road caused by the load (nose-up positive)."""
    s0, s1, _ = load_sums(p, load)
    return float(np.degrees(static_equilibrium(p, s0, s1)[1]))


@dataclass
class ScriptedDrive:
    """What happens in a verification run.

    manoeuvres  sequence of ("hold", seconds) | ("to", speed_kmh, accel_ms2)
                "hold" keeps the current speed (or standstill) for the given time;
                "to" accelerates or brakes to the target speed with the given magnitude.
    grade       knots (distance m, grade %) of the road profile, linearly interpolated
                and smoothed over 40 m
    curvature   knots (distance m, 1/radius in 1/m), same interpolation
    roughness   ISO 8608 class "A" | "B" | "C", or None for a perfectly smooth road
    loads       (time s, load name): the load present from that time on. Changes must
                happen at standstill and are applied over ``load_ramp_s``.
    """
    manoeuvres: Sequence[tuple]
    grade: Sequence[Tuple[float, float]] = ((0.0, 0.0),)
    curvature: Sequence[Tuple[float, float]] = ((0.0, 0.0),)
    roughness: Optional[str] = "A"
    loads: Sequence[Tuple[float, str]] = ((0.0, "driver_only"),)
    load_ramp_s: float = 2.0
    jerk_ms3: float = 4.0
    vehicle: HalfCarParams = field(default_factory=HalfCarParams)


def speed_profile(manoeuvres: Sequence[tuple], dt: float, jerk: float,
                  ends: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Jerk-limited acceleration and speed for a manoeuvre script.

    If ``ends`` is given, the end time (s) of each manoeuvre is appended to it.
    """
    a_list: List[float] = []
    v_list: List[float] = []
    v = a = 0.0

    def push() -> None:
        a_list.append(a)
        v_list.append(v)

    for m in manoeuvres:
        if m[0] == "hold":
            for _ in range(int(round(m[1] / dt))):
                a += max(-jerk * dt, min(jerk * dt, -a))            # bleed off any acceleration
                v = max(0.0, v + a * dt)
                push()
        elif m[0] == "to":
            target, mag = m[1] / 3.6, abs(m[2])
            sign = 1.0 if target > v else -1.0
            for _ in range(int(120.0 / dt)):                         # safety bound
                remaining = (target - v) * sign
                if remaining <= 0.0 and abs(a) < 0.05:
                    break
                # Start easing off when the remaining speed equals what the ramp-down covers.
                a_des = sign * mag if remaining > a * a / (2.0 * jerk) else 0.0
                a += max(-jerk * dt, min(jerk * dt, a_des - a))
                v += a * dt
                if v <= 0.0:
                    v, a = 0.0, 0.0
                    push()
                    break
                push()
        else:
            raise ValueError(f"unknown manoeuvre {m!r}")
        if ends is not None:
            ends.append(len(a_list) * dt)
    return np.asarray(a_list), np.asarray(v_list)


def _profile(knots: Sequence[Tuple[float, float]], s_grid: np.ndarray, smooth_m: float) -> np.ndarray:
    k = np.asarray(knots, float)
    y = np.interp(s_grid, k[:, 0], k[:, 1])
    n = max(1, int(smooth_m / (s_grid[1] - s_grid[0])))
    if n > 1:
        pad = np.pad(y, (n, n), mode="edge")
        y = np.convolve(pad, np.ones(n) / n, mode="same")[n:-n]
    return y


def simulate_scripted(drive: ScriptedDrive, cfg: Config, seed: int = 0, ideal_sensors: bool = False) -> Dict:
    """Run the half-car plant through a scripted drive and sample the sensors.

    Returns the same structure as ``simulate_trip(..., raw=True)["raw"]`` (so the
    whole estimation chain runs on it unchanged), plus extra ground truth.
    """
    rng = np.random.default_rng(seed)
    p = drive.vehicle
    dt = 1e-3
    sub = 10                                   # plant steps per 100 Hz sensor sample
    fs = 1.0 / (dt * sub)
    a_cmd, v_arr = speed_profile(drive.manoeuvres, dt, drive.jerk_ms3)
    n = (a_cmd.size // sub) * sub
    a_cmd, v_arr = a_cmd[:n], v_arr[:n]
    a_cmd = np.gradient(v_arr, dt)             # consistent with the clipped speed
    s_arr = np.concatenate([[0.0], np.cumsum(0.5 * (v_arr[1:] + v_arr[:-1]) * dt)])
    L = p.wheelbase
    length = float(s_arr[-1]) + cfg.camera.lookahead_m + L + 60.0

    # Road: grade, curvature, roughness.
    ds = 1.0
    s_grid = np.arange(0.0, length + ds, ds)
    gamma_grid = np.arctan(_profile(drive.grade, s_grid, 40.0) / 100.0)
    kappa_grid = _profile(drive.curvature, s_grid, 30.0)
    s_mid = s_arr + p.b                        # reference point: under the CG (rear axle is s_arr)
    gamma = np.interp(s_mid, s_grid, gamma_grid)
    gamma_rate = np.gradient(gamma, dt)
    kappa = np.interp(s_mid, s_grid, kappa_grid)
    if drive.roughness:
        ds_r = 0.05
        z_road = iso8608_profile(rng, length, ds_r, ROUGHNESS_GD[drive.roughness])
        s_r = np.arange(z_road.size) * ds_r
        z_rf = np.interp(s_arr + L, s_r, z_road)
        z_rr = np.interp(s_arr, s_r, z_road)
    else:
        z_rf = z_rr = np.zeros(n)

    # Load over time (sum m, sum m d, sum m d^2), ramped at each change.
    t = np.arange(n) * dt
    loads = sorted(drive.loads)
    sums = np.array([load_sums(p, name) for _, name in loads])
    S = np.tile(sums[0], (n, 1))
    for (t_c, _), new, old in zip(loads[1:], sums[1:], sums[:-1]):
        k0 = int(t_c / dt)
        if v_arr[min(k0, n - 1)] > 0.01:
            raise ValueError(f"load change at t = {t_c} s is not at standstill")
        w = np.clip((t - t_c) / drive.load_ramp_s, 0.0, 1.0)[:, None]
        S = np.where(t[:, None] >= t_c, old + w * (new - old), S)
    # The static trim is linear in the load, so two unit solutions give it for any load.
    c0 = static_equilibrium(p, 1.0, 0.0)[1]
    c1 = static_equilibrium(p, 0.0, 1.0)[1]
    theta_static = np.degrees(c0 * S[:, 0] + c1 * S[:, 1])
    # Quasi-static pitch the lamp should compensate: load plus the pitch from standing /
    # driving on the slope (not the transient squat / dive from accelerating).
    f_slope = G * np.sin(gamma)
    per_moment = np.where(f_slope > 0.0, slope_pitch_per_moment(p, False), slope_pitch_per_moment(p, True))
    theta_aim = theta_static + np.degrees(per_moment * (p.m_s + S[:, 0]) * f_slope * p.h)

    # ------------------------------------------------------------ plant loop (plain floats)
    x0 = static_equilibrium(p, S[0, 0], S[0, 1])
    zs, th, zuf, zur = (float(x) for x in x0)
    vs = om = vuf = vur = 0.0
    kf, kr, cf, cr, ktf, ktr = p.k_f, p.k_r, p.c_f, p.c_r, p.k_tf, p.k_tr
    a_, b_, h_, ms, iy = p.a, p.b, p.h, p.m_s, p.i_y
    fric, v0 = p.f_fric, 0.01
    ns = n // sub
    out = np.zeros((ns, 6))                    # mean over each sample: zdd_imu, thdd, om, th, (spare)
    A_l, V_l, G_l, GR_l = a_cmd.tolist(), v_arr.tolist(), gamma.tolist(), gamma_rate.tolist()
    ZF_l, ZR_l = z_rf.tolist(), z_rr.tolist()
    S0_l, S1_l, S2_l = S[:, 0].tolist(), S[:, 1].tolist(), S[:, 2].tolist()
    tanh, sin, cos = math.tanh, math.sin, math.cos
    acc = [0.0, 0.0, 0.0, 0.0]
    for k in range(n):
        ax, g = A_l[k], G_l[k]
        s0, s1, s2 = S0_l[k], S1_l[k], S2_l[k]
        cg = cos(g)
        zbf, zbr = zs + a_ * th, zs - b_ * th
        vbf, vbr = vs + a_ * om, vs - b_ * om
        ff = -kf * (zbf - zuf) - cf * (vbf - vuf) - fric * tanh((vbf - vuf) / v0)
        fr = -kr * (zbr - zur) - cr * (vbr - vur) - fric * tanh((vbr - vur) / v0)
        m_tot = ms + s0
        ax_eff = ax + G * sin(g)               # specific force along the road at the CG
        eta = p.anti_squat if ax_eff > 0.0 else p.anti_dive
        m_long = m_tot * ax_eff * h_           # nose-up moment on the body
        f_link = eta * m_long / L              # part carried by the links: front down, rear up
        fz = ff + fr - s0 * G * cg - ms * G * (cg - 1.0)
        my = a_ * (ff - f_link) - b_ * (fr + f_link) - s1 * G * cg + m_long
        m11, m12, m22 = m_tot, s1, iy + s2
        det = m11 * m22 - m12 * m12
        zdd = (fz * m22 - my * m12) / det
        thdd = (my * m11 - fz * m12) / det
        zufdd = (-ff + f_link + ktf * (ZF_l[k] - zuf)) / p.m_uf
        zurdd = (-fr - f_link + ktr * (ZR_l[k] - zur)) / p.m_ur
        vs += zdd * dt
        om += thdd * dt
        vuf += zufdd * dt
        vur += zurdd * dt
        zs += vs * dt
        th += om * dt
        zuf += vuf * dt
        zur += vur * dt
        acc[0] += zdd + p.x_imu * thdd
        acc[1] += thdd
        acc[2] += om
        acc[3] += th
        if k % sub == sub - 1:
            j = k // sub
            out[j, 0], out[j, 1], out[j, 2], out[j, 3] = (x / sub for x in acc)
            acc = [0.0, 0.0, 0.0, 0.0]

    # ------------------------------------------------------------ sensors at 100 Hz
    def mean(x: np.ndarray) -> np.ndarray:
        return x.reshape(ns, sub).mean(axis=1)

    a_s, v_s, g_s, gr_s, k_s = mean(a_cmd), mean(v_arr), mean(gamma), mean(gamma_rate), mean(kappa)
    zdd_imu, thdd_s, om_s, th_s = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    a_lat = v_s * v_s * k_s
    roll = -np.deg2rad(p.roll_gradient_deg_per_g) / G * a_lat
    # Turning. The wheel speed is the forward speed u of the rear axle. In the body frame
    # the longitudinal acceleration of a point is du/dt - r * v_y, with v_y its lateral
    # velocity. The rear axle slips sideways by the angle alpha_r (v_y = -u alpha_r), and a
    # point d ahead of it adds r * d. So the IMU sees  a_lat * alpha_r - r^2 d  on top of
    # du/dt: the first term is the same everywhere in the car, the second grows with the
    # distance from the rear axle. Both are quasi-steady (single-track model).
    yaw = v_s * k_s
    d_ra = p.x_imu + p.b
    idx_s = np.arange(sub - 1, n, sub)
    s_s = S[idx_s]
    alpha_r = p.rear_slip_gradient(s_s[:, 0], s_s[:, 1]) * a_lat
    a_turn = a_lat * alpha_r - yaw * yaw * d_ra
    a_lat_imu = a_lat + np.gradient(yaw, 1.0 / fs) * d_ra
    # Specific force in the road frame at the IMU, then rotated into the body frame.
    f_rx = a_s + G * np.sin(g_s) - p.z_imu * thdd_s + a_turn
    f_rz = zdd_imu + v_s * gr_s + G * np.cos(g_s) * np.cos(roll)
    f_body = np.stack([
        f_rx * np.cos(th_s) + f_rz * np.sin(th_s),
        a_lat_imu * np.cos(roll) + G * np.sin(roll),
        -f_rx * np.sin(th_s) + f_rz * np.cos(th_s),
    ], axis=1)
    w_body = np.rad2deg(np.stack([np.gradient(roll, 1.0 / fs), -(gr_s + om_s), v_s * k_s], axis=1))

    if ideal_sensors:
        inputs = np.concatenate([f_body, w_body, v_s[:, None]], axis=1)
    else:
        imu = imu_measurements(rng, cfg.imu, fs, f_body, w_body, v_s)
        v_wheel = wheel_speed_measurement(rng, cfg.imu, v_s, a_s)
        inputs = np.concatenate([imu, v_wheel[:, None]], axis=1)
    idx = idx_s
    gamma_ahead = np.interp(s_mid[idx] + cfg.camera.lookahead_m, s_grid, gamma_grid)
    cam, cam_bias = camera_pitch_measurements(rng, cfg.camera, fs, th_s, g_s, gamma_ahead, v_s)

    return {
        "inputs": inputs.astype(np.float64),
        "camera_pitch_deg": cam.astype(np.float64),
        "fs_hz": fs,
        "load_pitch_deg": theta_static[idx],
        "aim_pitch_deg": theta_aim[idx],
        "road_grade_deg": np.rad2deg(g_s),
        "body_pitch_deg": np.rad2deg(th_s),
        "speed_ms": v_s,
        "accel_ms2": a_s,
        "distance_m": s_arr[idx],
        "camera_bias_deg": cam_bias,
        "load_times": [tc for tc, _ in loads[1:]],
    }

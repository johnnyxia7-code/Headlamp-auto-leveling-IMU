"""Synthetic driving cycles: longitudinal acceleration and speed over time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class DriveCycle:
    accel: np.ndarray                  # along-road acceleration (m/s^2)
    speed: np.ndarray                  # m/s, >= 0
    distance: np.ndarray               # m
    stops: List[Tuple[int, int]]       # (start, end) sample index of standstill phases


def drive_cycle(rng: np.random.Generator, n: int, dt: float, max_speed_kmh: float) -> DriveCycle:
    """Generate a random but plausible drive: launches, cruising, braking and stops.

    A driver model tracks a sequence of target speeds with a random
    acceleration / deceleration demand and a jerk limit. Cruising adds
    small throttle fluctuations. Roughly one manoeuvre in five is a stop.
    """
    vmax = max_speed_kmh / 3.6
    a = np.zeros(n)
    v = np.zeros(n)
    jerk_max = rng.uniform(2.0, 8.0)
    v_now, a_now = 0.0, 0.0
    i = 0
    stops: List[Tuple[int, int]] = []

    # Start from standstill for a few seconds (lets a static estimate be formed).
    hold = int(rng.uniform(2.0, 8.0) / dt)
    stops.append((0, min(hold, n)))
    i = hold

    while i < n:
        if v_now < 0.5 or rng.random() > 0.2:
            v_tgt = rng.uniform(0.15, 1.0) * vmax if rng.random() < 0.7 else rng.uniform(5.0, 20.0)
        else:
            v_tgt = 0.0
        accel_lim = rng.uniform(0.6, 3.5) * (0.5 if v_now > 25 else 1.0)
        decel_lim = rng.uniform(0.8, 4.0) if rng.random() > 0.08 else rng.uniform(5.0, 8.5)
        k_track = rng.uniform(0.3, 0.8)

        # Track the target speed.
        while i < n and abs(v_tgt - v_now) > 0.3:
            a_dem = np.clip(k_track * (v_tgt - v_now) * 2.0, -decel_lim, accel_lim)
            a_now += np.clip(a_dem - a_now, -jerk_max * dt, jerk_max * dt)
            v_now = max(0.0, v_now + a_now * dt)
            if v_now == 0.0:
                a_now = 0.0
            a[i], v[i] = a_now, v_now
            i += 1

        # Cruise or stand still.
        if v_tgt == 0.0:
            dur = int(rng.uniform(2.0, 25.0) / dt)
            j = min(n, i + dur)
            a[i:j], v[i:j] = 0.0, 0.0
            stops.append((i, j))
            v_now, a_now, i = 0.0, 0.0, j
        else:
            dur = int(rng.uniform(3.0, 40.0) / dt)
            ou, theta, sig = 0.0, 0.5, rng.uniform(0.05, 0.25)
            for _ in range(dur):
                if i >= n:
                    break
                ou += -theta * ou * dt + sig * np.sqrt(dt) * rng.normal()
                a_tgt = ou + 0.3 * (v_tgt - v_now)
                a_now += np.clip(a_tgt - a_now, -jerk_max * dt, jerk_max * dt)
                v_now = max(0.0, v_now + a_now * dt)
                a[i], v[i] = a_now, v_now
                i += 1

    # Recompute acceleration from the (clipped) speed so they are consistent.
    a = np.gradient(v, dt)
    a[v <= 0.0] = 0.0
    distance = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
    return DriveCycle(accel=a, speed=v, distance=distance, stops=stops)

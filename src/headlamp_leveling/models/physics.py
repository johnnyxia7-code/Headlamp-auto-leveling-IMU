"""Model-free physics baseline: pitch from the direction of the acceleration change.

For a window of data the longitudinal acceleration ``a`` (from wheel speed)
is used as an instrument: the part of the measured specific force that is
correlated with ``a`` points along the road in the body frame, so

    theta = atan2(-cov(f_z, a), cov(f_x, a))

Constant accelerometer bias and constant road grade cancel out. With
``gyro_comp`` the integrated pitch rate removes the gravity leakage caused
by grade changes and by the vehicle's own squat / dive within the window,
which otherwise biases the estimate by roughly 1 / (1 + g * pitch_gradient).
"""

from __future__ import annotations

import numpy as np

from .. import G
from ..data.features import IDX


def _detrend(x: np.ndarray) -> np.ndarray:
    """Remove mean and linear trend from a 1-D signal."""
    t = np.arange(x.shape[0], dtype=np.float64)
    t -= t.mean()
    xc = x - x.mean()
    return xc - t * ((t @ xc) / (t @ t))


class PhysicsEstimator:
    def __init__(self, fs_hz: float, gyro_comp: bool = True, squat_comp: bool = True,
                 min_excitation_ms2: float = 0.0):
        self.fs_hz = fs_hz
        self.gyro_comp = gyro_comp
        self.squat_comp = squat_comp
        self.min_excitation = min_excitation_ms2

    def estimate(self, feats: np.ndarray) -> float:
        """feats: (W, C) unnormalised features of one window. Returns load pitch in degrees (NaN if unobservable)."""
        f = feats.astype(np.float64)
        ax, az, aw = f[:, IDX["ax"]], f[:, IDX["az"]], f[:, IDX["a_wheel"]]
        if aw.std() < max(self.min_excitation, 1e-3):
            return float("nan")
        if self.gyro_comp:
            phi = np.cumsum(-np.deg2rad(f[:, IDX["gy"]])) / self.fs_hz   # nose-up pitch relative to window start
            phi = _detrend(phi)                                           # removes gyro bias drift
            ax = ax - G * np.sin(phi)
            az = az - G * (np.cos(phi) - 1.0)
        ax, az, aw = _detrend(ax), _detrend(az), _detrend(aw)
        if self.squat_comp:
            # Dynamic pitch ~ K * a adds a -K * a^2 term to f_z; regress it out so
            # skewed acceleration distributions (hard braking) do not bias theta.
            a2 = _detrend(aw * aw)
            X = np.stack([aw, a2], axis=1)
            beta_z = np.linalg.lstsq(X, az, rcond=None)[0]
            beta_x = np.linalg.lstsq(X, ax, rcond=None)[0]
            return float(np.rad2deg(np.arctan2(-beta_z[0], beta_x[0])))
        return float(np.rad2deg(np.arctan2(-(az @ aw), ax @ aw)))

    def predict_windows(self, windows: np.ndarray) -> np.ndarray:
        return np.array([self.estimate(w) for w in windows])

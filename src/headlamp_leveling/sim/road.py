"""Road geometry: longitudinal grade, cross slope / curvature and ISO 8608 roughness."""

from __future__ import annotations

import numpy as np


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.ones(n) / n
    pad = np.pad(x, (n, n), mode="edge")
    return np.convolve(pad, k, mode="same")[n:-n]


def grade_profile(rng: np.random.Generator, length_m: float, ds: float, max_grade_pct: float) -> np.ndarray:
    """Smooth random road grade angle (rad, uphill positive) sampled every ``ds`` metres.

    Knots every 80-400 m are linearly interpolated and smoothed with a
    ~60 m moving average, which mimics vertical curves of real roads.
    """
    n = int(np.ceil(length_m / ds)) + 1
    s = np.arange(n) * ds
    knots = [0.0]
    while knots[-1] < length_m:
        knots.append(knots[-1] + rng.uniform(80.0, 400.0))
    knots = np.asarray(knots)
    grades = rng.uniform(-max_grade_pct, max_grade_pct, size=knots.size)
    flat = rng.random(knots.size) < 0.25          # keep some level stretches
    grades[flat] *= 0.1
    g = np.interp(s, knots, grades)
    g = _smooth(g, max(1, int(60.0 / ds)))
    return np.arctan(g / 100.0)


def curvature_profile(rng: np.random.Generator, length_m: float, ds: float) -> np.ndarray:
    """Road curvature (1/m, left positive): straights mixed with curves of 50-800 m radius."""
    n = int(np.ceil(length_m / ds)) + 1
    kappa = np.zeros(n)
    pos = 0.0
    while pos < length_m:
        seg = rng.uniform(50.0, 500.0)
        if rng.random() < 0.45:
            radius = rng.uniform(50.0, 800.0)
            i0, i1 = int(pos / ds), int(min(pos + seg, length_m) / ds)
            kappa[i0:i1] = rng.choice([-1.0, 1.0]) / radius
        pos += seg
    return _smooth(kappa, max(1, int(30.0 / ds)))


def iso8608_profile(rng: np.random.Generator, length_m: float, ds: float, gd_n0: float,
                    n0: float = 0.1, waviness: float = 2.0) -> np.ndarray:
    """Random road elevation (m) with ISO 8608 displacement PSD Gd(n) = Gd(n0) (n/n0)^-w.

    Synthesised in the spatial-frequency domain with random phases over
    0.01..10 cycles/m (wavelengths 0.1 m .. 100 m).
    """
    n = int(np.ceil(length_m / ds)) + 1
    nfft = int(2 ** np.ceil(np.log2(n)))
    freqs = np.fft.rfftfreq(nfft, d=ds)
    dn = freqs[1] - freqs[0]
    psd = np.zeros_like(freqs)
    band = (freqs >= 0.01) & (freqs <= 10.0)
    psd[band] = gd_n0 * (freqs[band] / n0) ** (-waviness)
    amp = np.sqrt(2.0 * psd * dn)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=freqs.size)
    spec = amp * np.exp(1j * phase) * nfft / 2.0
    z = np.fft.irfft(spec, n=nfft)[:n]
    return z - z.mean()

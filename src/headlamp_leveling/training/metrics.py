"""Accuracy metrics for pitch estimation and headlamp aim."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# Headlamp aim is specified in percent (cm drop per 10 m): 1 % = atan(0.01) = 0.573 deg.
DEG_PER_PERCENT = float(np.rad2deg(np.arctan(0.01)))


def regression_metrics(pred: np.ndarray, true: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    pred, true = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    valid = ~np.isnan(pred) & ~np.isnan(true)
    if mask is not None:
        valid &= mask
    n_total = int(mask.sum()) if mask is not None else int(true.size)
    if not valid.any():
        return {"n": 0, "coverage": 0.0}
    err = pred[valid] - true[valid]
    ae = np.abs(err)
    return {
        "n": int(valid.sum()),
        "coverage": float(valid.sum() / max(n_total, 1)),
        "mae_deg": float(ae.mean()),
        "rmse_deg": float(np.sqrt(np.mean(err ** 2))),
        "bias_deg": float(err.mean()),
        "p95_deg": float(np.percentile(ae, 95)),
        "max_deg": float(ae.max()),
        "within_0.1deg": float(np.mean(ae <= 0.1)),
        "within_0.2deg": float(np.mean(ae <= 0.2)),
        "mae_percent": float(ae.mean() / DEG_PER_PERCENT),
    }


def format_metrics(m: Dict[str, float]) -> str:
    if not m.get("n"):
        return "n=0"
    return (f"n={m['n']:5d} MAE={m['mae_deg']:.3f} deg RMSE={m['rmse_deg']:.3f} bias={m['bias_deg']:+.3f} "
            f"P95={m['p95_deg']:.3f} |e|<=0.1: {100 * m['within_0.1deg']:.1f}% <=0.2: {100 * m['within_0.2deg']:.1f}%")

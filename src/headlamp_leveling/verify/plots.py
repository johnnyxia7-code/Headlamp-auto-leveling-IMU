"""Time-series plot of one verification run (needs matplotlib)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

# Colours: one hue per quantity, used consistently in every panel.
C_TRUTH = "#1b1b1b"
C_EST = "#1f77b4"
C_LAMP = "#d95f02"
C_BODY = "#9a9a9a"
C_SPEED = "#2a7f62"
C_GRADE = "#8c6bb1"
C_BAD = "#c62828"


def plot_scenario(result: Dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc, raw, d = result["scenario"], result["raw"], result["data"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.5, 1.2, 0.6], "hspace": 0.12})
    ax_v, ax_p, ax_e, ax_s = axes

    def mark_events(ax) -> None:
        for t0, t1 in d.fault_windows:
            ax.axvspan(t0, t1, color=C_BAD, alpha=0.08, lw=0)
        for tc in d.load_times:
            ax.axvline(tc, color=C_TRUTH, ls=":", lw=1.0)

    # 1. What the car does.
    ax_v.plot(d.t_raw, raw["speed_ms"] * 3.6, color=C_SPEED, lw=1.4)
    ax_v.set_ylabel("speed (km/h)", color=C_SPEED)
    ax_g = ax_v.twinx()
    ax_g.plot(d.t_raw, np.tan(np.deg2rad(raw["road_grade_deg"])) * 100.0, color=C_GRADE, lw=1.2)
    ax_g.set_ylabel("road grade (%)", color=C_GRADE)
    lim = max(2.0, float(np.abs(ax_g.get_ylim()).max()))
    ax_g.set_ylim(-lim, lim)
    mark_events(ax_v)

    # 2. What the filter believes.
    ax_p.plot(d.t_raw, raw["body_pitch_deg"], color=C_BODY, lw=0.6, label="body pitch incl. squat / dive")
    ax_p.fill_between(d.t_est, d.est - 2 * d.est_std, d.est + 2 * d.est_std, color=C_EST, alpha=0.18, lw=0,
                      label="estimate ± 2σ")
    ax_p.plot(d.t_est, d.est, color=C_EST, lw=1.3, label="estimated pitch to compensate")
    ax_p.plot(d.t_est, d.est_truth, color=C_TRUTH, lw=1.3, ls="--", label="true pitch to compensate (load + slope)")
    ax_p.plot(d.t, d.correction, color=C_LAMP, lw=1.6, label="lamp correction")
    lo = min(float(d.est_truth.min()), float(d.correction.min())) - 0.8
    hi = max(float(d.est_truth.max()), float(d.correction.max())) + 0.8
    ax_p.set_ylim(lo, hi)
    ax_p.set_ylabel("pitch (deg)")
    ax_p.legend(loc="upper right", ncol=3, fontsize=8, frameon=False)
    mark_events(ax_p)

    # 3. What reaches the road.
    ax_e.axhspan(-0.25, 0.25, color="#2a7f62", alpha=0.10, lw=0)
    ax_e.axhline(0.0, color=C_TRUTH, lw=0.6)
    ax_e.plot(d.t, d.err, color=C_LAMP, lw=1.4)
    span = max(0.6, float(np.abs(d.err).max()) * 1.1)
    ax_e.set_ylim(-span, span)
    ax_e.set_ylabel("aim error (deg)\n+ = beam too high")
    mark_events(ax_e)

    # 4. Health.
    ax_s.fill_between(d.t_raw, 0, d.guard_failed.astype(float), step="post", color=C_BAD, alpha=0.5, lw=0)
    ax_s.fill_between(d.t, 1.2, 1.2 + d.failsafe.astype(float), step="post", color=C_TRUTH, alpha=0.7, lw=0)
    ax_s.set_yticks([0.5, 1.7])
    ax_s.set_yticklabels(["sensor\ndisqualified", "lamp in\nfail-safe"], fontsize=8)
    ax_s.set_ylim(0, 2.3)
    ax_s.set_xlabel("time (s)")
    mark_events(ax_s)

    for ax in (ax_v, ax_p, ax_e, ax_s):
        ax.spines["top"].set_visible(False)
        ax.grid(axis="x", color="#dddddd", lw=0.5)
    ax_v.set_xlim(0, float(d.t_raw[-1]))

    n_ok = sum(c.passed for c in result["checks"])
    verdict = "PASS" if result["passed"] else "FAIL"
    fig.suptitle(f"{sc.title}   —   {verdict} ({n_ok}/{len(result['checks'])} checks)", fontsize=12,
                 x=0.07, ha="left", y=0.965, color=C_TRUTH if result["passed"] else C_BAD)
    notes = []
    if d.load_times:
        notes.append("dotted line: load change")
    if d.fault_windows:
        notes.append("red band: fault present")
    notes.append("green band: ±0.25° target")
    fig.text(0.07, 0.925, "   ·   ".join(notes), fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.09, right=0.93, top=0.90, bottom=0.06)
    fig.savefig(path, dpi=130)
    plt.close(fig)

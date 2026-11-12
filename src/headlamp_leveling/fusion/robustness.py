"""Robustness evaluation: fault scenarios, with and without the input guard."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..sim.faults import SCENARIOS, inject
from ..sim.scenario import simulate_trip
from .run import closed_loop_series, run_ekf


def _metrics(raw, windows, cfg: Config, guard: bool, warmup_s: float) -> Dict[str, float]:
    res = run_ekf(raw, cfg, use_camera=True, guard=guard)
    cl = closed_loop_series(res, cfg)
    t, err, fsafe = cl["t"], np.abs(cl["err"]), cl["failsafe"]
    after = t >= warmup_s
    # Faults that end before the trip does: aim error once the fault is over (+30 s to recover).
    t_end = max((w.t1 for w in windows), default=0.0)
    rec = t >= t_end + 30.0 if windows and t_end + 60.0 < t[-1] else np.zeros_like(after)
    during = np.zeros_like(after)
    for w in windows:
        during |= (t >= w.t0) & (t <= w.t1 + 5.0)
    during &= after
    m = {
        "aim_mae_deg": float(err[after].mean()),
        "aim_max_deg": float(err[after].max()),
        "aim_mae_during_deg": float(err[during].mean()) if during.any() else float("nan"),
        "aim_mae_recovered_deg": float(err[rec].mean()) if rec.any() else float("nan"),
        "failsafe_frac": float(fsafe[after].mean()),
        "failsafe_at_end": bool(fsafe[-1]),
        "numeric_resets": int(res["numeric_resets"]),
        "finite": bool(np.all(np.isfinite(res["load_pitch_deg"]))),
    }
    if guard and windows and windows[0].kind not in ("spikes", "dropout"):
        g = res["guard"]
        fs = raw["fs_hz"]
        k0 = int(windows[0].t0 * fs)
        hit = np.flatnonzero(g["any_failed"][k0:])
        m["detect_s"] = float(hit[0] / fs) if hit.size else float("nan")
    return m


def _work(args) -> Tuple[int, Dict[str, Dict[str, Dict[str, float]]]]:
    cfg, i, seed, scenarios, warmup = args
    trip = simulate_trip(cfg, seed, raw=True)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sc in scenarios:
        raw, windows = inject(trip["raw"], sc, seed=i)
        out[sc] = {"unguarded": _metrics(raw, windows, cfg, False, warmup),
                   "guarded": _metrics(raw, windows, cfg, True, warmup)}
    return i, out


def evaluate_robustness(cfg: Config, n_trips: int = 12, out_dir: Optional[str] = None,
                        scenarios: Optional[List[str]] = None, workers: int = 0,
                        verbose: bool = True) -> Dict:
    """Run every fault scenario on the first ``n_trips`` test trips, guard off vs. on."""
    scenarios = scenarios or list(SCENARIOS)
    root = np.random.SeedSequence(cfg.dataset.seed)
    seeds = root.spawn(3)[2].spawn(max(n_trips, cfg.dataset.n_test_trips))[:n_trips]
    warmup = cfg.dataset.window_s
    jobs = [(cfg, i, s, scenarios, warmup) for i, s in enumerate(seeds)]
    workers = workers or min(os.cpu_count() or 1, n_trips)
    results: List[Dict] = [None] * n_trips        # type: ignore[list-item]
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            for i, r in pool.imap_unordered(_work, jobs):
                results[i] = r
                if verbose:
                    print(f"[robust] trip {sum(x is not None for x in results)}/{n_trips} done", flush=True)
    else:
        for job in jobs:
            i, r = _work(job)
            results[i] = r
            if verbose:
                print(f"[robust] trip {i + 1}/{n_trips} done", flush=True)

    report: Dict = {"trips": n_trips, "warmup_s": warmup, "scenarios": {}}
    for sc in scenarios:
        entry: Dict[str, Dict[str, float]] = {}
        for variant in ("unguarded", "guarded"):
            rows = [r[sc][variant] for r in results]
            agg = {}
            for key in ("aim_mae_deg", "aim_max_deg", "aim_mae_during_deg", "aim_mae_recovered_deg",
                        "failsafe_frac", "detect_s"):
                vals = np.array([row[key] for row in rows if key in row], float)
                vals = vals[np.isfinite(vals)]
                if key == "aim_max_deg":
                    agg[key] = float(vals.max()) if vals.size else float("nan")
                elif key == "detect_s":
                    agg["detect_s_median"] = float(np.median(vals)) if vals.size else float("nan")
                    agg["detected_trips"] = int(vals.size)
                else:
                    agg[key] = float(vals.mean()) if vals.size else float("nan")
            agg["failsafe_at_end_trips"] = int(sum(row["failsafe_at_end"] for row in rows))
            agg["non_finite_trips"] = int(sum(not row["finite"] for row in rows))
            agg["numeric_resets"] = int(sum(row["numeric_resets"] for row in rows))
            entry[variant] = agg
        entry["description"] = (SCENARIOS[sc].__doc__ or "").strip().split("\n")[0]
        report["scenarios"][sc] = entry

    out = Path(out_dir or "runs/robustness")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "robustness_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    (out / "robustness_report.md").write_text(render_markdown(report), encoding="utf-8")
    if verbose:
        print(render_markdown(report))
    return report


def render_markdown(r: Dict) -> str:
    def f(x: float, nd: int = 3) -> str:
        return "-" if x != x else f"{x:.{nd}f}"

    lines = [f"# Robustness evaluation ({r['trips']} test trips, aim error after {r['warmup_s']:.0f} s)", "",
             "Closed-loop headlamp aim error (deg) with the EKF (IMU + wheel + camera) and the leveling",
             "controller. *Unguarded*: missing samples are replaced by the last value, nothing else.",
             "*Guarded*: samples pass the resilient input layer first.", "",
             "| scenario | variant | aim MAE | worst | MAE during fault | MAE after recovery "
             "| time in fail-safe | fail-safe at end | detected after |",
             "|---|---|---|---|---|---|---|---|---|"]
    for sc, e in r["scenarios"].items():
        for variant in ("unguarded", "guarded"):
            a = e[variant]
            det = "-" if "detect_s_median" not in a or a["detect_s_median"] != a["detect_s_median"] \
                else f"{a['detect_s_median']:.1f} s ({a['detected_trips']}/{r['trips']})"
            bad = f" **({a['non_finite_trips']} trips NaN)**" if a["non_finite_trips"] else ""
            lines.append(f"| {sc if variant == 'unguarded' else ''} | {variant} | {f(a['aim_mae_deg'])}{bad} "
                         f"| {f(a['aim_max_deg'], 2)} | {f(a['aim_mae_during_deg'])} "
                         f"| {f(a['aim_mae_recovered_deg'])} | {100 * a['failsafe_frac']:.1f} % "
                         f"| {a['failsafe_at_end_trips']}/{r['trips']} | {det} |")
    lines += ["", "Scenarios:", ""]
    lines += [f"- `{sc}`: {e['description']}" for sc, e in r["scenarios"].items()]
    return "\n".join(lines) + "\n"

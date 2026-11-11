"""Run verification scenarios through the whole chain and report pass / fail."""

from __future__ import annotations

import json
from pathlib import Path
import copy
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import Config
from ..fusion.run import closed_loop_series, run_ekf
from ..sim.faults import apply as apply_fault
from ..sim.halfcar import simulate_scripted
from .criteria import CheckResult, RunData, evaluate_all
from .scenarios import SCENARIOS, VerifyScenario


def run_scenario(sc: VerifyScenario, cfg: Config, seed: int = 0) -> Dict:
    """Simulate one scenario and evaluate its criteria. Returns signals and check results."""
    raw = simulate_scripted(sc.drive, cfg, seed=seed)
    t_raw = np.arange(raw["inputs"].shape[0]) / raw["fs_hz"]
    t_total = float(t_raw[-1])
    windows = []
    for i, (channel, kind, t0, t1, value) in enumerate(sc.faults):
        apply_fault(raw, channel, kind, t0, min(t1, t_total + 1.0), value, seed=seed + i)
        windows.append((t0, min(t1, t_total)))
    # Several faults over the same span (e.g. glitches on three channels) count as one window.
    windows = sorted(set(windows))

    # The IMU position is installation data: the filter is given the plant's lever arm
    # (optionally with a calibration error).
    cfg = copy.deepcopy(cfg)
    cfg.fusion.imu_lever_x_m = sc.drive.vehicle.x_imu * sc.lever_arm_scale
    cfg.fusion.imu_lever_z_m = sc.drive.vehicle.z_imu * sc.lever_arm_scale
    res = run_ekf(raw, cfg, use_camera=sc.use_camera, guard=True)
    cl = closed_loop_series(res, cfg)
    n_est = res["load_pitch_deg"].size
    moving = np.flatnonzero(raw["speed_ms"] > 1.0)
    data = RunData(
        t=cl["t"], err=cl["err"], correction=cl["correction"], truth=cl["truth"], failsafe=cl["failsafe"],
        t_est=np.arange(n_est) / res["fs_hz"], est=res["load_pitch_deg"], est_std=res["load_pitch_std_deg"],
        est_truth=res["true_aim_pitch_deg"], t_raw=t_raw, speed=raw["speed_ms"],
        guard_failed=res["guard"]["any_failed"],
        t_drive=float(t_raw[moving[0]]) if moving.size else 0.0,
        load_times=raw["load_times"], fault_windows=windows, numeric_resets=int(res["numeric_resets"]))
    checks = evaluate_all(sc.criteria, data)
    return {"scenario": sc, "raw": raw, "res": res, "data": data, "checks": checks,
            "passed": all(c.passed for c in checks)}


def verify(cfg: Config, names: Optional[Sequence[str]] = None, out_dir: Optional[str] = None,
           plots: bool = True, seed: int = 0, verbose: bool = True) -> Dict:
    """Run the selected scenarios (default: all), write the report and optional plots."""
    names = list(names) if names else list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise KeyError(f"unknown scenario(s) {unknown}; choose from {sorted(SCENARIOS)}")
    out = Path(out_dir or "runs/verify")
    out.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []
    for name in names:
        r = run_scenario(SCENARIOS[name](), cfg, seed)
        results.append(r)
        if plots:
            from .plots import plot_scenario
            plot_scenario(r, out / f"{name}.png")
        if verbose:
            n_ok = sum(c.passed for c in r["checks"])
            print(f"[verify] {'PASS' if r['passed'] else 'FAIL'}  {name:26s} {n_ok}/{len(r['checks'])} checks",
                  flush=True)
            for c in r["checks"]:
                if not c.passed:
                    print(f"           x {c.name}: {c.measured}  (required: {c.requirement})", flush=True)

    report = {
        "seed": seed,
        "passed": sum(r["passed"] for r in results),
        "total": len(results),
        "scenarios": [{
            "name": r["scenario"].name, "title": r["scenario"].title, "group": r["scenario"].group,
            "purpose": r["scenario"].purpose, "passed": bool(r["passed"]),
            "duration_s": float(r["data"].t_raw[-1]),
            "checks": [{"name": c.name, "requirement": c.requirement, "measured": c.measured,
                        "passed": bool(c.passed)} for c in r["checks"]],
        } for r in results],
    }
    with open(out / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    (out / "report.md").write_text(render_markdown(report, plots), encoding="utf-8")
    if verbose:
        print(f"[verify] {report['passed']}/{report['total']} scenarios passed -> {out / 'report.md'}")
    return report


def render_markdown(report: Dict, plots: bool = True) -> str:
    lines = ["# Verification report", "",
             "Plant: half-car model (`sim/halfcar.py`), independent of the filter's internal model. ",
             "Chain under test: input layer → EKF → leveling controller → lamp.", "",
             f"**{report['passed']} of {report['total']} scenarios pass all their checks.**", "",
             "| Scenario | Group | Result | Checks | Failed checks |", "|---|---|---|---|---|"]
    for s in report["scenarios"]:
        failed = [c["name"] for c in s["checks"] if not c["passed"]]
        lines.append(f"| [{s['title']}](#{s['name']}) | {s['group']} | {'PASS' if s['passed'] else '**FAIL**'} "
                     f"| {sum(c['passed'] for c in s['checks'])}/{len(s['checks'])} | {', '.join(failed) or '–'} |")
    for s in report["scenarios"]:
        lines += ["", f"## {s['name']}", "", f"**{s['title']}** ({s['duration_s']:.0f} s)", "", s["purpose"], "",
                  "| Check | Requirement | Measured | Result |", "|---|---|---|---|"]
        for c in s["checks"]:
            lines.append(f"| {c['name']} | {c['requirement']} | {c['measured']} | "
                         f"{'pass' if c['passed'] else '**FAIL**'} |")
        if plots:
            lines += ["", f"![{s['name']}]({s['name']}.png)"]
    return "\n".join(lines) + "\n"

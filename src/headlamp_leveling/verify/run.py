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
    veh = sc.drive.vehicle
    cfg.fusion.imu_lever_x_m = veh.x_imu * sc.lever_arm_scale
    cfg.fusion.imu_lever_z_m = veh.z_imu * sc.lever_arm_scale
    cfg.fusion.imu_rear_axle_x_m = (veh.x_imu + veh.b) * sc.lever_arm_scale
    # Vehicle data, not installation data: the filter knows the side-slip gradient of the
    # unladen car. The plant's true value follows the load (up to 30 % more).
    cfg.fusion.rear_slip_rad_per_ms2 = veh.rear_slip_gradient()
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
           plots: bool = True, seed: int = 0, n_seeds: int = 1, verbose: bool = True) -> Dict:
    """Run the selected scenarios (default: all), write the report and optional plots.

    With ``n_seeds`` > 1 every scenario is repeated with seeds ``seed .. seed + n_seeds - 1``.
    The seed draws the sensor errors (biases, mounting misalignment, noise), the camera
    bias and the road profile, so this shows how robust each verdict is. Plots and the
    detailed check tables are for the first seed.
    """
    names = list(names) if names else list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise KeyError(f"unknown scenario(s) {unknown}; choose from {sorted(SCENARIOS)}")
    out = Path(out_dir or "runs/verify")
    out.mkdir(parents=True, exist_ok=True)
    seeds = [seed + i for i in range(max(1, n_seeds))]
    scenarios: List[Dict] = []
    for name in names:
        runs = []
        for k, sd in enumerate(seeds):
            r = run_scenario(SCENARIOS[name](), cfg, sd)
            if k == 0:
                first = r
                if plots:
                    from .plots import plot_scenario
                    plot_scenario(r, out / f"{name}.png")
            runs.append({"seed": sd, "passed": bool(r["passed"]),
                         "checks": [{"name": c.name, "requirement": c.requirement, "measured": c.measured,
                                     "passed": bool(c.passed)} for c in r["checks"]]})
        n_pass = sum(x["passed"] for x in runs)
        scenarios.append({
            "name": name, "title": first["scenario"].title, "group": first["scenario"].group,
            "purpose": first["scenario"].purpose, "passed": n_pass == len(runs), "seeds_passed": n_pass,
            "duration_s": float(first["data"].t_raw[-1]), "checks": runs[0]["checks"], "runs": runs,
        })
        if verbose:
            tag = "PASS" if n_pass == len(runs) else "FAIL"
            print(f"[verify] {tag}  {name:26s} {n_pass}/{len(runs)} seeds", flush=True)
            for x in runs:
                for c in x["checks"]:
                    if not c["passed"]:
                        print(f"           x seed {x['seed']}: {c['name']}: {c['measured']}", flush=True)

    report = {"seeds": seeds, "passed": sum(s["passed"] for s in scenarios), "total": len(scenarios),
              "scenarios": scenarios}
    with open(out / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    (out / "report.md").write_text(render_markdown(report, plots), encoding="utf-8")
    if verbose:
        print(f"[verify] {report['passed']}/{report['total']} scenarios pass with every seed "
              f"-> {out / 'report.md'}")
    return report


def render_markdown(report: Dict, plots: bool = True) -> str:
    n = len(report["seeds"])
    lines = ["# Verification report", "",
             "Plant: half-car model (`sim/halfcar.py`), independent of the filter's internal model. ",
             "Chain under test: input layer → EKF → leveling controller → lamp.", "",
             f"Each scenario was run with {n} seed{'s' if n > 1 else ''} "
             f"({', '.join(str(x) for x in report['seeds'])}). The seed draws the sensor errors, the camera "
             "bias and the road profile.", "",
             f"**{report['passed']} of {report['total']} scenarios pass all their checks with every seed.**", "",
             "| Scenario | Group | Seeds passed | Checks that failed (seeds) |", "|---|---|---|---|"]
    for s in report["scenarios"]:
        failed: Dict[str, List[int]] = {}
        for run in s["runs"]:
            for c in run["checks"]:
                if not c["passed"]:
                    failed.setdefault(c["name"], []).append(run["seed"])
        txt = "; ".join(f"{k} ({', '.join(map(str, v))})" for k, v in failed.items()) or "–"
        mark = f"{s['seeds_passed']}/{n}" if s["passed"] else f"**{s['seeds_passed']}/{n}**"
        lines.append(f"| [{s['title']}](#{s['name']}) | {s['group']} | {mark} | {txt} |")
    for s in report["scenarios"]:
        lines += ["", f"## {s['name']}", "", f"**{s['title']}** ({s['duration_s']:.0f} s)", "", s["purpose"], ""]
        header = "| Check | Requirement | " + " | ".join(f"Seed {x}" for x in report["seeds"]) + " |"
        lines += [header, "|---|---|" + "---|" * n]
        for i, c in enumerate(s["checks"]):
            cells = []
            for run in s["runs"]:
                rc = run["checks"][i]
                cells.append(rc["measured"] if rc["passed"] else f"**{rc['measured']} (fail)**")
            lines.append(f"| {c['name']} | {c['requirement']} | " + " | ".join(cells) + " |")
        if plots:
            lines += ["", f"Seed {report['seeds'][0]}:", "", f"![{s['name']}]({s['name']}.png)"]
    return "\n".join(lines) + "\n"

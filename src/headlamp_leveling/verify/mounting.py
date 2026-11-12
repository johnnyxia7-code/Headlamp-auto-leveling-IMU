"""IMU mounting-location study on the half-car verification plant.

Where the IMU sits changes what its accelerometers see:

    * vertical channel:      + x * pitch acceleration   (x: ahead of the pitch axis)
    * longitudinal channel:  - z * pitch acceleration   (z: above the pitch axis)
    * longitudinal channel:  - d * yaw rate^2           (d: ahead of the rear axle)
    * vertical channel:      heave + x * pitch acceleration on a rough road

The study runs the same scenarios and seeds with the IMU at each candidate
location, three times: with the filter told the true position, told 70 % of it,
and told nothing. Only the IMU position changes between the runs of one seed
(same road, same sensor errors), so differences are caused by the location.

A second part varies what a location changes but a rigid-body model cannot
predict: vibration level, residual mounting misalignment and accelerometer bias
drift. Those need measurements on the real body; the sweep shows how much each
one would cost.
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config
from .criteria import SteadyAccuracy
from .run import run_scenario
from .scenarios import SCENARIOS


@dataclass(frozen=True)
class MountLocation:
    name: str
    title: str
    behind_front_axle_m: float      # negative = ahead of the front axle
    height_m: float                 # above the road
    carrier: str                    # what the sensor would be fixed to

    def lever(self, veh) -> Tuple[float, float]:
        """(x forward of the CG, z above the CG) on a given vehicle."""
        return veh.a - self.behind_front_axle_m, self.height_m - veh.h


# Positions for the reference sedan of the plant (wheelbase 2.75 m, CG 1.25 m behind the
# front axle and 0.55 m high). All are on the vehicle centre line.
LOCATIONS: Dict[str, MountLocation] = {loc.name: loc for loc in (
    MountLocation("overhead_console", "Overhead console", 1.30, 1.22, "roof header, in a trim module"),
    MountLocation("under_rear_seat", "Under the rear seat", 2.15, 0.32, "floor pan / rear cross-member"),
    MountLocation("floor_centre", "Middle of the floor (centre tunnel)", 1.45, 0.33, "centre tunnel, sheet metal"),
    MountLocation("instrument_panel", "Lower instrument panel", 0.80, 0.60, "cross-car beam"),
    MountLocation("headlamp", "In the headlamp / front-end ECU", -0.75, 0.65, "lamp housing or front-end carrier"),
    MountLocation("rear_axle_floor", "Luggage floor above the rear axle", 2.75, 0.40, "rear floor / axle cross-member"),
)}

# (key, label, multiple of the true position given to the filter)
VARIANTS: Tuple[Tuple[str, str, float], ...] = (
    ("known", "position known", 1.0),
    ("error30", "position 30 % wrong", 0.7),
    ("unknown", "position not configured", 0.0),
)

DEFAULT_SCENARIOS = ("city_loaded", "city_no_camera", "sporty_driving", "rough_road", "curvy_road", "urban_turns",
                     "hill_route", "soft_vehicle")

# Sensor-error sweeps at one location: (key, label, {ImuConfig field: value}).
SENSITIVITY_LOCATION = "floor_centre"
SENSITIVITY_SCENARIOS = ("city_loaded", "city_no_camera", "rough_road")
SENSITIVITY: Tuple[Tuple[str, str, Dict[str, float]], ...] = (
    ("base", "baseline (0.15 m/s² vibration at 100 km/h, 0.05° misalignment)", {}),
    ("vib3", "vibration × 3", {"vibration_ms2": 0.45}),
    ("vib6", "vibration × 6", {"vibration_ms2": 0.90}),
    ("mis015", "mounting misalignment 0.15° (1σ)", {"misalign_deg": 0.15}),
    ("mis030", "mounting misalignment 0.30° (1σ)", {"misalign_deg": 0.30}),
    ("walk10", "accelerometer bias drift × 10 (about 3.5 mg in 5 min)", {"accel_bias_walk_mg_s": 0.2}),
    ("walk30", "accelerometer bias drift × 30 (about 10 mg in 5 min)", {"accel_bias_walk_mg_s": 0.6}),
)


def _run(task: Tuple) -> Dict:
    """One run. ``task`` = (key, cfg, location, scale, scenario, seed, imu overrides)."""
    key, cfg, loc_name, scale, sc_name, seed, imu = task
    cfg = copy.deepcopy(cfg)
    for field_name, value in imu.items():
        setattr(cfg.imu, field_name, value)
    sc = SCENARIOS[sc_name]()
    veh = sc.drive.vehicle
    x, z = LOCATIONS[loc_name].lever(veh)
    sc = replace(sc, drive=replace(sc.drive, vehicle=replace(veh, x_imu=x, z_imu=z)), lever_arm_scale=scale)
    r = run_scenario(sc, cfg, seed)
    d = r["data"]
    acc = next((c for c in sc.criteria if isinstance(c, SteadyAccuracy)), SteadyAccuracy())
    m = d.settled(acc.converge_s, acc.step_grace_s, acc.recover_s)
    e = d.err[m]
    conv = next((c.measured for c in r["checks"] if c.name == "convergence"), "nan s")
    return {"key": key, "passed": bool(r["passed"]), "mae": float(np.abs(e).mean()), "bias": float(e.mean()),
            "p95": float(np.percentile(np.abs(e), 95)), "worst": float(np.abs(e).max()),
            "converge_s": float(conv.split()[0]), "failsafe": bool(d.failsafe.any()),
            "failed": [c.name for c in r["checks"] if not c.passed]}


def _stats(runs: Sequence[Dict]) -> Dict:
    mae = np.array([r["mae"] for r in runs])
    return {"runs": len(runs), "passed": int(sum(r["passed"] for r in runs)), "mae_mean": float(mae.mean()),
            "mae_worst": float(mae.max()), "abs_bias_mean": float(np.mean([abs(r["bias"]) for r in runs])),
            "bias_mean": float(np.mean([r["bias"] for r in runs])),
            "p95_worst": float(max(r["p95"] for r in runs)), "worst": float(max(r["worst"] for r in runs)),
            "converge_median_s": float(np.median([r["converge_s"] for r in runs])),
            "converge_worst_s": float(max(r["converge_s"] for r in runs)),
            "failsafe_runs": int(sum(r["failsafe"] for r in runs))}


def mounting_study(cfg: Config, locations: Optional[Sequence[str]] = None,
                   scenarios: Optional[Sequence[str]] = None, n_seeds: int = 4, out_dir: Optional[str] = None,
                   sensitivity: bool = True, plots: bool = True, workers: int = 0, verbose: bool = True) -> Dict:
    """Run the study and write ``report.json``, ``report.md`` and (optionally) ``mounting.png``.

    Finished runs are appended to ``runs.jsonl`` in the output directory and reused,
    so an interrupted study continues where it stopped. Delete the file to start over.
    """
    loc_names = list(locations) if locations else list(LOCATIONS)
    sc_names = list(scenarios) if scenarios else list(DEFAULT_SCENARIOS)
    for n in loc_names:
        if n not in LOCATIONS:
            raise KeyError(f"unknown location {n!r}; choose from {sorted(LOCATIONS)}")
    for n in sc_names:
        if n not in SCENARIOS:
            raise KeyError(f"unknown scenario {n!r}; choose from {sorted(SCENARIOS)}")
    out = Path(out_dir or "runs/mounting")
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(max(1, n_seeds)))

    tasks: List[Tuple] = []
    for loc in loc_names:
        for vkey, _, scale in VARIANTS:
            for sc in sc_names:
                for sd in seeds:
                    tasks.append((f"loc|{loc}|{vkey}|{sc}|{sd}", cfg, loc, scale, sc, sd, {}))
    sens_loc = SENSITIVITY_LOCATION if SENSITIVITY_LOCATION in loc_names else loc_names[0]
    sens_sc = [s for s in SENSITIVITY_SCENARIOS if s in sc_names] or sc_names[:1]
    if sensitivity:
        for skey, _, imu in SENSITIVITY:
            for sc in sens_sc:
                for sd in seeds:
                    tasks.append((f"sens|{skey}|{sc}|{sd}", cfg, sens_loc, 1.0, sc, sd, imu))

    cache_path = out / "runs.jsonl"
    done: Dict[str, Dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["key"]] = rec
    todo = [t for t in tasks if t[0] not in done]
    if verbose:
        print(f"[mount] {len(tasks)} runs, {len(tasks) - len(todo)} already done", flush=True)
    workers = workers or min(os.cpu_count() or 1, max(1, len(todo)))
    with open(cache_path, "a", encoding="utf-8") as cache:
        def keep(rec: Dict) -> None:
            done[rec["key"]] = rec
            cache.write(json.dumps(rec) + "\n")
            cache.flush()
            if verbose and len(done) % 25 == 0:
                print(f"[mount] {len(done)}/{len(tasks)}", flush=True)
        if workers > 1 and len(todo) > 1:
            with mp.get_context("spawn").Pool(workers) as pool:
                for rec in pool.imap_unordered(_run, todo, chunksize=1):
                    keep(rec)
        else:
            for t in todo:
                keep(_run(t))

    veh = SCENARIOS[sc_names[0]]().drive.vehicle
    report: Dict = {"seeds": seeds, "scenarios": sc_names, "locations": [], "sensitivity": [],
                    "sensitivity_location": sens_loc, "sensitivity_scenarios": sens_sc,
                    "vehicle": {"wheelbase_m": veh.wheelbase, "cg_behind_front_axle_m": veh.a, "cg_height_m": veh.h}}
    for loc in loc_names:
        L = LOCATIONS[loc]
        x, z = L.lever(veh)
        entry = {"name": loc, "title": L.title, "carrier": L.carrier, "behind_front_axle_m": L.behind_front_axle_m,
                 "height_m": L.height_m, "x_cg_m": x, "z_cg_m": z, "ahead_of_rear_axle_m": x + veh.b,
                 "variants": {}, "by_scenario": {}}
        for vkey, _, _ in VARIANTS:
            runs = [done[f"loc|{loc}|{vkey}|{sc}|{sd}"] for sc in sc_names for sd in seeds]
            entry["variants"][vkey] = _stats(runs)
            entry["by_scenario"][vkey] = {sc: _stats([done[f"loc|{loc}|{vkey}|{sc}|{sd}"] for sd in seeds])
                                          for sc in sc_names}
        report["locations"].append(entry)
    if sensitivity:
        for skey, label, _ in SENSITIVITY:
            runs = [done[f"sens|{skey}|{sc}|{sd}"] for sc in sens_sc for sd in seeds]
            report["sensitivity"].append({"name": skey, "label": label, **_stats(runs)})

    with open(out / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    if plots:
        plot_mounting(report, out / "mounting.png")
    (out / "report.md").write_text(render_markdown(report, plots), encoding="utf-8")
    if verbose:
        for e in report["locations"]:
            k, u = e["variants"]["known"], e["variants"]["unknown"]
            print(f"[mount] {e['name']:18s} known: MAE {k['mae_mean']:.3f} (worst {k['mae_worst']:.3f}), "
                  f"{k['passed']}/{k['runs']} pass | not configured: MAE {u['mae_mean']:.3f}, "
                  f"{u['passed']}/{u['runs']} pass", flush=True)
        print(f"[mount] -> {out / 'report.md'}")
    return report


def render_markdown(report: Dict, plots: bool = True) -> str:
    v = report["vehicle"]
    n_runs = len(report["scenarios"]) * len(report["seeds"])
    lines = [
        "# IMU mounting-location study", "",
        "Plant: half-car model (`sim/halfcar.py`). Chain under test: input layer → EKF → leveling controller → lamp.",
        "",
        f"Reference vehicle: wheelbase {v['wheelbase_m']:.2f} m, centre of gravity {v['cg_behind_front_axle_m']:.2f} m "
        f"behind the front axle and {v['cg_height_m']:.2f} m above the road. All locations are on the centre line.",
        "",
        f"Each location was run with {len(report['scenarios'])} scenarios × {len(report['seeds'])} seeds = {n_runs} runs "
        "per column. Between locations only the IMU position changes: road, load and sensor errors of a seed are "
        "the same. \"Aim error\" is the steady-state error of the lamp aim (after convergence).", "",
        "## Locations", "",
        "| Location | Fixed to | Behind front axle | Height | Ahead of CG | Above CG | Ahead of rear axle |",
        "|---|---|---|---|---|---|---|"]
    for e in report["locations"]:
        lines.append(f"| {e['title']} | {e['carrier']} | {e['behind_front_axle_m']:+.2f} m | {e['height_m']:.2f} m | "
                     f"{e['x_cg_m']:+.2f} m | {e['z_cg_m']:+.2f} m | {e['ahead_of_rear_axle_m']:+.2f} m |")
    lines += ["", "## Result by location", ""]
    if plots:
        lines += ["![aim error by location](mounting.png)", ""]
    for vkey, vlabel, _ in VARIANTS:
        lines += [f"### Filter: {vlabel}", "",
                  "| Location | Runs passed | Aim error, mean of runs | Worst run | Mean bias | Worst P95 | "
                  "Convergence, median / worst |", "|---|---|---|---|---|---|---|"]
        for e in report["locations"]:
            s = e["variants"][vkey]
            lines.append(f"| {e['title']} | {s['passed']}/{s['runs']} | {s['mae_mean']:.3f}° | {s['mae_worst']:.3f}° | "
                         f"{s['bias_mean']:+.3f}° | {s['p95_worst']:.3f}° | {s['converge_median_s']:.0f} s / "
                         f"{s['converge_worst_s']:.0f} s |")
        lines.append("")
    for vkey, vlabel, _ in VARIANTS:
        if vkey == "error30":
            continue
        lines += [f"### By scenario, filter: {vlabel}", "",
                  "Mean aim error of the seeds (worst seed in brackets), in degrees.", "",
                  "| Location | " + " | ".join(f"`{s}`" for s in report["scenarios"]) + " |",
                  "|---|" + "---|" * len(report["scenarios"])]
        for e in report["locations"]:
            cells = [f"{e['by_scenario'][vkey][s]['mae_mean']:.2f} ({e['by_scenario'][vkey][s]['mae_worst']:.2f})"
                     for s in report["scenarios"]]
            lines.append(f"| {e['title']} | " + " | ".join(cells) + " |")
        lines.append("")
    if report["sensitivity"]:
        title = next(e["title"] for e in report["locations"] if e["name"] == report["sensitivity_location"])
        lines += ["## Sensitivity to what the rigid-body model does not predict", "",
                  f"IMU at \"{title}\", position known. Scenarios: "
                  + ", ".join(f"`{s}`" for s in report["sensitivity_scenarios"])
                  + f" × {len(report['seeds'])} seeds. How strongly a real location vibrates, how well its "
                  "alignment holds and how much its temperature moves must be measured on the body.", "",
                  "| Condition | Runs passed | Aim error, mean of runs | Worst run | Convergence, median / worst |",
                  "|---|---|---|---|---|"]
        for s in report["sensitivity"]:
            lines.append(f"| {s['label']} | {s['passed']}/{s['runs']} | {s['mae_mean']:.3f}° | {s['mae_worst']:.3f}° | "
                         f"{s['converge_median_s']:.0f} s / {s['converge_worst_s']:.0f} s |")
        lines.append("")
    return "\n".join(lines) + "\n"


def plot_mounting(report: Dict, path: Path) -> None:
    """Aim error by location: one bar per filter variant, a tick for the worst run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#2a78d6", "#eb6834", "#1baf7a")          # fixed order: known, 30 % wrong, not configured
    ink, muted, grid = "#1b1b1b", "#555555", "#dddddd"
    locs = report["locations"]
    fig, ax = plt.subplots(figsize=(10.5, 0.95 * len(locs) + 2.0))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    h = 0.24
    y0 = np.arange(len(locs))[::-1].astype(float)
    xmax = 0.0
    for i, (vkey, vlabel, _) in enumerate(VARIANTS):
        mean = np.array([e["variants"][vkey]["mae_mean"] for e in locs])
        worst = np.array([e["variants"][vkey]["mae_worst"] for e in locs])
        y = y0 + (1 - i) * (h + 0.03)
        ax.barh(y, mean, height=h, color=colors[i], label=f"filter: {vlabel}", lw=0)
        ax.plot(worst, y, ls="none", marker="|", ms=11, mew=2.0, color=colors[i])
        for yy, m, w in zip(y, mean, worst):
            ax.text(w + 0.006, yy, f"{m:.2f} / {w:.2f}", va="center", fontsize=8, color=muted)
        xmax = max(xmax, float(worst.max()))
    ax.axvline(0.15, color=ink, lw=0.8, ls="--")
    ax.text(0.152, y0[0] + 0.55, "0.15° target", fontsize=8, color=muted, va="bottom")
    ax.set_yticks(y0)
    ax.set_yticklabels([f"{e['title']}\n{e['x_cg_m']:+.2f} m fwd, {e['z_cg_m']:+.2f} m up of CG" for e in locs],
                       fontsize=9, color=ink)
    ax.set_xlim(0, xmax * 1.22)
    ax.set_ylim(y0[-1] - 0.6, y0[0] + 0.85)
    ax.set_xlabel("steady-state aim error (deg): bar = mean of runs, tick = worst run; labels = mean / worst",
                  fontsize=9, color=muted)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=grid, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize=9, frameon=False)
    n = len(report["scenarios"]) * len(report["seeds"])
    fig.suptitle(f"Lamp aim error by IMU location ({n} runs per bar)", fontsize=12, x=0.02, ha="left", color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)

"""Evaluation: window-level accuracy and closed-loop headlamp aim error on the test split."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..control.leveling import LevelingController
from ..data.dataset import WindowDataset
from ..data.features import IDX, build_features
from ..data.generate import load_splits
from ..data.tripset import TripSet
from ..models.physics import PhysicsEstimator
from .metrics import DEG_PER_PERCENT, format_metrics, regression_metrics
from .train import load_checkpoint, predict

Estimator = Callable[[np.ndarray], np.ndarray]   # (N, W, C) raw windows -> (N,) load pitch (deg)


def model_estimator(ckpt: Dict, batch: int = 256) -> Estimator:
    model, norm = ckpt["model"], ckpt["normalizer_obj"]

    @torch.no_grad()
    def run(windows: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(windows), batch):
            x = torch.from_numpy(np.ascontiguousarray(norm(windows[i:i + batch]).transpose(0, 2, 1)))
            out.append(model(x)[:, 0].numpy())
        return np.concatenate(out) if out else np.zeros(0)
    return run


def closed_loop(trips: TripSet, estimator: Estimator, window: int, cfg) -> Dict[str, float]:
    """Run estimator + LevelingController over whole trips; measure residual aim error.

    Residual aim error = true static load pitch - applied correction (deg);
    positive means the beam points higher than the nominal setting (glare).
    """
    fs = trips.fs_hz
    period = max(1, int(round(cfg.control.update_period_s * fs)))
    dt = period / fs
    errors, errors_warm = [], []
    for i in range(len(trips)):
        tr = trips.trip(i)
        feats = build_features(tr["inputs"], fs)
        ends = np.arange(period - 1, len(feats), period)
        full = ends[ends >= window - 1]
        est = np.full(ends.size, np.nan)
        exc = np.zeros(ends.size)
        if full.size:
            wins = np.stack([feats[e - window + 1:e + 1] for e in full])
            est[ends >= window - 1] = estimator(wins)
            exc[ends >= window - 1] = wins[:, :, IDX["a_wheel"]].std(axis=1)
        ctrl = LevelingController(cfg.control)
        for k, e in enumerate(ends):
            seg = feats[e - period + 1:e + 1]
            standstill = bool(np.all(seg[:, IDX["v_wheel"]] < 0.1))
            tilt = float(np.rad2deg(np.arctan2(seg[:, IDX["ax"]].mean(), seg[:, IDX["az"]].mean())))
            out = ctrl.update(dt, est[k], exc[k], standstill, tilt)
            err = float(tr["load_pitch_deg"][e] - out.correction_deg)
            errors.append(err)
            if e / fs >= 60.0:
                errors_warm.append(err)
    errors, errors_warm = np.abs(errors), np.abs(errors_warm)

    def stats(a: np.ndarray) -> Dict[str, float]:
        return {"mae_deg": float(a.mean()), "p95_deg": float(np.percentile(a, 95)),
                "within_0.2deg": float(np.mean(a <= 0.2)),
                "within_0.25percent": float(np.mean(a <= 0.25 * DEG_PER_PERCENT)),
                "within_0.5percent": float(np.mean(a <= 0.5 * DEG_PER_PERCENT))}
    return {"all": stats(errors), "after_60s": stats(errors_warm)}


def evaluate(ckpt_path: str, data_dir: Optional[str] = None, out_dir: Optional[str] = None,
             split: str = "test", verbose: bool = True) -> Dict:
    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt["cfg"]
    trips = load_splits(data_dir or cfg.dataset.out_dir, [split])[split]
    W = ckpt["window"]
    S = max(1, int(round(cfg.dataset.stride_s * trips.fs_hz)))
    ds = WindowDataset(trips, W, S, normalizer=ckpt["normalizer_obj"])
    pred, y, exc = predict(ckpt["model"], DataLoader(ds, batch_size=512), torch.device("cpu"))
    phys = PhysicsEstimator(trips.fs_hz, min_excitation_ms2=cfg.control.min_excitation_ms2)
    phys_pred = np.array([phys.estimate(ds.raw_window(k)) for k in range(len(ds))])
    excited = exc >= cfg.control.min_excitation_ms2

    report: Dict = {"checkpoint": str(ckpt_path), "split": split, "windows": len(ds),
                    "excited_fraction": float(excited.mean())}
    report["window_level"] = {
        "model_all": regression_metrics(pred[:, 0], y[:, 0]),
        "model_excited": regression_metrics(pred[:, 0], y[:, 0], excited),
        "physics_excited": regression_metrics(phys_pred, y[:, 0], excited),
        "grade_mae_deg": float(np.mean(np.abs(pred[:, 1] - y[:, 1]))),
    }

    # Breakdowns by true load pitch and by road / load category.
    groups: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    bins = [(-9, 0.25), (0.25, 0.75), (0.75, 1.25), (1.25, 9)]
    for lo, hi in bins:
        groups["load_pitch_bin"][f"[{lo:g},{hi:g})"] = (y[:, 0] >= lo) & (y[:, 0] < hi)
    for key in ("grade_class", "roughness_class", "load_case"):
        vals = np.array([trips.meta[t].get(key, "n/a") for t in ds.trip_ids])
        for v in np.unique(vals):
            groups[key][str(v)] = vals == v
    report["breakdown"] = {g: {name: {"model": regression_metrics(pred[:, 0], y[:, 0], m & excited),
                                      "physics": regression_metrics(phys_pred, y[:, 0], m & excited)}
                               for name, m in members.items()}
                           for g, members in groups.items()}

    report["closed_loop"] = {
        "model": closed_loop(trips, model_estimator(ckpt), W, cfg),
        "physics": closed_loop(trips, phys.predict_windows, W, cfg),
    }

    out = Path(out_dir or Path(ckpt_path).parent / f"eval_{split}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    if verbose:
        wl = report["window_level"]
        print(f"[eval] model   (all)     {format_metrics(wl['model_all'])}")
        print(f"[eval] model   (excited) {format_metrics(wl['model_excited'])}")
        print(f"[eval] physics (excited) {format_metrics(wl['physics_excited'])}")
        for name, r in report["closed_loop"].items():
            a = r["after_60s"]
            print(f"[eval] closed loop {name:8s} after 60 s: MAE={a['mae_deg']:.3f} deg P95={a['p95_deg']:.3f} "
                  f"|aim err|<=0.25%: {100 * a['within_0.25percent']:.1f}%")
        print(f"[eval] report -> {out / 'report.md'}")
    return report


def render_markdown(r: Dict) -> str:
    def row(name: str, m: Dict) -> str:
        if not m.get("n"):
            return f"| {name} | 0 | - | - | - | - | - |"
        return (f"| {name} | {m['n']} | {m['mae_deg']:.3f} | {m['rmse_deg']:.3f} | {m['bias_deg']:+.3f} "
                f"| {m['p95_deg']:.3f} | {100 * m['within_0.2deg']:.1f}% |")

    hdr = "| | n | MAE (deg) | RMSE | bias | P95 | within 0.2 deg |\n|---|---|---|---|---|---|---|"
    wl = r["window_level"]
    lines = [f"# Evaluation report ({r['split']} split)", "",
             f"Checkpoint: `{r['checkpoint']}` - {r['windows']} windows, "
             f"{100 * r['excited_fraction']:.1f}% with sufficient longitudinal excitation.", "",
             "## Window-level load pitch accuracy", "", hdr,
             row("model, all windows", wl["model_all"]),
             row("model, excited windows", wl["model_excited"]),
             row("physics baseline, excited", wl["physics_excited"]), "",
             f"Auxiliary road-grade MAE: {wl['grade_mae_deg']:.3f} deg", ""]
    for g, members in r["breakdown"].items():
        lines += [f"### By {g} (excited windows)", "", hdr]
        for name, d in members.items():
            lines += [row(f"{name} - model", d["model"]), row(f"{name} - physics", d["physics"])]
        lines.append("")
    lines += ["## Closed-loop headlamp aim error (estimator + leveling controller)", "",
              "| estimator | window | MAE (deg) | P95 (deg) | within 0.2 deg | within 0.25 % | within 0.5 % |",
              "|---|---|---|---|---|---|---|"]
    for name, res in r["closed_loop"].items():
        for span, a in res.items():
            lines.append(f"| {name} | {span} | {a['mae_deg']:.3f} | {a['p95_deg']:.3f} | "
                         f"{100 * a['within_0.2deg']:.1f}% | {100 * a['within_0.25percent']:.1f}% | "
                         f"{100 * a['within_0.5percent']:.1f}% |")
    return "\n".join(lines) + "\n"

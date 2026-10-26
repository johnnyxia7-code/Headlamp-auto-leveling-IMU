"""Run the fusion EKF over simulated trips and evaluate it against ground truth."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .. import CHANNELS
from ..config import Config
from ..control.leveling import LevelingController
from ..sim.scenario import simulate_trip
from ..training.metrics import DEG_PER_PERCENT
from .ekf import PitchEKF

AX, AY, AZ, GY, GZ, VW = (CHANNELS.index(c) for c in ("ax", "ay", "az", "gy", "gz", "v_wheel"))

# (time index at fusion rate) -> (load pitch estimate deg, 1-sigma deg)
LoadPitchMeasurements = Dict[int, tuple]


def _block_mean(x: np.ndarray, q: int) -> np.ndarray:
    n = (x.shape[0] // q) * q
    return x[:n].reshape(n // q, q, *x.shape[1:]).mean(axis=1)


def _block_last_finite(x: np.ndarray, q: int) -> np.ndarray:
    n = (x.shape[0] // q) * q
    blocks = x[:n].reshape(n // q, q)
    out = np.full(blocks.shape[0], np.nan)
    for j in range(q):  # later samples in the block overwrite earlier ones
        col = blocks[:, j]
        ok = np.isfinite(col)
        out[ok] = col[ok]
    return out


def run_ekf(raw: Dict[str, np.ndarray], cfg: Config, use_camera: Optional[bool] = None,
            load_pitch_meas: Optional[LoadPitchMeasurements] = None) -> Dict[str, np.ndarray]:
    """Filter one trip given its full-rate signals (``simulate_trip(..., raw=True)["raw"]``).

    Returns time series at the fusion rate: estimates, 1-sigma, truth and flags.
    """
    fcfg = copy.deepcopy(cfg.fusion)
    if use_camera is not None:
        fcfg.use_camera = use_camera
    q = max(1, int(round(raw["fs_hz"] / fcfg.fs_hz)))
    fs = raw["fs_hz"] / q
    fcfg.fs_hz = fs
    dt = 1.0 / fs
    x = _block_mean(raw["inputs"].astype(np.float64), q)
    cam = _block_last_finite(raw["camera_pitch_deg"].astype(np.float64), q)
    idx = np.arange(q - 1, x.shape[0] * q, q)            # truth sampled at block ends

    ekf = PitchEKF(fcfg)
    n = x.shape[0]
    keys = ("load_pitch_deg", "load_pitch_std_deg", "road_grade_deg", "road_grade_std_deg", "body_pitch_deg")
    out = {k: np.empty(n) for k in keys}
    out["standstill"] = np.zeros(n, bool)
    out["fault"] = np.zeros(n, bool)
    for k in range(n):
        o = ekf.step(dt, x[k, AX], x[k, AZ], x[k, GY], x[k, VW], cam[k], x[k, AY], x[k, GZ])
        if load_pitch_meas and k in load_pitch_meas:
            z, sd = load_pitch_meas[k]
            ekf.update_load_pitch(z, sd)
            o = ekf.output()
        for key in keys:
            out[key][k] = getattr(o, key)
        out["standstill"][k] = o.standstill
        out["fault"][k] = any(o.faults.values())
    out["fs_hz"] = fs
    out["true_load_pitch_deg"] = raw["load_pitch_deg"][idx].astype(np.float64)
    out["true_road_grade_deg"] = raw["road_grade_deg"][idx].astype(np.float64)
    out["camera_frames"] = np.isfinite(cam)
    out["final_state"] = ekf.x.copy()
    return out


def closed_loop_from_series(res: Dict[str, np.ndarray], cfg: Config, warmup_s: float) -> np.ndarray:
    """Feed the EKF output to the leveling controller; return |aim error| after warm-up (deg)."""
    fs = res["fs_hz"]
    period = max(1, int(round(cfg.control.update_period_s * fs)))
    ctrl = LevelingController(cfg.control)
    errs = []
    for k in range(period - 1, len(res["load_pitch_deg"]), period):
        out = ctrl.update(period / fs, res["load_pitch_deg"][k], estimate_std_deg=res["load_pitch_std_deg"][k],
                          fault=bool(res["fault"][k]))
        if k / fs >= warmup_s:
            errs.append(abs(res["true_load_pitch_deg"][k] - out.correction_deg))
    return np.asarray(errs)


def _stats(err: np.ndarray) -> Dict[str, float]:
    a = np.abs(err)
    return {"mae_deg": float(a.mean()), "p95_deg": float(np.percentile(a, 95)),
            "within_0.1deg": float(np.mean(a <= 0.1)), "within_0.2deg": float(np.mean(a <= 0.2)),
            "within_0.25percent": float(np.mean(a <= 0.25 * DEG_PER_PERCENT)),
            "within_0.5percent": float(np.mean(a <= 0.5 * DEG_PER_PERCENT))}


def nn_load_pitch_measurements(trip: Dict, ckpt: Dict, fusion_fs: float, period_s: float,
                               std_deg: float) -> LoadPitchMeasurements:
    """Run a trained window model on the decimated inputs and map its outputs to fusion steps."""
    from ..data.features import build_features
    from ..training.evaluate import model_estimator
    fs = ckpt["fs_hz"]
    W = ckpt["window"]
    feats = build_features(trip["inputs"], fs)
    step = max(1, int(round(period_s * fs)))
    ends = np.arange(W - 1, len(feats), step)
    if ends.size == 0:
        return {}
    preds = model_estimator(ckpt)(np.stack([feats[e - W + 1:e + 1] for e in ends]))
    return {int(round((e + 1) / fs * fusion_fs)) - 1: (float(p), std_deg) for e, p in zip(ends, preds)}


def evaluate_fusion(cfg: Config, n_trips: int = 20, out_dir: Optional[str] = None,
                    checkpoint: Optional[str] = None, nn_std_deg: float = 0.35,
                    verbose: bool = True) -> Dict:
    """Compare fusion variants on the first ``n_trips`` trips of the test split (same seeds)."""
    root = np.random.SeedSequence(cfg.dataset.seed)
    test_seeds = root.spawn(3)[2].spawn(max(n_trips, cfg.dataset.n_test_trips))[:n_trips]
    warmup = cfg.dataset.window_s
    ckpt = None
    if checkpoint:
        from ..training.train import load_checkpoint
        ckpt = load_checkpoint(checkpoint)

    variants: Dict[str, Callable] = {
        "ekf_imu_wheel": lambda raw, trip: run_ekf(raw, cfg, use_camera=False),
        "ekf_imu_wheel_camera": lambda raw, trip: run_ekf(raw, cfg, use_camera=True),
    }
    if ckpt is not None:
        def with_nn(raw, trip, cam):
            meas = nn_load_pitch_measurements(trip, ckpt, cfg.fusion.fs_hz, warmup / 2.0, nn_std_deg)
            return run_ekf(raw, cfg, use_camera=cam, load_pitch_meas=meas)
        variants["ekf_imu_wheel_nn"] = lambda raw, trip: with_nn(raw, trip, False)
        variants["ekf_imu_wheel_camera_nn"] = lambda raw, trip: with_nn(raw, trip, True)

    est_err: Dict[str, List[np.ndarray]] = {k: [] for k in variants}
    grade_err: Dict[str, List[np.ndarray]] = {k: [] for k in variants}
    cl_err: Dict[str, List[np.ndarray]] = {k: [] for k in variants}
    z_scores: Dict[str, List[np.ndarray]] = {k: [] for k in variants}
    conv_time: Dict[str, List[float]] = {k: [] for k in variants}
    cam_frames = 0.0
    for i, seed in enumerate(test_seeds):
        trip = simulate_trip(cfg, seed, raw=True)
        raw = trip["raw"]
        for name, fn in variants.items():
            res = fn(raw, trip)
            fs = res["fs_hz"]
            err = res["load_pitch_deg"] - res["true_load_pitch_deg"]
            after = np.arange(err.size) / fs >= warmup
            est_err[name].append(err[after])
            grade_err[name].append((res["road_grade_deg"] - res["true_road_grade_deg"])[after])
            z_scores[name].append(err[after] / np.maximum(res["load_pitch_std_deg"][after], 1e-6))
            cl_err[name].append(closed_loop_from_series(res, cfg, warmup))
            # Convergence from a wrong initial value: only trips whose load differs from the
            # filter's initial value by >= 0.3 deg; first time the error stays within 0.2 deg for 30 s.
            if abs(res["true_load_pitch_deg"][0] - cfg.fusion.init_load_pitch_deg) >= 0.3:
                ok = np.abs(err) <= 0.2
                hold = int(30 * fs)
                run = np.convolve(ok.astype(float), np.ones(hold), mode="valid") >= hold
                conv_time[name].append(float(np.argmax(run) / fs) if run.any() else float("inf"))
            if name == "ekf_imu_wheel_camera":
                cam_frames += float(res["camera_frames"].mean())
        if verbose:
            print(f"[fusion] trip {i + 1}/{n_trips} ({trip['meta']['load_case']}, {trip['meta']['grade_class']})",
                  flush=True)

    report: Dict = {"trips": n_trips, "warmup_s": warmup,
                    "camera_frame_fraction": cam_frames / max(n_trips, 1), "variants": {}}
    for name in variants:
        e = np.concatenate(est_err[name])
        z = np.concatenate(z_scores[name])
        report["variants"][name] = {
            "load_pitch": _stats(e),
            "load_pitch_bias_deg": float(e.mean()),
            "consistency_within_2sigma": float(np.mean(np.abs(z) <= 2.0)),
            "road_grade_mae_deg": float(np.mean(np.abs(np.concatenate(grade_err[name])))),
            "convergence_trips": len(conv_time[name]),
            "convergence_s_median": float(np.median(conv_time[name])) if conv_time[name] else float("nan"),
            "convergence_s_p90": float(np.percentile(conv_time[name], 90, method="higher")) if conv_time[name] else float("nan"),
            "closed_loop": _stats(np.concatenate(cl_err[name])),
        }

    out = Path(out_dir or "runs/fusion")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "fusion_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    (out / "fusion_report.md").write_text(render_fusion_markdown(report), encoding="utf-8")
    if verbose:
        print(render_fusion_markdown(report))
    return report


def render_fusion_markdown(r: Dict) -> str:
    lines = [f"# Sensor-fusion EKF evaluation ({r['trips']} test trips, after {r['warmup_s']:.0f} s)", "",
             f"Camera frames available {100 * r['camera_frame_fraction']:.1f} % of filter steps.", "",
             "| variant | load pitch MAE | P95 | within 0.2 deg | bias | within 2 sigma | grade MAE "
             "| convergence median / P90 | closed-loop MAE | closed-loop P95 | within 0.25 % |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in r["variants"].items():
        lp, cl = v["load_pitch"], v["closed_loop"]
        lines.append(f"| {name} | {lp['mae_deg']:.3f} | {lp['p95_deg']:.3f} | {100 * lp['within_0.2deg']:.1f} % "
                     f"| {v['load_pitch_bias_deg']:+.3f} | {100 * v['consistency_within_2sigma']:.1f} % "
                     f"| {v['road_grade_mae_deg']:.3f} | {v['convergence_s_median']:.0f} / {v['convergence_s_p90']:.0f} s "
                     f"(n={v['convergence_trips']}) "
                     f"| {cl['mae_deg']:.3f} | {cl['p95_deg']:.3f} | {100 * cl['within_0.25percent']:.1f} % |")
    return "\n".join(lines) + "\n"

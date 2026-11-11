"""Command line interface: ``hal <command>`` (or ``python -m headlamp_leveling``)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .config import load_config


def _cmd_generate(a) -> None:
    from .data.generate import generate_dataset
    cfg = load_config(a.config)
    generate_dataset(cfg, a.out)


def _cmd_train(a) -> None:
    from .training.train import train
    cfg = load_config(a.config)
    if a.arch:
        cfg.model.arch = a.arch
    if a.epochs:
        cfg.train.epochs = a.epochs
    train(cfg, a.data, a.out)


def _cmd_evaluate(a) -> None:
    from .training.evaluate import evaluate
    evaluate(a.checkpoint, a.data, a.out, a.split)


def _cmd_export(a) -> None:
    from .export import export
    export(a.checkpoint, a.out, onnx=not a.no_onnx)


def _cmd_import_logs(a) -> None:
    from .data.real_logs import import_logs
    cfg = load_config(a.config)
    kwargs = {}
    if a.wheelbase:
        kwargs = {"wheelbase_m": a.wheelbase, "h_ref_front_mm": a.h_ref_front, "h_ref_rear_mm": a.h_ref_rear}
    ts = import_logs(a.csv, cfg.model_fs_hz, **kwargs)
    ts.save(a.out)
    print(f"[import] {len(ts)} logs, {ts.inputs.shape[0] / ts.fs_hz / 3600:.2f} h -> {a.out}")


def _cmd_fuse_eval(a) -> None:
    from .fusion.run import evaluate_fusion
    cfg = load_config(a.config)
    evaluate_fusion(cfg, a.trips, a.out, a.checkpoint, a.nn_std)


def _cmd_robust_eval(a) -> None:
    from .fusion.robustness import evaluate_robustness
    cfg = load_config(a.config)
    evaluate_robustness(cfg, a.trips, a.out, a.scenarios or None)


def _cmd_verify(a) -> None:
    from .verify import SCENARIOS, verify
    if a.list:
        for name, make in SCENARIOS.items():
            sc = make()
            print(f"{name:26s} [{sc.group}] {sc.title}")
        return
    report = verify(load_config(a.config), a.scenario or None, a.out, plots=not a.no_plots, seed=a.seed,
                    n_seeds=a.seeds)
    if a.strict and report["passed"] < report["total"]:
        sys.exit(1)


def _cmd_simulate(a) -> None:
    import numpy as np
    from .sim.scenario import simulate_trip
    cfg = load_config(a.config)
    trip = simulate_trip(cfg, a.seed)
    print(json.dumps(trip["meta"], indent=2))
    if a.out:
        np.savez_compressed(a.out, **{k: v for k, v in trip.items() if k != "meta"})
        print(f"[simulate] trip arrays -> {a.out}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="hal", description="IMU-based automatic headlamp leveling toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="simulate train/val/test trips")
    g.add_argument("-c", "--config", default=None)
    g.add_argument("-o", "--out", default=None, help="output directory (default: dataset.out_dir)")
    g.set_defaults(func=_cmd_generate)

    t = sub.add_parser("train", help="train a pitch estimator")
    t.add_argument("-c", "--config", default=None)
    t.add_argument("-d", "--data", default=None, help="dataset directory (default: dataset.out_dir)")
    t.add_argument("-o", "--out", default=None, help="run directory (default: train.out_dir)")
    t.add_argument("--arch", choices=["tcn", "gru", "hybrid"], default=None)
    t.add_argument("--epochs", type=int, default=None)
    t.set_defaults(func=_cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a checkpoint (window metrics + closed-loop leveling)")
    e.add_argument("checkpoint")
    e.add_argument("-d", "--data", default=None)
    e.add_argument("-o", "--out", default=None)
    e.add_argument("--split", default="test")
    e.set_defaults(func=_cmd_evaluate)

    x = sub.add_parser("export", help="export TorchScript / ONNX + metadata")
    x.add_argument("checkpoint")
    x.add_argument("-o", "--out", default=None)
    x.add_argument("--no-onnx", action="store_true")
    x.set_defaults(func=_cmd_export)

    i = sub.add_parser("import-logs", help="convert recorded CSV logs into a dataset split")
    i.add_argument("csv", nargs="+")
    i.add_argument("-o", "--out", required=True, help="output .npz, e.g. data/real/test.npz")
    i.add_argument("-c", "--config", default=None)
    i.add_argument("--wheelbase", type=float, default=None, help="m, needed for ride-height labels")
    i.add_argument("--h-ref-front", type=float, default=None, help="reference front ride height, mm")
    i.add_argument("--h-ref-rear", type=float, default=None, help="reference rear ride height, mm")
    i.set_defaults(func=_cmd_import_logs)

    f = sub.add_parser("fuse-eval", help="evaluate the IMU + wheel + camera EKF on simulated test trips")
    f.add_argument("-c", "--config", default=None)
    f.add_argument("-n", "--trips", type=int, default=20)
    f.add_argument("-o", "--out", default="runs/fusion")
    f.add_argument("--checkpoint", default=None, help="also fuse this window model as a load-pitch measurement")
    f.add_argument("--nn-std", type=float, default=0.35, help="1-sigma assigned to the model output, deg")
    f.set_defaults(func=_cmd_fuse_eval)

    r = sub.add_parser("robust-eval", help="inject sensor faults; compare the EKF with / without the input guard")
    r.add_argument("-c", "--config", default=None)
    r.add_argument("-n", "--trips", type=int, default=12)
    r.add_argument("-o", "--out", default="runs/robustness")
    r.add_argument("--scenarios", nargs="*", default=None, help="subset of fault scenarios (default: all)")
    r.set_defaults(func=_cmd_robust_eval)

    v = sub.add_parser("verify", help="scripted scenarios on the half-car plant with pass / fail criteria")
    v.add_argument("-c", "--config", default=None)
    v.add_argument("-o", "--out", default="runs/verify")
    v.add_argument("--scenario", nargs="*", default=None, help="scenario names (default: all)")
    v.add_argument("--list", action="store_true", help="list the scenarios and exit")
    v.add_argument("--no-plots", action="store_true", help="skip the PNG time-series plots (no matplotlib needed)")
    v.add_argument("--seed", type=int, default=0, help="first seed (sensor errors, camera bias, road profile)")
    v.add_argument("--seeds", type=int, default=1, help="number of seeds to run each scenario with")
    v.add_argument("--strict", action="store_true", help="exit with status 1 if any scenario fails")
    v.set_defaults(func=_cmd_verify)

    s = sub.add_parser("simulate", help="simulate one trip and print its scenario")
    s.add_argument("-c", "--config", default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("-o", "--out", default=None)
    s.set_defaults(func=_cmd_simulate)

    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)   # progress shows up live when piped to a log
    a.func(a)


if __name__ == "__main__":
    main(sys.argv[1:])

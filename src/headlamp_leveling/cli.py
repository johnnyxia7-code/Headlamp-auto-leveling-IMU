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

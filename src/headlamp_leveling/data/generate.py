"""Generate simulated train / val / test splits in parallel."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..config import Config
from ..sim.scenario import simulate_trip
from .tripset import TripSet


def _worker(args):
    cfg, seed = args
    return simulate_trip(cfg, seed)


def simulate_split(cfg: Config, n_trips: int, seed_seq: np.random.SeedSequence, workers: int = 0) -> TripSet:
    seeds = seed_seq.spawn(n_trips)
    jobs = [(cfg, s) for s in seeds]
    workers = workers or os.cpu_count() or 1
    if workers > 1 and n_trips > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            trips = pool.map(_worker, jobs, chunksize=max(1, n_trips // (workers * 4)))
    else:
        trips = [_worker(j) for j in jobs]
    return TripSet.from_trips(trips, cfg.model_fs_hz)


def generate_dataset(cfg: Config, out_dir: str | None = None, verbose: bool = True) -> Dict[str, Path]:
    out = Path(out_dir or cfg.dataset.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = np.random.SeedSequence(cfg.dataset.seed)
    split_seeds = dict(zip(("train", "val", "test"), root.spawn(3)))
    counts = {"train": cfg.dataset.n_train_trips, "val": cfg.dataset.n_val_trips, "test": cfg.dataset.n_test_trips}
    paths, summary = {}, {}
    for split, n in counts.items():
        t0 = time.time()
        ts = simulate_split(cfg, n, split_seeds[split], cfg.dataset.workers)
        paths[split] = out / f"{split}.npz"
        ts.save(paths[split])
        lp = ts.labels["load_pitch_deg"]
        summary[split] = {
            "trips": len(ts),
            "hours": float(ts.inputs.shape[0] / ts.fs_hz / 3600.0),
            "load_pitch_deg": {"min": float(lp.min()), "max": float(lp.max()), "mean": float(lp.mean())},
            "load_cases": dict(Counter(m["load_case"] for m in ts.meta)),
            "grade_classes": dict(Counter(m["grade_class"] for m in ts.meta)),
        }
        if verbose:
            print(f"[generate] {split}: {n} trips, {summary[split]['hours']:.2f} h in {time.time() - t0:.1f} s"
                  f" -> {paths[split]}")
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump({"config": cfg.to_dict(), "splits": summary}, fh, indent=2)
    return paths


def load_splits(data_dir: str | Path, splits: List[str] = ("train", "val", "test")) -> Dict[str, TripSet]:
    return {s: TripSet.load(Path(data_dir) / f"{s}.npz") for s in splits if (Path(data_dir) / f"{s}.npz").exists()}

"""Training loop for the neural pitch estimators."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config, config_from_dict
from ..data.dataset import WindowDataset
from ..data.features import FEATURES, Normalizer
from ..data.generate import load_splits
from ..models.nets import build_model, count_params
from .metrics import format_metrics, regression_metrics


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def window_sizes(cfg: Config, fs_hz: float) -> Tuple[int, int]:
    return int(round(cfg.dataset.window_s * fs_hz)), max(1, int(round(cfg.dataset.stride_s * fs_hz)))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, ys, excs = [], [], []
    for x, y, exc in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
        excs.append(exc.numpy())
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(excs)


def loss_fn(pred: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:
    d = cfg.train.huber_delta_deg
    pitch = nn.functional.huber_loss(pred[:, 0], y[:, 0], delta=d)
    grade = nn.functional.huber_loss(pred[:, 1] / 5.0, y[:, 1] / 5.0, delta=d)
    return pitch + cfg.train.aux_grade_weight * grade


def train(cfg: Config, data_dir: Optional[str] = None, out_dir: Optional[str] = None,
          verbose: bool = True) -> Path:
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    out = Path(out_dir or cfg.train.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.train.device)

    splits = load_splits(data_dir or cfg.dataset.out_dir, ["train", "val"])
    if "train" not in splits or "val" not in splits:
        raise FileNotFoundError(f"train/val splits not found in {data_dir or cfg.dataset.out_dir}; run `hal generate`")
    fs = splits["train"].fs_hz
    W, S = window_sizes(cfg, fs)
    train_ds = WindowDataset(splits["train"], W, S, augment=cfg.train.augment, seed=cfg.train.seed)
    val_ds = WindowDataset(splits["val"], W, S, normalizer=train_ds.normalizer)
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
                          num_workers=cfg.train.num_workers)
    val_dl = DataLoader(val_ds, batch_size=512, num_workers=cfg.train.num_workers)

    model = build_model(cfg.model.arch, len(FEATURES), cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps = cfg.train.epochs * max(1, len(train_dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.train.lr, total_steps=steps, pct_start=0.15)
    if verbose:
        print(f"[train] arch={cfg.model.arch} params={count_params(model):,} device={device} "
              f"windows train={len(train_ds)} val={len(val_ds)} W={W} ({cfg.dataset.window_s:.0f} s @ {fs:g} Hz)")

    ckpt_path = out / "model.pt"
    history = []
    best, bad_epochs = float("inf"), 0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        t0, run_loss, nb = time.time(), 0.0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item()
            nb += 1
        pred, y, exc = predict(model, val_dl, device)
        m_all = regression_metrics(pred[:, 0], y[:, 0])
        m_exc = regression_metrics(pred[:, 0], y[:, 0], exc >= cfg.control.min_excitation_ms2)
        row = {"epoch": epoch, "train_loss": run_loss / max(nb, 1), "val_mae_deg": m_all["mae_deg"],
               "val_p95_deg": m_all["p95_deg"], "val_mae_excited_deg": m_exc.get("mae_deg", float("nan")),
               "grade_mae_deg": float(np.mean(np.abs(pred[:, 1] - y[:, 1]))),
               "lr": sched.get_last_lr()[0], "time_s": time.time() - t0}
        history.append(row)
        if verbose:
            print(f"[train] epoch {epoch:3d} loss={row['train_loss']:.4f} val {format_metrics(m_all)} "
                  f"| excited MAE={row['val_mae_excited_deg']:.3f} ({row['time_s']:.0f}s)")
        if m_all["mae_deg"] < best - 1e-4:
            best, bad_epochs = m_all["mae_deg"], 0
            torch.save({
                "model_state": model.state_dict(),
                "arch": cfg.model.arch,
                "in_ch": len(FEATURES),
                "features": list(FEATURES),
                "config": cfg.to_dict(),
                "normalizer": train_ds.normalizer.to_dict(),
                "window": W,
                "fs_hz": fs,
                "epoch": epoch,
                "val_metrics": {"all": m_all, "excited": m_exc},
            }, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.patience:
                if verbose:
                    print(f"[train] early stop at epoch {epoch} (best val MAE {best:.3f} deg)")
                break

    with open(out / "history.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    with open(out / "config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)
    if verbose:
        print(f"[train] best val MAE {best:.3f} deg -> {ckpt_path}")
    return ckpt_path


def load_checkpoint(path: str | Path, device: str = "cpu") -> Dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(ckpt["config"])
    model = build_model(ckpt["arch"], ckpt["in_ch"], cfg.model)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    ckpt["model"] = model
    ckpt["cfg"] = cfg
    ckpt["normalizer_obj"] = Normalizer.from_dict(ckpt["normalizer"])
    return ckpt

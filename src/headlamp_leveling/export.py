"""Export a trained checkpoint for deployment (TorchScript, ONNX, metadata JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch

from .models.nets import count_params
from .training.train import load_checkpoint


def export(ckpt_path: str, out_dir: Optional[str] = None, onnx: bool = True, verbose: bool = True) -> Dict[str, str]:
    ckpt = load_checkpoint(ckpt_path)
    model = ckpt["model"].eval()
    out = Path(out_dir or Path(ckpt_path).parent / "export")
    out.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, ckpt["in_ch"], ckpt["window"])
    paths: Dict[str, str] = {}

    traced = torch.jit.trace(model, example)
    paths["torchscript"] = str(out / "pitch_estimator.ts")
    traced.save(paths["torchscript"])

    if onnx:
        try:
            paths["onnx"] = str(out / "pitch_estimator.onnx")
            torch.onnx.export(model, example, paths["onnx"], input_names=["features"],
                              output_names=["load_pitch_and_grade_deg"], opset_version=17,
                              dynamic_axes={"features": {0: "batch"}}, dynamo=False)
        except Exception as exc:  # onnx is an optional dependency
            paths.pop("onnx", None)
            if verbose:
                print(f"[export] ONNX export skipped: {exc}")

    meta = {
        "arch": ckpt["arch"],
        "parameters": count_params(model),
        "input": {"shape": [1, ckpt["in_ch"], ckpt["window"]], "layout": "NCW",
                  "features": ckpt["features"], "fs_hz": ckpt["fs_hz"],
                  "normalization": "x_norm = (x - mean) / std per feature", **ckpt["normalizer"]},
        "output": {"0": "static load pitch, deg, nose-up positive",
                   "1": "road grade, deg, uphill positive (auxiliary)"},
        "control": ckpt["config"]["control"],
        "val_metrics": ckpt["val_metrics"],
    }
    paths["metadata"] = str(out / "model_meta.json")
    with open(paths["metadata"], "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    if verbose:
        for k, v in paths.items():
            print(f"[export] {k}: {v}")
    return paths

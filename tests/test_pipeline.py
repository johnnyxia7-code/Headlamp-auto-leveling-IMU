import csv

import numpy as np
import pytest
import torch

from headlamp_leveling import CHANNELS
from headlamp_leveling.config import Config, load_config
from headlamp_leveling.data.dataset import WindowDataset
from headlamp_leveling.data.features import FEATURES
from headlamp_leveling.data.generate import generate_dataset, simulate_split
from headlamp_leveling.data.real_logs import load_log, ride_height_to_pitch_deg
from headlamp_leveling.data.tripset import TripSet
from headlamp_leveling.models import build_model


@pytest.fixture(scope="module")
def small_cfg(tmp_path_factory):
    cfg = Config()
    cfg.sim.trip_duration_s = 90.0
    cfg.dataset.out_dir = str(tmp_path_factory.mktemp("data"))
    cfg.dataset.n_train_trips, cfg.dataset.n_val_trips, cfg.dataset.n_test_trips = 6, 2, 2
    cfg.dataset.window_s, cfg.dataset.stride_s = 20.0, 5.0
    cfg.dataset.workers = 1
    cfg.train.out_dir = str(tmp_path_factory.mktemp("run"))
    cfg.train.epochs, cfg.train.batch_size = 1, 16
    cfg.model.channels = [8, 8]
    return cfg


def test_config_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("dataset:\n  window_s: 12\nvehicle:\n  wheelbase_m: [2.5, 2.6]\n")
    cfg = load_config(p)
    assert cfg.dataset.window_s == 12 and cfg.vehicle.wheelbase_m == (2.5, 2.6)
    p.write_text("dataset:\n  windw_s: 12\n")
    with pytest.raises(KeyError):
        load_config(p)


def test_tripset_save_load(tmp_path, small_cfg):
    ts = simulate_split(small_cfg, 2, np.random.SeedSequence(0), workers=1)
    ts.save(tmp_path / "x.npz")
    back = TripSet.load(tmp_path / "x.npz")
    assert len(back) == 2 and back.fs_hz == ts.fs_hz
    np.testing.assert_array_equal(back.inputs, ts.inputs)
    assert back.meta[0]["load_case"] == ts.meta[0]["load_case"]


def test_window_dataset(small_cfg):
    ts = simulate_split(small_cfg, 2, np.random.SeedSequence(1), workers=1)
    W = int(small_cfg.dataset.window_s * ts.fs_hz)
    ds = WindowDataset(ts, W, 50)
    x, y, exc = ds[0]
    assert x.shape == (len(FEATURES), W) and y.shape == (2,)
    assert torch.isfinite(x).all() and exc >= 0


@pytest.mark.parametrize("arch", ["tcn", "gru", "hybrid"])
def test_models_forward(arch):
    cfg = Config()
    model = build_model(arch, len(FEATURES), cfg.model).eval()
    out = model(torch.randn(3, len(FEATURES), 600))
    assert out.shape == (3, 2)


def test_end_to_end_train_evaluate_export(small_cfg, tmp_path):
    from headlamp_leveling.export import export
    from headlamp_leveling.training.evaluate import evaluate
    from headlamp_leveling.training.train import train

    generate_dataset(small_cfg, verbose=False)
    ckpt = train(small_cfg, verbose=False)
    report = evaluate(str(ckpt), verbose=False)
    assert report["window_level"]["model_all"]["n"] > 0
    assert "model" in report["closed_loop"]
    paths = export(str(ckpt), str(tmp_path / "exp"), onnx=False, verbose=False)
    ts_model = torch.jit.load(paths["torchscript"])
    assert ts_model(torch.zeros(1, len(FEATURES), int(small_cfg.dataset.window_s * 20))).shape == (1, 2)


def test_ride_height_label_sign():
    # Rear lower than reference -> nose up (positive).
    assert ride_height_to_pitch_deg(np.array([400.0]), np.array([380.0]), 2.7, 400.0, 400.0)[0] > 0


def test_load_csv_log(tmp_path):
    p = tmp_path / "log.csv"
    t = np.arange(0, 60, 0.01)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("t",) + CHANNELS + ("h_front_mm", "h_rear_mm"))
        for ti in t:
            w.writerow([ti, 0.1, 0.0, 9.8, 0.0, 0.0, 0.0, 10.0, 400.0, 390.0])
    trip = load_log(p, 20.0, wheelbase_m=2.7, h_ref_front_mm=400.0, h_ref_rear_mm=400.0)
    assert trip["inputs"].shape[1] == len(CHANNELS)
    assert trip["inputs"].shape[0] == pytest.approx(60 * 20, abs=2)
    assert np.allclose(trip["load_pitch_deg"], np.rad2deg(np.arctan2(0.01, 2.7)), atol=1e-6)

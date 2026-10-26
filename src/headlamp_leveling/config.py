"""Typed project configuration loaded from YAML.

Every section has defaults, so a YAML file only needs to list the values it
overrides. Unknown keys raise an error to catch typos early.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class SimConfig:
    fs_hz: float = 100.0                 # internal simulation / IMU sample rate
    decimation: int = 5                  # model input rate = fs_hz / decimation
    trip_duration_s: float = 240.0
    max_speed_kmh: float = 130.0
    # Road grade classes sampled per trip: (name, max |grade| in percent, probability)
    grade_classes: List[Tuple[str, float, float]] = field(default_factory=lambda: [
        ("flat", 1.0, 0.25),
        ("rolling", 5.0, 0.40),
        ("hilly", 10.0, 0.30),
        ("mountain", 14.0, 0.05),
    ])
    # ISO 8608 roughness classes: (name, Gd(n0) in m^3, probability)
    roughness_classes: List[Tuple[str, float, float]] = field(default_factory=lambda: [
        ("A", 16e-6, 0.45),
        ("B", 64e-6, 0.40),
        ("C", 256e-6, 0.15),
    ])
    load_change_prob: float = 0.3        # chance that the load changes at one stop during a trip


@dataclass
class VehicleConfig:
    wheelbase_m: Tuple[float, float] = (2.45, 3.10)
    front_axle_rate_kn_m: Tuple[float, float] = (38.0, 60.0)   # both wheels combined, at the wheel
    rear_axle_rate_kn_m: Tuple[float, float] = (34.0, 62.0)
    pitch_gradient_deg_per_g: Tuple[float, float] = (1.2, 4.0)  # quasi-static pitch per g of long. accel
    pitch_freq_hz: Tuple[float, float] = (1.1, 1.8)
    pitch_damping: Tuple[float, float] = (0.25, 0.55)
    heave_freq_hz: Tuple[float, float] = (1.0, 1.6)
    roll_gradient_deg_per_g: Tuple[float, float] = (3.0, 7.0)
    passenger_mass_kg: Tuple[float, float] = (45.0, 110.0)
    trunk_mass_kg: Tuple[float, float] = (10.0, 180.0)
    tongue_load_kg: Tuple[float, float] = (25.0, 100.0)


@dataclass
class ImuConfig:
    accel_noise_ug_rthz: float = 150.0     # accelerometer noise density
    accel_bias_mg: float = 15.0            # per-trip turn-on bias (1 sigma, per axis)
    accel_bias_walk_mg_s: float = 0.02     # bias random walk (1 sigma per sqrt(s))
    accel_scale_err: float = 0.005         # scale factor error (1 sigma)
    gyro_noise_dps_rthz: float = 0.008
    gyro_bias_dps: float = 0.3
    misalign_deg: float = 0.05             # residual mounting misalignment after EOL calibration (1 sigma)
    vibration_ms2: float = 0.15            # broadband body vibration at 100 km/h (scales with speed)
    accel_lsb_mg: float = 0.061            # +-2 g, 16-bit
    gyro_lsb_dps: float = 0.0076           # +-250 dps, 16-bit
    wheel_speed_noise_ms: float = 0.02
    tyre_radius_err: float = 0.015         # per-trip rolling radius error (1 sigma, relative)
    slip_per_ms2: float = 0.002            # longitudinal tyre slip per m/s^2


@dataclass
class DatasetConfig:
    out_dir: str = "data/sim"
    n_train_trips: int = 600
    n_val_trips: int = 100
    n_test_trips: int = 100
    window_s: float = 30.0
    stride_s: float = 3.0
    seed: int = 7
    workers: int = 0                      # 0 = use all CPUs


@dataclass
class ModelConfig:
    arch: str = "tcn"                      # tcn | gru | hybrid
    channels: List[int] = field(default_factory=lambda: [32, 48, 64, 64])
    kernel_size: int = 5
    dropout: float = 0.1
    gru_hidden: int = 64
    gru_layers: int = 1


@dataclass
class TrainConfig:
    out_dir: str = "runs/default"
    epochs: int = 40
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    huber_delta_deg: float = 0.2
    aux_grade_weight: float = 0.1
    patience: int = 10
    device: str = "auto"
    seed: int = 0
    num_workers: int = 0
    augment: bool = True


@dataclass
class ControlConfig:
    update_period_s: float = 1.0           # how often a new window estimate is produced
    min_excitation_ms2: float = 0.25       # std of longitudinal accel needed to accept an estimate
    filter_tau_s: float = 20.0             # time constant of the confidence-gated estimate filter
    deadband_deg: float = 0.05
    max_rate_deg_s: float = 0.5            # actuator slew limit
    range_deg: Tuple[float, float] = (-1.0, 3.0)   # correction range (nose-up pitch the lamp can compensate)
    failsafe_deg: float = 0.0              # safe correction used on sensor fault
    steps_per_deg: float = 40.0            # stepper actuator resolution


@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    control: ControlConfig = field(default_factory=ControlConfig)

    @property
    def model_fs_hz(self) -> float:
        return self.sim.fs_hz / self.sim.decimation

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls, values: Optional[Dict[str, Any]]):
    values = values or {}
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(names)
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, v in values.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, v)
        elif isinstance(v, list) and v and isinstance(v[0], list):
            kwargs[name] = [tuple(x) for x in v]
        elif isinstance(v, list) and typing.get_origin(hint) is tuple:
            kwargs[name] = tuple(v)
        else:
            kwargs[name] = v
    return cls(**kwargs)


def config_from_dict(d: Optional[Dict[str, Any]]) -> Config:
    return _build(Config, d)


def load_config(path: Optional[str | Path] = None) -> Config:
    if path is None:
        return Config()
    with open(path, "r", encoding="utf-8") as fh:
        return config_from_dict(yaml.safe_load(fh))

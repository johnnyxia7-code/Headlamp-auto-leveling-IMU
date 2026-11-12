"""Vehicle parameters, load cases and the static load pitch they produce.

Sign conventions (used throughout the project):
    * body frame: x forward, y left, z up
    * pitch angles are positive nose-up
    * the reference state is the unladen vehicle with driver, at which the
      headlamp aim is set at end of line; ``load_pitch`` is measured from there
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .. import G
from ..config import VehicleConfig


@dataclass
class VehicleParams:
    wheelbase_m: float
    front_axle_rate_n_m: float
    rear_axle_rate_n_m: float
    pitch_gradient_rad_per_ms2: float
    pitch_freq_hz: float
    pitch_damping: float
    heave_freq_hz: float
    roll_gradient_rad_per_ms2: float


@dataclass
class LoadItem:
    name: str
    mass_kg: float
    x_m: float          # longitudinal position behind the front axle


@dataclass
class LoadCase:
    name: str
    items: List[LoadItem] = field(default_factory=list)

    @property
    def total_mass_kg(self) -> float:
        return float(sum(i.mass_kg for i in self.items))


def _u(rng: np.random.Generator, bounds: Tuple[float, float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def sample_vehicle(rng: np.random.Generator, cfg: VehicleConfig) -> VehicleParams:
    return VehicleParams(
        wheelbase_m=_u(rng, cfg.wheelbase_m),
        front_axle_rate_n_m=_u(rng, cfg.front_axle_rate_kn_m) * 1e3,
        rear_axle_rate_n_m=_u(rng, cfg.rear_axle_rate_kn_m) * 1e3,
        pitch_gradient_rad_per_ms2=np.deg2rad(_u(rng, cfg.pitch_gradient_deg_per_g)) / G,
        pitch_freq_hz=_u(rng, cfg.pitch_freq_hz),
        pitch_damping=_u(rng, cfg.pitch_damping),
        heave_freq_hz=_u(rng, cfg.heave_freq_hz),
        roll_gradient_rad_per_ms2=np.deg2rad(_u(rng, cfg.roll_gradient_deg_per_g)) / G,
    )


# Load scenarios relative to "driver only". Probabilities favour the realistic
# everyday cases but keep the extremes (full + trunk, trailer) well represented.
LOAD_SCENARIOS = (
    ("driver_only", 0.15),
    ("front_passenger", 0.12),
    ("rear_passengers", 0.12),
    ("full_cabin", 0.12),
    ("trunk_only", 0.12),
    ("full_cabin_trunk", 0.15),
    ("trailer", 0.10),
    ("max_rear", 0.12),
)


def sample_load_case(rng: np.random.Generator, veh: VehicleParams, cfg: VehicleConfig,
                     scenario: str | None = None) -> LoadCase:
    L = veh.wheelbase_m
    x_front_row = 0.38 * L + rng.normal(0.0, 0.05)
    x_rear_row = 0.80 * L + rng.normal(0.0, 0.05)
    x_trunk = L + rng.uniform(0.35, 0.90)
    x_towball = L + rng.uniform(0.90, 1.25)

    if scenario is None:
        names, probs = zip(*LOAD_SCENARIOS)
        scenario = str(rng.choice(names, p=np.asarray(probs) / np.sum(probs)))

    def pax() -> float:
        return _u(rng, cfg.passenger_mass_kg)

    items: List[LoadItem] = []
    if scenario == "front_passenger":
        items.append(LoadItem("front_passenger", pax(), x_front_row))
    elif scenario == "rear_passengers":
        for k in range(int(rng.integers(1, 4))):
            items.append(LoadItem(f"rear_passenger_{k}", pax(), x_rear_row))
    elif scenario in ("full_cabin", "full_cabin_trunk", "max_rear"):
        items.append(LoadItem("front_passenger", pax(), x_front_row))
        for k in range(3):
            items.append(LoadItem(f"rear_passenger_{k}", pax(), x_rear_row))
        if scenario != "full_cabin":
            trunk = _u(rng, cfg.trunk_mass_kg)
            if scenario == "max_rear":
                trunk = cfg.trunk_mass_kg[1] * rng.uniform(0.8, 1.0)
            items.append(LoadItem("trunk", trunk, x_trunk))
        if scenario == "max_rear":
            items.append(LoadItem("tongue_load", cfg.tongue_load_kg[1] * rng.uniform(0.7, 1.0), x_towball))
    elif scenario == "trunk_only":
        items.append(LoadItem("trunk", _u(rng, cfg.trunk_mass_kg), x_trunk))
    elif scenario == "trailer":
        if rng.random() < 0.5:
            items.append(LoadItem("front_passenger", pax(), x_front_row))
        items.append(LoadItem("tongue_load", _u(rng, cfg.tongue_load_kg), x_towball))
    elif scenario != "driver_only":
        raise ValueError(f"unknown load scenario {scenario!r}")
    return LoadCase(scenario, items)


def axle_load_change_n(veh: VehicleParams, load: LoadCase) -> Tuple[float, float]:
    """Front/rear axle force change (N) from the load items (lever rule)."""
    L = veh.wheelbase_m
    df = sum(i.mass_kg * G * (L - i.x_m) / L for i in load.items)
    dr = sum(i.mass_kg * G * i.x_m / L for i in load.items)
    return float(df), float(dr)


def static_load_pitch_rad(veh: VehicleParams, load: LoadCase) -> float:
    """Body pitch (nose-up positive) relative to the road caused by the load.

    Each axle compresses by dF / k. Rear compression lowers the rear and
    raises the nose; items behind the rear axle also unload the front axle.
    """
    df, dr = axle_load_change_n(veh, load)
    dz_front = df / veh.front_axle_rate_n_m
    dz_rear = dr / veh.rear_axle_rate_n_m
    return float(np.arctan2(dz_rear - dz_front, veh.wheelbase_m))

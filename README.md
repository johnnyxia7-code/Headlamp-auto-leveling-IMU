# Headlamp auto-leveling with an IMU

This project trains models for **automatic headlamp leveling (AHL)** using a low-cost 6-axis IMU and
the wheel-speed signal, so the vehicle does not need ride-height sensors on the axles.

It covers the full workflow:

1. **Physics simulator.** Simulates vehicle loading, suspension pitch, road grade and roughness,
   driver behaviour and MEMS IMU errors, and outputs labelled training data.
2. **Dataset tooling.** Builds sliding-window datasets and imports real CSV logs. Logs can be
   labelled with ride-height sensors on a reference vehicle.
3. **Estimators.**
   - An analytic physics baseline.
   - Three small neural networks (TCN, GRU, and a physics-structured hybrid), small enough for an
     MCU.
4. **Training and evaluation.** Reports window-level accuracy and a closed-loop simulation of
   **headlamp aim error** with the leveling controller in the loop.
5. **Leveling controller.** Gates estimates on excitation, filters them, tracks load changes at
   standstill, and applies a deadband, slew limit, range clamp and fail-safe.
6. **Export.** Writes TorchScript, ONNX (optional) and a JSON file with the normalisation and
   interface description for embedded integration.

---

## 1. Problem

Load in the cabin, trunk or on a tow ball compresses the rear suspension more than the front. The
body pitches nose-up by roughly 0.3–1.5°, and the low beam rises with it, which can glare oncoming
traffic. AHL lowers the headlamp by the **static load pitch** `θ_load`. That pitch is measured
relative to the *road*, against the reference state "unladen vehicle + driver" that the aim is set
for at end of line.

An accelerometer at rest measures the body attitude relative to *gravity*:

```
φ = γ (road grade) + θ (body pitch relative to road)
```

A static tilt reading therefore cannot separate a loaded car on a flat road from an empty car on a
slope. MEMS bias (±15 mg ≈ ±0.9°) makes it worse. The signal that does work comes from driving
dynamics. With body axes x forward and z up, and pitch positive nose-up:

```
f_x =  a·cos θ + g·sin φ
f_z = −a·sin θ + g·cos φ (+ heave)
```

A change of the along-road acceleration `a` moves the specific-force vector *along the road*.
Correlating `(f_x, f_z)` with the wheel-speed derivative `a` therefore gives
`θ = atan2(−cov(f_z,a), cov(f_x,a))`. Constant grade and constant accelerometer bias both cancel.

Several real-world effects break this ideal picture:

| Effect | Impact |
|---|---|
| Squat / dive (dynamic pitch ≈ 1–4 °/g) | Correlates with `a` and biases the estimate by ~0.3–0.5° |
| Grade changes within the window | Gravity leaks into `f_x` |
| Road roughness (ISO 8608 A–C) | Heave and pitch vibration ≫ the 0.01 m/s² signal |
| Tyre radius / slip, wheel-speed noise | Scale and noise errors on `a` |
| Load changes at stops | Step change in the label |

The physics baseline (`models/physics.py`) compensates the first two with gyro-integrated pitch and
a quadratic squat term. The neural models learn the remaining corrections from data.

**Regulatory context.** UN Regulation No. 48 governs headlamp installation: the initial aim, the
allowed vertical inclination range under the load conditions it defines, and when automatic
leveling is mandatory. These limits are given in percent (1 % = 0.573°). The exact values depend on
the series of amendments, the lamp type and the mounting height, so take them from the current text
for your vehicle category. The evaluation reports aim error in both degrees and percent.

## 2. Repository layout

```
configs/                default.yaml (full scale), quick.yaml (smoke test)
src/headlamp_leveling/
  config.py             typed config (YAML -> dataclasses)
  sim/vehicle.py        vehicle params, load scenarios, static load pitch (lever rule + axle rates)
  sim/road.py           road grade, curvature, ISO 8608 roughness
  sim/driving.py        driver model: launches, cruising, braking, stops
  sim/imu.py            MEMS IMU + wheel-speed error models
  sim/scenario.py       full trip simulation -> sensor log + labels
  data/tripset.py       variable-length trip storage (.npz)
  data/features.py      feature building (+ wheel acceleration), normaliser
  data/dataset.py       sliding-window PyTorch dataset
  data/generate.py      parallel train/val/test generation
  data/real_logs.py     CSV log import (direct labels or ride-height sensors)
  models/physics.py     analytic baseline
  models/nets.py        TCN, GRU, Hybrid (covariance features + TCN)
  training/train.py     training loop, checkpointing
  training/evaluate.py  window metrics, breakdowns, closed-loop aim error
  control/leveling.py   leveling controller
  sim/camera.py         front camera as auxiliary pitch sensor (lane vanishing point)
  fusion/ekf.py         13-state EKF: IMU + wheel speed + camera (+ NN) -> load pitch with sigma
  fusion/run.py         run the EKF on simulated trips, `hal fuse-eval`
  export.py             TorchScript / ONNX / metadata export
  cli.py                `hal` command line
docs/ekf_design.md      EKF specification: states, models, Jacobians, tuning, embedded notes
tests/                  pytest suite
```

## 3. Quick start

```bash
pip install -e ".[dev]"          # numpy, pyyaml, torch, pytest

# smoke test (~2 min on CPU; checks the pipeline only, the 3-epoch model is undertrained)
hal generate -c configs/quick.yaml
hal train    -c configs/quick.yaml --arch tcn
hal evaluate runs/quick/model.pt
hal export   runs/quick/model.pt

# full run (~67 h of simulated driving; ~25 min per model on a 4-core CPU)
hal generate -c configs/default.yaml
hal train    -c configs/default.yaml --arch hybrid -o runs/hybrid
hal evaluate runs/hybrid/model.pt
pytest -q
```

`hal simulate --seed 3 -o trip.npz` prints the scenario of a single simulated trip (vehicle, load
case, road class) and saves its signals for inspection.

Outputs:

- `runs/<name>/model.pt`: checkpoint with the config, normaliser and validation metrics.
- `history.csv`: per-epoch training log.
- `eval_test/report.md|json`: accuracy overall and by load-pitch bin, road grade class, roughness
  class and load case, plus closed-loop aim error for the model vs. the physics baseline.
- `export/`: `pitch_estimator.ts`, `pitch_estimator.onnx` (if `onnx` is installed) and
  `model_meta.json`.

## 4. Signals and conventions

Model input is 5 Hz, decimated with a boxcar anti-alias filter from the 100 Hz IMU. Windows are
120 s long (600 samples). The axes follow ISO 8855 (x forward, y left, z up).

| Channel | Unit | Notes |
|---|---|---|
| `ax, ay, az` | m/s² | specific force (reads +g on `az` at rest) |
| `gx, gy, gz` | deg/s | right-handed, so nose-up pitch rate gives **negative** `gy` |
| `v_wheel` | m/s | vehicle speed from wheel speeds (0 at standstill) |
| `a_wheel` | m/s² | derived: smoothed derivative of `v_wheel` |

Outputs:

- `[0]`: static load pitch in degrees, nose-up positive. This drives the leveling.
- `[1]`: road grade in degrees, uphill positive. This is an auxiliary output.

## 5. Models

| Arch | Idea | Params |
|---|---|---|
| `physics` | gyro-compensated instrumental-variable regression with a squat term; no training | 0 |
| `tcn` | strided, dilated residual 1-D CNN with mean+max pooling | ~121 k |
| `gru` | conv front-end + GRU; can run sample-by-sample with persistent state | ~32 k |
| `hybrid` | learned projections → high-passed **covariance matrix** (the physics observable) + TCN context | ~136 k |

All models are trained with a Huber loss on the load pitch plus a down-weighted road-grade auxiliary
loss. Training uses AdamW with a one-cycle LR schedule and early stopping on validation MAE.
Augmentation adds extra accelerometer/gyro bias and wheel-speed scale on top of the simulator's
domain randomisation.

## 6. Leveling controller

`control/leveling.py` runs at 1 Hz:

- **Driving.** Each window estimate is accepted only if `std(a_wheel) ≥ 0.25 m/s²`. Accepted
  estimates are fused with a first-order filter. The filter gain starts high (`1/(n+1)`) for fast
  convergence after start, then settles to `τ = 20 s`.
- **Standstill.** The grade cannot change while the vehicle is stopped, so any change in
  accelerometer tilt is a load change, for example passengers or luggage. The controller tracks it
  directly.
- **Actuator.** A 0.05° deadband, a 0.5 °/s slew limit, a −1…+3° range and stepper-step output.
- **Fault.** The controller moves to the fail-safe position and latches there.

The closed-loop metric is `aim error = true load pitch − applied correction`. A positive error means
the beam is too high. The evaluation reports it both over the whole trip and after the first full window (120 s).

## 7. Sensor fusion EKF (IMU + wheel speed + camera)

`fusion/ekf.py` estimates the load pitch with a 13-state extended Kalman filter instead of
fixed windows. The full specification, with models, Jacobians, tuning rationale and embedded
notes, is in [`docs/ekf_design.md`](docs/ekf_design.md).

- **States:**
  - speed and acceleration;
  - road grade and grade rate;
  - static load pitch;
  - dynamic pitch and pitch rate, plus the online pitch gradient;
  - accelerometer x/z and gyro biases;
  - tyre-radius scale;
  - camera mounting bias.
- **Measurements, as sequential scalar updates:**
  - accelerometer x/z, with roll compensated from `ay − v·yaw_rate`;
  - gyro pitch rate;
  - wheel speed;
  - camera pitch to the road plane at 30 m look-ahead, with grade-change compensation;
  - optionally, the neural window estimate as a direct load-pitch measurement.
- **Robustness:**
  - measurement noise adapted online, so the filter copes with road roughness varying ~10×;
  - innovation gating against camera outliers;
  - per-sensor NIS fault flags;
  - zero-velocity updates at standstill, with the grade frozen there;
  - event-triggered load-change tracking at standstill, with a delayed decision so a
    drive-away isn't mistaken for loading;
  - covariance re-opened at drive-away.
- **Controller interface:** the controller accepts the EKF estimate whenever its σ ≤ 0.3°.

```bash
hal fuse-eval -c configs/default.yaml -n 40 -o runs/fusion                     # EKF with / without camera
hal fuse-eval -c configs/default.yaml -n 40 --checkpoint runs/tcn/model.pt     # + NN measurement
```

Results on 40 simulated test trips, measured after the first 120 s. These are the same trips as
the window models' test split.

| Variant | Load pitch MAE | P95 | within 0.2° | within 2σ | Closed-loop MAE | Closed-loop P95 | Convergence P90 |
|---|---|---|---|---|---|---|---|
| EKF: IMU + wheel speed | 0.106° | 0.449° | 87.0 % | 86.7 % | 0.113° | 0.467° | 409 s |
| EKF: + camera | 0.071° | 0.236° | 91.8 % | 90.2 % | 0.074° | 0.228° | 83 s |
| EKF: + NN measurement (TCN) | 0.089° | 0.287° | 91.2 % | 86.7 % | 0.095° | 0.309° | 179 s |
| EKF: + camera + NN | **0.064°** | **0.192°** | **95.5 %** | **91.5 %** | **0.068°** | **0.204°** | **83 s** |
| *for comparison: TCN window model (§8)* | 0.143° | 0.435° | 77.8 % | – | 0.159° | 0.569° | – |

- **The EKF beats the window models.** Even without the camera, its closed-loop error is about 30 %
  lower than the TCN's (0.113° vs. 0.159°). The camera roughly halves the TCN's error (0.074°).
  Fusing all three gives the best result, with 98 % of the time within 0.5 % aim error.
- **The camera mainly speeds up convergence from a wrong initial value.** Convergence is measured
  on the 22 trips that start at least 0.3° away from the filter's initial value. The 90th-percentile
  time is 83 s with the camera and 409 s without it. Camera frames are available on only 11 % of
  filter steps.
- **The remaining errors come mostly from ISO 8608 class C (rough) roads** (up to about 0.5° on
  single trips without the camera) and from load changes in the first 2 s of a stop. See
  [`docs/ekf_design.md`](docs/ekf_design.md) §13.
- **The filter is mildly overconfident.** 87–92 % of errors fall within 2σ, against an ideal of
  95 %. The controller's `max_estimate_std_deg` gate leaves margin for this.

## 8. Results (simulated test split)

These are results for the window models (TCN, hybrid) and the physics baseline on the simulated
test split: 80 unseen trips (13.3 h), each 10 min long, with 7,760 windows. The models were trained
for up to 25 epochs on 400 trips with `configs/default.yaml`. For the sensor-fusion EKF, see §7.

**Window-level load-pitch error**

| Estimator | MAE (deg) | RMSE | P95 | within 0.1° | within 0.2° |
|---|---|---|---|---|---|
| physics baseline | 0.196 | 0.313 | 0.601 | 41.0 % | 67.9 % |
| `tcn` | **0.143** | **0.221** | **0.435** | **52.2 %** | **77.8 %** |
| `hybrid` | 0.158 | 0.241 | 0.447 | 47.3 % | 72.8 % |
| `gru` | not benchmarked at full scale | | | | |

**Closed loop: estimator + leveling controller, aim error after the first window**

| Estimator | MAE (deg) | P95 (deg) | within 0.25 % | within 0.5 % |
|---|---|---|---|---|
| physics baseline | 0.186 | 0.624 | 55.3 % | 83.0 % |
| `tcn` | **0.159** | **0.569** | **62.8 %** | **86.2 %** |
| `hybrid` | 0.172 | 0.577 | 58.0 % | 83.4 % |

What these numbers show:

- **Window length matters most.** With 30 s windows at 20 Hz, neither the physics baseline nor the
  networks got below about 0.4° MAE. The load signal (`a·θ ≈ 0.01 m/s²`) needs several
  acceleration and braking events to rise above ride vibration. Moving to 120 s windows at 5 Hz
  keeps the same 600-sample input and cut the error to the values above.
- **The learned models help most where the physics baseline is weak.** On mountain roads the TCN
  error is 0.191° against 0.292° for the baseline. With a trailer it is 0.113° against 0.187°.
- **Heavy loads are still hardest.** For load pitch above 1.25°, the TCN MAE is 0.26° with a bias of
  −0.16°. These loads are rare in the training mix, and the Huber loss shrinks predictions toward
  the mean.
- **The controller narrows the gap.** Its filtering averages the window estimates, so the
  closed-loop MAE of the model and the baseline end up close.
- **Next steps for the window models:** weight the heavy-load cases, and use the models as an extra
  measurement in the EKF rather than on their own. That fused variant is the best one in §7.
- **Before the first full window,** the controller holds its initial value (the reference state).
  This is why the whole-trip MAE is about 0.25°. In a vehicle, the value stored in NVM at the last
  ignition-off covers this period.

The full per-category tables are in `runs/<arch>/eval_test/report.md`.

## 9. Using real vehicle data

The simulator only provides the starting point. Close the sim-to-real gap with logs from an
instrumented reference vehicle.

1. Record the IMU at the headlamp ECU mounting position, wheel speed from CAN, and front/rear
   ride-height sensors. Cover representative loads (driver only up to full plus trunk and a trailer),
   roads and drivers.
2. Export one CSV per drive with the columns `t, ax, ay, az, gx, gy, gz, v_wheel` plus either
   `load_pitch_deg`, or `h_front_mm, h_rear_mm`.
3. Import the logs. This command converts ride heights to pitch relative to the reference heights
   and low-pass filters them to the static component:

   ```bash
   hal import-logs logs/*.csv -o data/real/test.npz --wheelbase 2.85 --h-ref-front 402 --h-ref-rear 395
   ```

4. Evaluate the simulator-trained model with `hal evaluate runs/hybrid/model.pt -d data/real`.
   Then fine-tune by putting real `train.npz` / `val.npz` into a directory and training on it, with
   a lower `lr` and fewer epochs.
5. Re-fit the simulator ranges in `configs/*.yaml` (axle rates, pitch gradient, IMU noise) to the
   measured vehicle so the synthetic data stays useful for rare cases.

## 10. Deployment notes

- **MCU budget.** The GRU (~32 k parameters) is the cheapest model to deploy. Its streaming
  form needs no 120 s window buffer. The TCN and hybrid models fit in int8 within ~150 kB of flash.
  Inference runs at 1 Hz.
- **Interface.** The interface is described in `export/model_meta.json`: feature order,
  normalisation mean/std, sample rate, window length, output semantics and controller parameters.
- **Functional safety.** A wrong aim causes glare or reduced visibility. Run the HARA/FMEA for your
  item definition. Recommended diagnostics include plausibility checks on the IMU (norm ≈ g at
  standstill, gyro/accel consistency), wheel-speed timeouts, a range check on the estimate and the
  latched fail-safe position.
- **End-of-line.** Calibrate the IMU mounting pitch together with the headlamp aim. Residual
  misalignment adds directly to the error (simulated at 0.05° 1σ).
- **EKF on target.** See [`docs/ekf_design.md`](docs/ekf_design.md) §11: about 0.35 MMAC/s at
  50 Hz, about 1 kB of RAM, and notes on single precision.
- **NVM.** Store the last fused estimate at ignition-off and restore it at start, so the lamps are
  correct before the first excitation.

## 11. Limitations / next steps

- The vehicle model is planar (pitch, heave, simplified roll). There is no tyre or suspension
  kinematic coupling, and no anti-squat or anti-dive geometry beyond the pitch gradient.
- The IMU is assumed to sit on the body. If it sits in the headlamp itself, the actuator position
  must be removed from the measured pitch.
- Labels from ride-height sensors include suspension hysteresis. Static labels taken at standstill
  are the most reliable.
- The model could gain a confidence or uncertainty output (for example a Gaussian NLL head), which
  the controller could use instead of the excitation heuristic.

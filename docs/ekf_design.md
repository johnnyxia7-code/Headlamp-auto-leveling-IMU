# Sensor-fusion EKF for automatic headlamp leveling

This document specifies the extended Kalman filter (EKF) in
`src/headlamp_leveling/fusion/ekf.py`. The filter fuses a 6-axis IMU, the
wheel-speed signal and an auxiliary front camera. It can also take the learned
window estimator as an extra measurement. Its output is the **static load pitch**
of the body relative to the road, together with its uncertainty. The leveling
controller turns that into a headlamp correction.

Conventions throughout: ISO 8855 vehicle axes (x forward, y left, z up), angles
nose-up positive, SI units inside the filter.

---

## 1. Why a filter, and what it has to separate

The accelerometer measures the specific force

```
f_x =  a·cos θ + g·sin(γ + θ) + b_ax
f_z = −a·sin θ + g·cos(γ + θ)·cos ρ + b_az
```

The symbols are:

- `a`: along-road acceleration.
- `γ`: road grade.
- `θ = θ_s + θ_d`: body pitch relative to the road. `θ_s` is the static load
  pitch (the wanted output); `θ_d` is squat and dive under acceleration.
- `ρ`: roll.
- `b`: accelerometer biases.

At standstill only the sum `γ + θ + b_ax/g` is visible, so a loaded car on a flat
road looks the same as an empty car on a slope. Driving breaks this ambiguity in
three ways:

- **Acceleration changes.** A change of `a` moves the specific-force vector along
  the road, so its direction in the body frame shows `θ`. This shows up in the
  `−a·sin θ` term of `f_z`.
- **The gyro** measures `γ̇ + θ̇_d`, so grade changes and squat and dive do not leak
  into the estimate.
- **The camera** measures the pitch to the road plane ahead directly.

A window regression (`models/physics.py`) or a neural network sees only a finite
window and has to learn all of this implicitly. The EKF models these effects
explicitly and accumulates information over the whole drive. It also gives a
calibrated uncertainty that the controller can act on.

---

## 2. Architecture

```
 IMU 100 Hz ─┐                                                     ┌─> load pitch θ_s, σ(θ_s)
 (ax ay az   │   ┌──────────────────────┐   ┌───────────────────┐  │   road grade γ, σ(γ)
  gx gy gz)  ├──>│ pre-processing 50 Hz │──>│        EKF        │──┼─> standstill flag
 wheel speed ┤   │ · block average      │   │ predict           │  └─> sensor fault flags
 (CAN)       │   │ · roll from ay − v·r │   │ ZUPT / wheel      │            │
 camera      ┤   │ · adaptive R (diff)  │   │ gyro, acc_x, acc_z│            v
 10–15 Hz    │   │ · standstill detect  │   │ camera (gated)    │   ┌───────────────────┐
 (+ quality) │   │ · load-change detect │   │ NN (optional)     │   │ leveling controller│─> stepper
 NN window ──┘   └──────────────────────┘   └───────────────────┘   └───────────────────┘
 estimate 0.02 Hz
```

The steps run in this order in every 20 ms cycle (`PitchEKF.step`):

1. Roll `ρ = asin((a_y − v·ω_z)/g)` from the lateral channel.
2. Standstill detection. When the car leaves standstill, the speed and
   acceleration covariances are re-opened.
3. Load-change detection (standstill only).
4. Prediction.
5. Adaptive measurement-noise tracking.
6. Updates, each a sequential scalar update:
   - at standstill, ZUPT pseudo-measurements; otherwise the wheel-speed update;
   - gyro;
   - `acc_x`, then `acc_z`;
   - camera, when a frame is available.
7. NN measurement, when one arrives (in `fusion/run.py`).

---

## 3. State vector (13)

| # | State | Meaning | Unit | Initial σ | Process noise (per √s) |
|---|---|---|---|---|---|
| 0 | `v` | speed along the road | m/s | 0.5 | 0.01 |
| 1 | `a` | along-road acceleration | m/s² | 0.5 | `q_jerk` = 2.0 |
| 2 | `γ` | road grade | rad | 8° | – (driven by `γ̇`) |
| 3 | `γ̇` | grade rate | rad/s | 0.01 | `q_grade_curv`·√v (0 at standstill) |
| 4 | `θ_s` | **static load pitch** | rad | 1.0° (or NVM) | 0.002° driving; 2°/√s on a load event |
| 5 | `θ_d` | dynamic pitch (squat/dive) | rad | 0.2° | – |
| 6 | `θ̇_d` | dynamic pitch rate | rad/s | 0.5°/s | `q_pitch_dyn_rate`·v |
| 7 | `k_p` | pitch gradient | rad/(m/s²) | 40 % of nominal | 0.1 % of nominal |
| 8 | `b_ax` | accel x bias | m/s² | 20 mg | 0.02 mg |
| 9 | `b_az` | accel z bias | m/s² | 20 mg | 0.02 mg |
| 10 | `b_gy` | gyro y bias | rad/s | 0.5°/s | 0.002°/s |
| 11 | `s_w` | wheel-speed scale (tyre radius) | – | 2 % | 1e-5 |
| 12 | `b_cam` | camera pitch extrinsic bias | rad | 0.2° | 1e-4° |

The squat and dive dynamics use a nominal per-vehicle-programme natural frequency
and damping (`pitch_freq_hz`, `pitch_damping`). The pitch gradient `k_p` is
estimated online, because it changes with load and differs between vehicles.

## 4. Process model

The model is Euler-discretised with `dt = 1/50 s` (`propagate()`):

```
v'    = v + a·dt
a'    = a                                       + w_jerk
γ'    = γ + γ̇·dt
γ̇'    = γ̇·(1 − dt/τ_γ)                          + w_curv        (variance ∝ distance travelled)
θ_s'  = θ_s                                     + w_load
θ_d'  = θ_d + θ̇_d·dt
θ̇_d'  = θ̇_d + (ω²(k_p·a − θ_d) − 2ζω·θ̇_d)·dt   + w_road
k_p, b_ax, b_az, b_gy, s_w, b_cam: random walks
```

The non-trivial entries of the Jacobian `F = ∂f/∂x` are:

```
∂v'/∂a = dt     ∂γ'/∂γ̇ = dt     ∂γ̇'/∂γ̇ = 1 − dt/τ_γ     ∂θ_d'/∂θ̇_d = dt
∂θ̇_d'/∂a = ω²k_p·dt   ∂θ̇_d'/∂θ_d = −ω²dt   ∂θ̇_d'/∂θ̇_d = 1 − 2ζω·dt   ∂θ̇_d'/∂k_p = ω²a·dt
```

Two process-noise choices carry physical meaning:

- **Grade rate.** Noise on `γ̇` scales with the distance travelled, so it is zero
  at standstill: the road cannot change while the car is stopped.
- **Static load pitch.** `θ_s` is nearly constant while driving (fuel use,
  settling). Its process noise is raised only on a detected load change at
  standstill (section 7).

## 5. Measurement models

The functions `h_*()` return both `h(x)` and the Jacobian row `H`. Every one of
them is checked against finite differences in `tests/test_fusion.py`.

| Sensor | Model `h(x)` | Non-zero `H` | R |
|---|---|---|---|
| `acc_x` | `a cos θ + g sin φ + b_ax` | `a: cos θ`; `θ_s, θ_d: −a sin θ + g cos φ`; `γ: g cos φ`; `b_ax: 1` | adaptive |
| `acc_z` | `−a sin θ + g cos φ cos ρ + b_az` | `a: −sin θ`; `θ_s, θ_d: −a cos θ − g sin φ cos ρ`; `γ: −g sin φ cos ρ`; `b_az: 1` | adaptive |
| `gyro` | `−(γ̇ + θ̇_d) + b_gy` | `γ̇, θ̇_d: −1`; `b_gy: 1` | adaptive |
| `wheel` | `s_w · v` | `v: s_w`; `s_w: v` | `(0.03 + 0.003·|a|)²` (slip) |
| `camera` | `θ_s + θ_d − γ̇·d/v + b_cam` | `θ_s, θ_d, b_cam: 1`; `γ̇: −d/v`; `v: γ̇·d/v²` | `(0.25°)²` |
| `nn` | `θ_s` | `θ_s: 1` | `σ_nn²` |

In the table, `φ = γ + θ` and `θ = θ_s + θ_d`. Roll `ρ` is an input, not a state.

**Camera.** The vanishing point of the lane markings lies on the horizon of the
road plane at the look-ahead distance `d` (about 30 m). The camera therefore
measures the body pitch relative to the road *ahead*. If the grade changes between
the car and that point, the reading shifts by `γ(s+d) − γ(s) ≈ γ̇·d/v`. The model
includes this term, so cresting a hill does not look like a load change. A frame
is used only above 5 m/s, when the lane-detection quality flag is set (simulated as
availability), and when it passes a χ²(1) 99.9 % innovation gate. The gate rejects
misdetections such as shadows or wrong lane association. The mounting bias
`b_cam` is estimated, so the camera does not need a perfect extrinsic calibration.

**Camera latency (production).** Image processing typically adds 50–150 ms.
`θ_s` changes slowly, so the latency only matters through `θ_d`. Store
`(θ_d, γ̇, v)` for the last 200 ms and evaluate `h` and `H` with the values at the
frame timestamp. The simulator has no latency.

**Neural-network measurement.** The window model (`training/`) can be fused as a
direct observation of `θ_s` (`update_load_pitch`). Consecutive windows overlap, so
their errors are correlated. Updates are therefore applied only every `window/2`,
with `σ_nn` set to about 1.5–2× the model's validation RMSE.

## 6. Measurement noise: adaptive R

Measured against the simulator's ground truth, the vertical-accelerometer noise at
the 50 Hz filter rate is **mostly white**. Its lag-1 autocorrelation is 0.15–0.5,
and it decays within about 0.1 s. Its level depends strongly on the road:
0.13–0.36 m/s² on ISO 8608 class A roads, but 0.7–1.5 m/s² on class C roads. A
fixed R is therefore either overconfident on rough roads or wastes information on
smooth ones.

The filter estimates each channel's noise variance online from first differences.
For white noise plus a slow signal, `½·E[(z_k − z_{k−1})²]` equals the noise
variance:

```
σ̂²_k = σ̂²_{k−1} + α·(½(z_k − z_{k−1})² − σ̂²_{k−1}),   α = dt / 2 s
R = max(R_floor, σ̂²) · 2.0          (factor 2: residual correlation of road vibration)
```

This adaptation needs no vehicle model, so filter errors cannot feed back into it
(an innovation-based R estimate can lock onto those errors). Section 10 shows the
resulting consistency: 87–92 % of errors fall within 2σ, against an ideal of 95 %.

A shaping-filter model (heave states driven by road-velocity noise) was also tried.
It made consistency worse, because the disturbance is not concentrated at the
body-bounce frequency.

## 7. Standstill logic

**Detection.** A standstill is declared when the wheel speed has stayed below
0.05 m/s for 0.5 s. The filter then applies:

- zero-velocity pseudo-measurements `v = 0` (σ 0.01), `a = 0` (σ 0.05) and `γ̇ = 0`;
- no grade process noise, so the grade is frozen.

Gyro bias becomes observable, because the gyro must read `b_gy`.

**Load changes.** Doors open and passengers or luggage come and go. While the car
stands still, any change in the accelerometer tilt is a load change, because the
road cannot change. The handling is event-triggered:

1. The tilt is `atan2(f_x, f_z) − θ_d`, with the filter's dynamic pitch removed,
   and low-pass filtered (τ = 0.3 s). Removing `θ_d` keeps the brake-dive rebound,
   which the gyro observes through `θ̇_d`, from looking like loading.
2. After a settle time, a reference tilt is stored.
3. The decision uses the tilt from **1 s earlier**, and is taken only if the car
   is still standing now. A drive-away begins before the wheel-speed sensor
   reports motion, and without this delay that onset was mistaken for loading.
4. If the delayed tilt differs from the reference by 0.04°–3°, the filter raises
   the `θ_s` process noise for that step and moves the reference. Changes of more
   than 3° are vehicle motion, not load.

Raising the `θ_s` noise for the whole stop instead let the brake rebound leak into
`θ_s`. In the simulation that produced up to 2° errors.

**Drive-away.** When the standstill ends, the variances of `v` and `a` are
re-opened (σ_a = 2 m/s²) and their correlations are cleared. Otherwise the launch
acceleration, which `acc_x` sees before the wheel-speed derivative catches up,
would be explained as body pitch.

## 8. Robustness and diagnostics

- **Innovation gating.** Each scalar update computes `NIS = ν²/S`. Camera and NN
  measurements use the χ²(1) 99.9 % gate (10.8). The IMU and wheel speed use a
  loose gate (30) that only rejects gross errors.
- **Fault detection.** Each sensor keeps an exponential moving average of its NIS
  over a 5 s window. Before averaging, every sample is capped at 25, so a single
  transient cannot trip the monitor: a drive-away, a pothole or an outlier frame.
  An average above 9 sets a fault flag. Monitoring starts only after a 30 s
  warm-up, while the filter is still converging.
- **Fault classes.**
  - *Critical:* `acc_x`, `acc_z`, gyro and wheel speed. A fault here sets
    `critical_fault`. The controller then moves to the fail-safe position and
    latches there.
  - *Auxiliary:* camera and NN. A fault here only disables that input; the IMU and
    wheel-speed filter carries on.

  `tests/test_fusion.py` injects a gyro stuck at +20 °/s and checks that it is
  flagged. Earlier versions had no warm-up and no cap, and they let a camera fault
  latch the fail-safe. In those versions, false faults fired on 7 of 40 test trips;
  in one trip the lamp was parked at the fail-safe position, giving a 1.18°
  closed-loop error. Since the fix, no test trip shows a false fault.
- **Numerical safety.** Updates are sequential and scalar, so no matrix inversion
  is needed. After each update the covariance is symmetrised.
- **Controller interface.** The controller uses the EKF estimate directly whenever
  `σ(θ_s) ≤ 0.3°` (`max_estimate_std_deg`). Otherwise it holds its last position.

## 9. Observability summary

| Quantity | Observable from |
|---|---|
| `θ_s` | acceleration changes (`a·θ` term in `f_z`), camera, tilt changes at standstill |
| `θ_d`, `θ̇_d`, `k_p` | gyro during acceleration and braking (`θ̇_d` follows `k_p·a`) |
| `γ` | only `γ + b_ax/g` is observable. The grade estimate carries the x-bias (about 0.6° MAE in simulation), which does not matter because it is auxiliary |
| `b_az` | separable from `θ_s` because `a` varies |
| `b_gy` | standstill (ZUPT) |
| `s_w` | acceleration events (scale of `a` from wheel speed versus `f_x`) |
| `b_cam` | camera versus the IMU-derived `θ_s` over time |

## 10. Results (simulation)

To reproduce:

```bash
hal fuse-eval -c configs/default.yaml -n 40 -o runs/fusion [--checkpoint runs/tcn/model.pt]
```

The evaluation uses the same trips as the test split of the neural models. Metrics
are taken after the first 120 s (the window length of the neural models), so the
two can be compared.

| Variant | Load pitch MAE | P95 | within 0.2° | within 2σ | Closed-loop MAE | Closed-loop P95 | Convergence P90 |
|---|---|---|---|---|---|---|---|
| EKF: IMU + wheel speed | 0.106° | 0.449° | 87.0 % | 86.7 % | 0.113° | 0.467° | 409 s |
| EKF: + camera | 0.071° | 0.236° | 91.8 % | 90.2 % | 0.074° | 0.228° | 83 s |
| EKF: + NN measurement (TCN) | 0.089° | 0.287° | 91.2 % | 86.7 % | 0.095° | 0.309° | 179 s |
| EKF: + camera + NN | **0.064°** | **0.192°** | **95.5 %** | **91.5 %** | **0.068°** | **0.204°** | **83 s** |
| *for comparison: TCN window model (README §8)* | 0.143° | 0.435° | 77.8 % | – | 0.159° | 0.569° | – |

- **Convergence** is measured on the 22 trips whose true load differs from the initial value by at
  least 0.3°.
- **Consistency:** 87–92 % of errors fall within 2σ, slightly below the ideal 95 %. The worst trips
  are on ISO 8608 class C roads, where the residual correlation of the vibration exceeds the
  inflation factor of 2.
- **Fault handling:** no false faults on any of the 40 trips.

## 11. Embedded implementation notes

- **Cost.** With 13 states, one prediction is two 13×13 matrix products (about
  4.4 k MAC). The six scalar updates cost about 2·13² MAC each. At 50 Hz the total
  is about 0.35 MMAC/s, which is well within a Cortex-M4F. RAM is about 1 kB for
  `P`, `x` and the buffers.
- **Single precision.** `P` spans about 12 orders of magnitude (wheel scale versus
  grade rate). Options:
  - Scale the states to similar magnitudes: degrees and milli-g as units.
  - Or use a UD- or Joseph-form update.
  - In either case, symmetrise after each update and floor the diagonal.
- **Rates.** IMU at ≥ 100 Hz, block-averaged to 50 Hz (anti-alias). Wheel speed
  from CAN at 50–100 Hz, with the ABS sensor's dead zone at about 1 km/h handled by
  the standstill logic. Camera at 10–15 Hz, using the frame timestamp.
- **NVM.** At ignition-off, store `θ_s`, `σ(θ_s)`, the biases, `s_w`, `k_p` and
  `b_cam`. At ignition-on, call `reset(load_pitch_deg, load_pitch_std_deg)`
  with σ inflated (e.g. +0.3°), because the load may have changed while parked.
  The standstill load-change logic also covers loading after start-up.
- **End-of-line.** The IMU mounting pitch is calibrated together with the headlamp
  aim. The camera extrinsic only needs to be roughly right, because `b_cam` is
  estimated.

## 12. Validation plan

1. **Simulation.** `hal fuse-eval` over all load cases, road grade and roughness
   classes, with the camera on and off. Repeat with the vehicle parameters swept
   beyond the nominal model (`pitch_freq_hz`, `k_p`).
2. **Fault injection.** Stuck, offset and noisy IMU channels, wheel-speed dropout
   and camera outliers. Check fault detection and the fail-safe.
3. **Hardware-in-the-loop.** Replay recorded CAN and IMU logs at real time on the
   target ECU. Check that the results match the Python reference bit for bit, to
   within float tolerance.
4. **Vehicle tests.** Use a reference car with ride-height sensors, imported with
   `hal import-logs`. Run the load-case matrix from the UN R48 load conditions,
   plus trailer, on urban, rural, motorway and mountain roads, by day and night
   (camera availability).

## 13. Known limitations

- The vehicle model is planar. Load-dependent roll, and pitch coupling from
  anti-dive and anti-squat geometry, are only approximated by `k_p`.
- `γ` is biased by `b_ax/g` (section 9).
- The simulated camera has no latency, rolling shutter or lens model.
- A load change during roughly the first 2 s of a stop, before the reference tilt
  is stored, is not caught at standstill. It is corrected later through driving
  excitation, at the convergence rate shown above. The settle time trades this
  against brake-rebound leakage. A sweep over 16 trips gave these camera-off MAEs:
  1.0 s → 0.088°, 1.5 s → 0.076°, 2.0 s → 0.069°, 2.5 s → 0.064°. The default is
  2.0 s, because passengers rarely get out sooner.

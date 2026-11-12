# Verification simulator

`hal verify` checks whether the whole leveling chain behaves plausibly: input
layer → EKF → leveling controller → lamp. It runs scripted driving scenarios on
an independent vehicle model, applies explicit pass/fail criteria, and writes a
report with a time-series plot per scenario.

```bash
hal verify --list                              # the scenarios
hal verify -o runs/verify                      # all scenarios, one seed, report + plots
hal verify --seeds 4                           # each scenario with four seeds
hal verify --scenario hill_route --seed 3      # one scenario, one seed
hal verify --no-plots --strict                 # for CI: no matplotlib, exit 1 on a failure
```

Output: `runs/verify/report.md`, `report.json` and one PNG per scenario. The
plots need matplotlib (`pip install -e ".[plot]"`).

---

## 1. Why a second vehicle model

The training simulator (`sim/scenario.py`) and the EKF make the same
simplification: body pitch is a second-order response to the longitudinal
acceleration, and the IMU sits on the pitch axis. Testing the filter only against
that model checks the arithmetic, not the assumptions. Results on it are
optimistic by construction.

The verification plant (`sim/halfcar.py`) is built from the mechanics instead:

| | Training simulator | Verification plant |
|---|---|---|
| Body | pitch only, second-order in `a` | heave and pitch, front and rear unsprung masses (4 DOF) |
| Suspension | one pitch stiffness and damping | separate front / rear springs, dampers with friction, tyre stiffness |
| Load | changes the static pitch only | point masses: change mass, inertia and trim |
| Squat / dive | one gradient | load transfer at CG height, different anti-squat and anti-dive |
| Slope | tilts gravity only | also pitches the body (load transfer from `g sin γ`) |
| Road roughness | one input, through the pitch mode | front axle, then the rear one wheelbase later |
| IMU position | on the pitch axis | lever arm (default 0.9 m ahead, 0.1 m below the CG) |
| Driving | random cycle | scripted manoeuvres, repeatable |
| Rate | 100 Hz | 1 kHz plant, 100 Hz sensors |

Sensor error models, the camera model and the fault injection are shared with the
training simulator.

The plant's own physics is tested against hand calculations in
`tests/test_verify.py`: static trim against the lever rule, steady squat against
the analytic pitch gradient, and the accelerometer reading while standing on a
slope.

Two definitions matter for reading the results:

- **Pitch to compensate.** The truth the lamp is judged against is the body's
  quasi-static pitch relative to the road: the load pitch plus the pitch from
  standing or driving on a slope. Transient squat and dive are excluded, because a
  static leveling system must not follow them.
- **Aim error** = pitch to compensate − applied correction. Positive means the
  beam is too high (glare); negative means too low (less range).

## 2. Scenarios and criteria

A scenario is a script of manoeuvres (`hold`, `to <speed> at <accel>`), a road
(grade and curvature profiles, ISO 8608 roughness class), a load timeline, and
optionally sensor faults. `verify/scenarios.py` holds 18 of them; adding one is a
function of about ten lines.

The criteria and their default limits:

| Criterion | Requirement |
|---|---|
| Convergence | from an unknown load, \|aim error\| ≤ 0.25° within 90 s of driving off, and stays there |
| Steady-state accuracy | aim error MAE ≤ 0.15° and P95 ≤ 0.30° once settled |
| No glare | beam never more than 0.35° above target once settled |
| Lamp steady | lamp moves ≤ 0.05–0.15° in a phase where the load cannot have changed |
| Aim held | \|aim error\| ≤ 0.25° throughout a phase |
| Load change tracked | \|aim error\| ≤ 0.25° within 12 s of a load change at standstill |
| Actuator activity | lamp travel ≤ 0.5°/min while the load is constant |
| Filter consistency | estimate error within 3σ at least 95 % of the time |
| No false alarm | no fail-safe and no sensor disqualified on healthy data |
| Fault isolated | faulty sensor disqualified within 2–6 s |
| Beam not raised | correction drops by ≤ 0.10° while a fault is present |
| Recovery | back within 0.25° and out of fail-safe within 60 s after the fault ends |
| Leveling continues | no fail-safe at all for a non-critical fault |
| Numerically sound | all outputs finite, no filter reset |

**These limits are engineering targets I set before running the scenarios.** They
are not derived from a requirement specification. 0.25° is about 0.44 % of
headlamp aim. Replace them with the project's real requirements.

"Settled" excludes the first 90 s after driving off, 15 s after each load change,
and each fault window plus 60 s. Scenarios that require riding through a fault
count the fault window too.

## 3. Results

Each scenario was run with four seeds. The seed draws the sensor errors (biases,
mounting misalignment, noise), the camera bias and the road profile.

**66 of 72 runs pass. 12 of the 18 scenarios pass with every seed; the other six
each fail with one seed.**

| Scenario | What it proves | Seeds passed | Steady-state aim error (MAE per seed) | Failed checks |
|---|---|---|---|---|
| `city_loaded` | finds an unknown load from normal driving; lamp still while unobservable | 4/4 | 0.112° / 0.136° / 0.080° / 0.063° | – |
| `city_no_camera` | same without the camera | 3/4 | 0.137° / 0.156° / 0.098° / 0.109° | steady-state accuracy (seed 1) |
| `motorway_cruise` | no drift over 200 s without excitation | 4/4 | 0.027° / 0.075° / 0.120° / 0.141° | – |
| `hill_route` | slope is not mistaken for load; slope-induced pitch is followed | 3/4 | 0.055° / 0.075° / 0.067° / 0.139° | convergence (seed 3); steady-state accuracy (seed 3); filter consistency (seed 3); aim held (seed 3) |
| `parked_on_slope_loading` | loading on a 10 % slope is attributed to the load | 4/4 | 0.046° / 0.049° / 0.028° / 0.075° | – |
| `unload_at_stop` | 2.2° step when the load is removed | 4/4 | 0.044° / 0.075° / 0.031° / 0.033° | – |
| `sporty_driving` | lamp does not chase squat and dive | 4/4 | 0.037° / 0.067° / 0.064° / 0.068° | – |
| `rough_road` | ISO 8608 class C road (relaxed limits) | 3/4 | 0.087° / 0.261° / 0.151° / 0.186° | convergence (seed 1); steady-state accuracy (seed 1) |
| `curvy_road` | lateral dynamics do not leak into pitch | 4/4 | 0.133° / 0.111° / 0.000° / 0.131° | – |
| `urban_turns` | junction turns and roundabouts (yaw rate up to 25 °/s) are not read as load | 4/4 | 0.028° / 0.027° / 0.010° / 0.007° | – |
| `soft_vehicle` | different vehicle, filter not retuned | 3/4 | 0.142° / 0.211° / 0.105° / 0.004° | convergence (seed 1); steady-state accuracy (seed 1); filter consistency (seed 1) |
| `lever_arm_miscalibrated` | IMU position known only to 25 % | 4/4 | 0.135° / 0.137° / 0.075° / 0.104° | – |
| `fault_wheel_speed_zero` | wheel speed reads 0 while driving | 3/4 | 0.103° / 0.126° / 0.090° / 0.082° | beam not raised during fault (seed 2) |
| `fault_imu_outage` | IMU missing for 15 s | 4/4 | 0.102° / 0.135° / 0.098° / 0.074° | – |
| `fault_accel_z_frozen` | vertical accelerometer frozen 30 s | 4/4 | 0.087° / 0.103° / 0.125° / 0.077° | – |
| `fault_gyro_offset` | gyro offset: degraded mode, no fail-safe | 3/4 | 0.145° / 0.165° / 0.080° / 0.117° | steady-state accuracy (seed 1) |
| `fault_glitches` | glitches on 0.5 % of IMU samples | 4/4 | 0.112° / 0.132° / 0.082° / 0.061° | – |
| `fault_camera_wrong` | camera 1.2° wrong for 60 s | 4/4 | 0.118° / 0.146° / 0.080° / 0.063° | – |

Reading the results:

- **Convergence is fast.** In most runs the aim error is within 0.25° 1–3 s after
  driving off. Four runs take 100–260 s: rough road with seeds 0 and 1, soft
  vehicle with seed 1, and hill route with seed 3.
- **Typical steady-state error is 0.06–0.14°**, against a 0.15° limit. The filter
  sits close to that limit, so a seed with unlucky sensor errors crosses it.
- **Every failure is a margin failure.** The worst steady-state error in any run
  is 0.26° (rough road, seed 1). Nothing diverges, and no run ends in a false
  fail-safe.
- **The faults are handled.** All fault scenarios isolate the fault in time and
  recover. One run (wheel speed reads zero, seed 2) lets the beam rise by 0.16°
  during the fault, against a 0.10° limit.

### What limits the accuracy

Measured on `city_loaded` without the camera, by switching effects off one at a
time:

| Effect | Contribution to the aim error |
|---|---|
| Residual IMU mounting misalignment (0.05° 1σ after end-of-line calibration) | between −0.05° and +0.08° across the four seeds; adds directly, cannot be estimated |
| Squat / dive asymmetry | about 0.045° with one pitch gradient; the second gradient (finding 4) removed about 0.03° of it |
| Everything else (noise, roughness) | about 0.02° mean, ±0.06° scatter |

The mounting misalignment is the largest single term, and it is a calibration
quantity, not an algorithm one.

## 4. Findings so far

The simulator found five problems that the training simulator could not show.
Four are fixed.

1. **IMU lever arm (fixed).** With the IMU 0.9 m ahead of the pitch axis, every
   acceleration change adds `x · pitch acceleration` to the vertical
   accelerometer, correlated with the acceleration itself. The load pitch was
   biased by +0.2° on the sedan and −0.5° on the soft SUV. The EKF now removes the
   term using the gyro (`fusion.imu_lever_x_m`, `imu_lever_z_m`).
2. **Gyro noise in that compensation (fixed).** A two- or three-point derivative of
   the gyro amplifies its noise about 100 times. The same noisy sample is also the
   gyro measurement, and the correlated errors biased the load pitch by up to
   0.9°. The derivative is now the least-squares slope over five samples, centred
   on the compensated sample, at the cost of a 40 ms delay.
3. **False fail-safe when loading on a slope (fixed).** The tilt step from loading
   the car looked like a sensor inconsistency to the residual monitor, which
   latched the fail-safe for the rest of the trip. The monitor now pauses during
   and just after a load change at standstill.
4. **Different squat and dive (fixed).** Real suspensions have different
   anti-squat and anti-dive. One pitch gradient left about 0.05° of bias. The EKF
   now estimates a second gradient for braking.
5. **Slope-induced pitch (open, tolerated).** A car on a slope is pitched relative
   to the road by load transfer, about 0.15° on an 8 % grade. The EKF has no model
   for it. It still follows it, through the slow load-pitch state and the camera,
   and `hill_route` passes with three of four seeds.

   Modelling it inside the filter was tried and removed. Grade and accelerometer
   bias cannot be told apart, and once the grade was allowed to drive the body
   pitch, the filter used it to explain every other mismatch. It reported a 4.7°
   slope on a flat road, and at best 2 of 17 scenarios passed.

A sixth finding concerns the scenarios, not the algorithm: my first criterion for
the hill route required the lamp not to move at all. That ignored finding 5 and
was wrong.

## 5. Sensitivities worth knowing

- **IMU position.** Telling the filter a lever arm that is 25 % short
  (`lever_arm_miscalibrated`) still passes with all four seeds; the error grows
  by about 0.02°. Not telling it at all costs 0.2–0.5°.
- **IMU location.** Six mounting locations are compared in
  [`imu_mounting.md`](imu_mounting.md). Inside the cabin the location hardly
  matters; alignment and temperature stability matter much more.
- **Gyro loss.** Without the gyro the lever-arm term cannot be removed. Leveling
  continues with about 0.03° more error on the sedan (`fault_gyro_offset`).
  Predicting the term from the filter's own states was tried and was worse (0.5°).
- **Vehicle mismatch.** A plant with 1.08 Hz pitch frequency and 4.5°/g squat,
  against the filter's nominal 1.4 Hz and 2.5°/g, passes with three of four seeds
  without retuning.

- **Camera fault monitor.** The monitor's window is 50 camera frames
  (`fusion.fault_window_aux_updates`). With 100 frames, a correct camera is never
  distrusted on the training simulator's 40 trips, but `fault_camera_wrong` fails
  with one seed (P95 0.75°). With 50 frames, all four seeds pass, and on the
  training simulator the mean aim error is 0.01° higher because a correct camera is
  distrusted on an occasional trip.

## 6. Limitations of the verification itself

- The plant is still a model. It is planar: no roll dynamics beyond a roll
  gradient, no tyre longitudinal dynamics, linear springs, no bump stops, no
  aerodynamic pitch.
- Load changes are a 2 s ramp of point masses. Real loading is slower and less
  regular.
- The sensor error models are shared with the training simulator. An error in
  them would not be caught here.
- Four seeds show whether a verdict is stable, not its probability. A release
  decision needs more seeds and, above all, recorded vehicle data: replay it with
  `hal import-logs` and compare against ride-height sensors.
- The criteria limits are my targets, not requirements (see section 2).

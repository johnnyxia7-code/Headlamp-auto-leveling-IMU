# Resilient sensor-input layer

This document specifies the input layer in
`src/headlamp_leveling/fusion/input_guard.py` and the matching resilience
features of the EKF (`fusion/ekf.py`). Together they decide, for every sample of
every sensor, whether the estimator may use it, and what the system does when it
may not.

The design goal is **graceful degradation**: a disturbed or failed input should
cost as little leveling performance as physics allows, the lamp should enter its
fail-safe state only when the load pitch is no longer observable, and normal
operation should resume by itself when the input is healthy again.

---

## 1. Division of work

| Layer | Knows | Catches |
|---|---|---|
| **Input guard** (100 Hz, per sample) | sensor ranges, noise behaviour, simple physics that holds for any car, load and road | missing data, saturation, frozen signals, spikes, grossly implausible values |
| **EKF residual monitor** (50 Hz) | the vehicle model and the current state estimate | faults small enough to look plausible on their own, such as a biased camera |
| **EKF robust update** | innovation statistics | single outliers that pass the guard (Huber weighting, χ² gate) |
| **Leveling controller** | estimate uncertainty, fail-safe requests | holds or parks the lamp |

The guard is deliberately **model-free**. It cannot be fooled by a wrong state
estimate, and a diverging filter cannot talk it into rejecting good data. The EKF
monitor is the complement: it sees faults the guard cannot, but only as long as
the filter itself is healthy.

The guard **never invents data**. It does not hold, interpolate or extrapolate.
A rejected sample is passed on as "not available" (NaN), and the EKF skips that
measurement update. In a Kalman filter this is the statistically correct
treatment of a missing measurement: the covariance grows through the process
noise, and the remaining sensors carry the estimate.

## 2. Channels

| Channel | Role | Loss of the channel |
|---|---|---|
| `ax`, `az` | accelerometer, longitudinal and vertical | **critical**: the load pitch is not observable |
| `v_wheel` | vehicle speed from wheel speeds | **critical**: acceleration and standstill unknown |
| `gy` | pitch gyro | degraded mode: the EKF keeps its accuracy on accelerometer + wheel speed |
| `ay`, `gz` | lateral acceleration, yaw rate | degraded: roll compensation holds its last value |
| `gx` | roll gyro | monitored only |
| `camera` | pitch to the road plane | auxiliary: absence is normal operation |

The pitch gyro is not critical. On six test trips, removing it after 120 s
changed the load-pitch MAE by at most ±0.03° per trip. It mainly speeds up the
tracking of grade changes.

## 3. Per-sample pipeline

Each raw sample runs through these checks in order. The first one that fails
rejects the sample.

1. **Validity.** The value must be finite. NaN stands for a receive timeout, an
   invalid flag or a failed checksum.
2. **Range.** The value must lie inside the sensor's full scale minus a margin
   (±19 m/s², ±245 °/s, 0–90 m/s, camera ±10°). A value at the limit is a
   saturated sensor, not a measurement.
3. **Stuck.** A live MEMS signal always carries noise above its quantisation
   step, so a value that repeats exactly for 0.3 s is a frozen sensor or a frozen
   bus message. A wheel speed of exactly zero is exempt, because standstill is a
   legitimate constant. For the camera the test counts identical frames (5).
4. **Spike.**
   - *IMU channels:* a causal Hampel test. The sample is compared with the
     median of the last 11 accepted samples. It is rejected if it deviates by
     more than `6 × 1.4826 × MAD`, and by at least 2.5 m/s² or 25 °/s. The floor
     keeps ordinary road shocks from being rejected. If four samples in a row
     fail the test, the signal has really changed level: the window is cleared
     and the new level is accepted.
   - *Wheel speed:* the change since the last accepted sample must be physically
     possible, at most 15 m/s² times the elapsed time plus 0.3 m/s. The allowance
     grows with time, so the signal is accepted again after a real gap.

## 4. Cross-sensor plausibility

These rules use only physics that holds whatever the load, road and driving
style. Each rule has its own debounce: it must be violated for 2 s before it
disqualifies its channels.

| Rule | Condition (violated when) | Disqualifies | Applies |
|---|---|---|---|
| Gravity norm | `\|‖f‖ − g\| > 0.8 m/s²` | `ax`, `az` | standstill, 1.5 s after stopping |
| No rotation at rest | `\|gyro\| > 3 °/s` on an axis | that gyro axis | standstill |
| Moving at zero speed | `ax` changed by > 1 m/s² since stopping, or `az` sample-to-sample noise > 0.08 m/s² | `v_wheel` | wheel speed reads 0 (1 s qualify) |
| Vertical level | `\|lowpass(az) − g\| > 1.2 m/s²` | `az` | driving faster than 3 m/s |
| No sustained pitch or roll rate | `\|lowpass(gyro)\| > 5 °/s` | `gy`, `gx` | driving |
| Accelerometer vs. wheel speed | `\|lowpass(ax) − d(v_wheel)/dt\| > 3.5 m/s²` | `ax`, `v_wheel` | driving |

Notes on the thresholds:

- **Vertical level.** A 14 % grade changes `az` by only 0.1 m/s², and the
  accelerometer bias adds at most about 0.45 m/s². The 1.2 m/s² limit is far
  outside both.
- **Accelerometer vs. wheel speed.** The legitimate difference is
  `g·sin(grade + pitch)`, at most about 2.5 m/s² on a 25 % ramp. The rule cannot
  tell which of the two sensors is wrong, so it disqualifies both.
- **Moving at zero speed.** A wheel-speed signal that drops to zero while the car
  drives would otherwise be the most damaging fault of all: the filter would
  apply zero-velocity updates and read every tilt change as a load change. The
  vibration level is measured only from 1 s after the stop, so the road vibration
  from before the stop does not trigger it. The 0.08 m/s² limit must be checked
  against engine idle vibration on the target vehicle.

A rule that currently does not apply (a standstill rule while driving) keeps a
qualified fault. A fault can only heal while its rule is being checked and found
satisfied.

## 5. Health states and debounce

Each channel has one of three states:

- **OK.** Samples are used.
- **DEGRADED.** Individual samples are being rejected. The EKF skips them; the
  fault is not qualified.
- **FAILED.** Rejections added up to 0.5 s (`qualify_s`), or a plausibility rule
  disqualified the channel. No sample of the channel reaches the EKF.

The debounce is time-based. Bad time builds up at full rate and decays four
times slower during good samples, so an intermittent fault (every second sample
bad) still qualifies. A FAILED channel heals after 2 s of continuous good
samples (`heal_s`). The wheel speed needs 4 s, because the "moving at zero
speed" rule takes about 3.5 s to object to a zero that is wrong.

The camera's health is judged per received frame, because frames are sparse and
their absence is normal.

## 6. System reaction

| Situation | Reaction |
|---|---|
| Single samples rejected | EKF skips those updates. No visible effect. |
| Short gap in all IMU data (< 0.5 s) | EKF predicts through the gap. When data returns, the fast states (squat and dive, grade rate) are re-opened so the correction lands on them and not on the load pitch. |
| Gyro, lateral channel or camera FAILED | Degraded mode. Leveling continues. |
| Critical channel FAILED for less than 3 s | EKF coasts. The controller keeps the last correction. |
| Critical channel FAILED for 3 s or longer (`failsafe_after_s`) | Fail-safe request: the beam is never raised. The lamp holds the correction it had at that moment, or moves down to `failsafe_deg` if that is lower. |
| Critical channel healed | The EKF is told how long the outage lasted and re-opens the affected states in proportion. The fail-safe request ends, and leveling resumes once the estimate's σ is back under the controller's limit. |
| EKF residual monitor trips for 3 s without the guard finding a cause | Latched fail-safe for the rest of the drive cycle. An unexplained model inconsistency is not trusted to heal. |

**Why "hold, never raise".** A fail-safe that drives the lamp back to the unladen
position would raise the beam of a loaded car and glare oncoming traffic. UN R48
is understood to require the same: after a failure of the leveling device, the
dipped beam must not assume a position with less dip than at the time of the
failure. Verify this against the current text of the regulation.

The guard's fail-safe request does **not** latch. The EKF monitor's does. This
is the difference between a known, cleared cause and an unknown one.

## 7. Resilience inside the EKF

- **Missing inputs.** Any input of `PitchEKF.step` may be NaN. Each measurement
  update runs only if its input is present. Without a wheel speed the filter
  treats standstill as unknown: it applies no zero-velocity updates and suspends
  the load-change logic.
- **Robust update.** IMU and wheel-speed updates are Huber M-estimates. An
  innovation beyond 3σ keeps its direction, but its gain is scaled by
  `3 / |ν/√S|`. A moderate outlier nudges the state; it does not pull it.
  Innovations with NIS above 30 are still rejected outright.
- **Timing.** A step longer than 50 ms is predicted in sub-steps, so a late
  sample does not destabilise the Euler-discretised pitch dynamics.
- **Gaps.**
  - After a wheel-speed gap, the speed variance grows by
    `(2 m/s² × gap)²` before the first new sample is used.
  - After an IMU gap, the squat/dive and grade-rate states are re-opened.
- **Gate lockout.** If every sample of a sensor has been gated out for 1 s, the
  filter may have drifted away from a healthy sensor, and it would then reject
  that sensor forever. The states the sensor observes are re-opened so the filter
  can re-acquire. The residual monitor keeps running, so a real fault still
  shows.
- **Residual monitor.**
  - It has hysteresis: a fault sets above a mean NIS of 9 and clears below 3.
  - A critical sensor that delivers nothing relaxes its monitor towards
    "consistent".
  - A faulted auxiliary input (camera, network) is still compared with the
    estimate but not used, so it can heal when it agrees again.
  - The camera monitor uses only the pitch-state uncertainty. The full innovation
    variance also contains the grade-rate uncertainty times `d/v`, which let a
    constant camera offset pass as "grade change ahead" at low speed.
- **Isolation hand-over.** When the guard disqualifies a channel, the EKF
  re-opens the states that channel's last, already faulty samples acted on, and
  restarts the residual monitors. Without this, the monitor kept reporting the
  removed fault and latched the fail-safe.
- **Numerical safety net.** After every cycle the state and covariance are
  checked for non-finite values and non-positive variances. On a violation the
  filter restarts from the last good load-pitch estimate with σ = 0.5°, so the
  controller holds the lamp until the filter has re-converged.

## 8. Fault injection and results

`sim/faults.py` injects faults into the full-rate signals of a simulated trip.
`hal robust-eval` runs every scenario twice on the same trips:

- *unguarded*: missing samples are replaced by the last received value, and
  nothing else is done. The EKF's own gating, Huber weighting and monitor are
  active in both variants.
- *guarded*: samples pass the input layer first.

```bash
hal robust-eval -n 12 -o runs/robustness
```

<!-- robustness results -->

## 9. Embedded notes

- **Cost.** Per sample and channel: one sort of 11 values (twice), a handful of
  comparisons and three timers. At 100 Hz and 8 channels this is negligible next
  to the EKF. RAM: an 11-sample window per IMU channel plus about 20 scalars per
  channel.
- **Timeouts.** In the vehicle, "missing" comes from the bus layer: an alive
  counter that does not advance, a failed CRC, a signal-invalid flag or a receive
  timeout. Map all of them to "no sample this cycle".
- **Thresholds to calibrate per vehicle:**
  - `still_vibration_max_ms2` against engine idle vibration, including diesel
    and start-stop;
  - `stuck_s` against the real bus period (a 50 Hz CAN signal sampled at 100 Hz
    repeats once by design);
  - the spike floors against the worst rough-road data.
- **Diagnostics.** `SensorInputGuard.summary()` returns rejected-sample counts
  per channel and reason, and how often each plausibility rule qualified. These
  map directly to diagnostic trouble codes and their freeze-frame data.

## 10. Known limitations

- A fault that is small and slow enough to look like physics is not detected by
  the guard. Examples: a 0.3 m/s² accelerometer offset, or a gyro bias drift.
  The EKF absorbs these in its bias states by design.
- A longitudinal accelerometer offset is indistinguishable from a road grade. It
  corrupts the grade estimate, which is only an auxiliary output, and leaves the
  load pitch alone.
- The plausibility rules need the right driving state. A vertical-accelerometer
  offset that appears and disappears within one stop is only checked by the
  gravity-norm rule.
- All results are from simulation. The fault models are simple (steps, freezes,
  dropouts, white glitches). Real intermittent contacts and EMC bursts should be
  replayed from recorded data.

# autoware_gtsam_fusion_localizer

A GTSAM iSAM2 factor-graph fusion localizer for Autoware, designed as a drop-in replacement
for the `autoware_ekf_localizer` package in `autoware_core`. It fuses **IMU**, **vehicle
twist**, **NDT pose**, and **dual-antenna RTK GNSS pose** into a globally consistent
map-frame pose+twist estimate.

The output topic namespace matches the autoware_core EKF
(`/localization/pose_twist_fusion_filter/*`) so downstream consumers (planner, controller,
`pose_instability_detector`, `localization_error_monitor`) require no changes to swap from
EKF to this node.

---

## Table of contents

1. [Why a factor graph?](#why-a-factor-graph)
2. [System overview](#system-overview)
3. [Data flow](#data-flow)
4. [Topic interface](#topic-interface)
5. [TF requirements](#tf-requirements)
6. [Factors](#factors)
7. [Three phases of capability](#three-phases-of-capability)
8. [Build](#build)
9. [Run](#run)
10. [Parameter reference](#parameter-reference)
11. [Verification & troubleshooting](#verification--troubleshooting)
12. [Tuning guide](#tuning-guide)
13. [Diagnostics](#diagnostics)
14. [Code layout](#code-layout)
15. [Threading model](#threading-model)
16. [Known limitations](#known-limitations)
17. [References](#references)

---

## Why a factor graph?

The EKF is a *sequential* state estimator: it carries forward a single state plus covariance,
applies each measurement once, and forgets it. This is fast but has three structural
weaknesses for autonomous-driving localization:

1. **No revisiting the past.** A GNSS outlier that pulls the state off-track cannot be
   undone by later, contradictory measurements.
2. **No outlier tolerance beyond covariance.** The EKF trusts the noise model literally; a
   3-sigma outlier degrades the state in proportion to its (under-stated) covariance.
3. **No joint calibration.** Sensor calibration constants (GNSS antenna lever arm,
   wheel-odometry scale, IMU bias) are usually estimated offline or treated as fixed.

A factor graph jointly optimizes a sliding window of states. The iSAM2 algorithm (Kaess et
al.) re-computes only the affected portion of the graph, so this remains real-time. The
graph framework also makes it natural to:

- promote any unknown into an estimable variable (lever arm, twist scale, IMU bias);
- attach a robust kernel (Huber, Cauchy, DCS) to any factor to suppress outliers;
- add measurements from new sensors as new factor types without rewriting the estimator.

This package implements all three benefits. See the *Three phases* section below for which
parameters enable each.

---

## System overview

```
┌───────────────────────────────────────────────────────────────┐
│  ROS 2 subscriptions                                          │
│   /imu/data            sensor_msgs/Imu                        │
│   .../twist_w_cov      geometry_msgs/TwistWithCovarianceStamp │
│   .../ndt_pose_w_cov   geometry_msgs/PoseWithCovarianceStamp  │
│   .../gnss_pose_w_cov  geometry_msgs/PoseWithCovarianceStamp  │
│   .../gnss_fix         sensor_msgs/NavSatFix                  │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│  SensorBuffer (thread-safe queues)                            │
│  - IMU deque (high rate)                                      │
│  - twist deque                                                │
│  - NDT deque                                                  │
│  - GNSS pose <-> NavSatFix stamp-matcher                      │
│  Wakes worker thread on each NDT or GNSS arrival.             │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│  Worker thread (drains buffer, builds factors)                │
│  - ImuPreintegrator: integrates IMU between keyframes         │
│  - Builds KeyframeInputs {imu_preint, ndt, gnss, twist}       │
│  - Calls FactorGraphManager::add_keyframe()                   │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│  FactorGraphManager (iSAM2 or IncrementalFixedLagSmoother)    │
│  - Adds CombinedImuFactor between X(i),V(i),B(i) -> ...j      │
│  - Adds PriorFactor<Pose3>      (NDT)                         │
│  - Adds PriorFactor<Pose3>      (GNSS)                        │
│  - Adds PriorFactor<Vector3>    (twist linear velocity)       │
│  - Optional: GnssLeverArmFactor, ScaledTwistFactor,           │
│              SwitchablePose3PriorFactor                       │
│  - Computes marginal covariance for X(j), V(j)                │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│  Publisher timer (50 Hz)                                      │
│  - Reads latest NavState + bias                               │
│  - Reorders covariance GTSAM [rot;trans] -> ROS [trans;rot]   │
│  - Publishes pose / twist / kinematic_state / TF              │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│  Output topics (match autoware_core EKF)                      │
│   /localization/pose_twist_fusion_filter/pose                 │
│   /localization/pose_twist_fusion_filter/pose_with_covariance │
│   .../biased_pose, .../biased_pose_with_covariance            │
│   .../twist, .../twist_with_covariance                        │
│   /localization/kinematic_state                               │
│   /diagnostics                                                │
│   TF: map -> base_link                                        │
└───────────────────────────────────────────────────────────────┘
```

---

## Data flow

### Keyframes

The node only creates new graph variables at *keyframes*, not at every IMU sample. A new
keyframe is created when any of these is true:

- An NDT pose arrives (debounced by `keyframe.min_dt_sec`).
- A GNSS pose with a matching NavSatFix arrives (same debounce).
- `keyframe.max_dt_sec` has elapsed since the last keyframe even without aiding (forces a
  keyframe so IMU integration error stays bounded).

At each keyframe `j` the node adds three new variables:

- `X(j)` — `gtsam::Pose3` in the `map` frame
- `V(j)` — `gtsam::Vector3` velocity in the `map` frame
- `B(j)` — `gtsam::imuBias::ConstantBias` (accel + gyro biases, modeled as a slow random walk)

It also adds one or more *factors* connecting `X(j), V(j), B(j)` to the previous keyframe
`X(i), V(i), B(i)` and to the new measurements.

### IMU between keyframes

Between keyframes, every IMU sample is preintegrated via `gtsam::PreintegratedCombinedMeasurements`.
At the next keyframe a single `gtsam::CombinedImuFactor(X(i), V(i), X(j), V(j), B(i), B(j))` is
added. Preintegration is computed only once and then cheaply re-linearized when biases are
updated, so the cost is independent of how many IMU samples were integrated.

IMU samples are pre-rotated from `imu_link` into `base_link` at the input stage using the
static TF looked up once at startup. The preintegrator therefore sees body-frame
measurements and the `body_P_sensor` parameter is identity.

---

## Topic interface

### Subscriptions

| Remap key            | Default topic                                                                | Type                                       |
|----------------------|------------------------------------------------------------------------------|--------------------------------------------|
| `~/input/imu`        | `/sensing/imu/imu_data`                                                      | `sensor_msgs/Imu`                          |
| `~/input/twist`      | `/localization/twist_estimator/twist_with_covariance`                        | `geometry_msgs/TwistWithCovarianceStamped` |
| `~/input/ndt_pose`   | `/localization/pose_estimator/pose_with_covariance`                          | `geometry_msgs/PoseWithCovarianceStamped`  |
| `~/input/gnss_pose`  | `/sensing/gnss/pose_with_covariance`                                         | `geometry_msgs/PoseWithCovarianceStamped`  |
| `~/input/gnss_fix`   | `/sensing/gnss/fix`                                                          | `sensor_msgs/NavSatFix`                    |

The GNSS pose and NavSatFix are matched by stamp inside the SensorBuffer with a tolerance of
`keyframe.gnss_fix_match_tolerance_sec` (default 50 ms). Pose messages that never receive a
matching fix within ~1 s are dropped. The NavSatFix `status.status` field is the
`is_fix` source — only values listed in `gnss.accept_fix_statuses` are admitted (defaults to
`[0, 1, 2]` = `STATUS_FIX`, `STATUS_SBAS_FIX`, `STATUS_GBAS_FIX`).

### Publishers (match the autoware_core EKF output namespace)

| Remap key                                | Default topic                                                                 | Type                                       |
|------------------------------------------|-------------------------------------------------------------------------------|--------------------------------------------|
| `~/output/pose`                          | `/localization/pose_twist_fusion_filter/pose`                                 | `geometry_msgs/PoseStamped`                |
| `~/output/pose_with_covariance`          | `/localization/pose_twist_fusion_filter/pose_with_covariance`                 | `geometry_msgs/PoseWithCovarianceStamped`  |
| `~/output/biased_pose`                   | `/localization/pose_twist_fusion_filter/biased_pose`                          | `geometry_msgs/PoseStamped`                |
| `~/output/biased_pose_with_covariance`   | `/localization/pose_twist_fusion_filter/biased_pose_with_covariance`          | `geometry_msgs/PoseWithCovarianceStamped`  |
| `~/output/twist`                         | `/localization/pose_twist_fusion_filter/twist`                                | `geometry_msgs/TwistStamped`               |
| `~/output/twist_with_covariance`         | `/localization/pose_twist_fusion_filter/twist_with_covariance`                | `geometry_msgs/TwistWithCovarianceStamped` |
| `~/output/kinematic_state`               | `/localization/kinematic_state`                                               | `nav_msgs/Odometry`                        |
| (fixed)                                  | `/diagnostics`                                                                | `diagnostic_msgs/DiagnosticArray`          |
| TF broadcaster                           | `map -> base_link`                                                            | `tf2_msgs/TFMessage`                       |

`biased_pose` is published identical to `pose` (see *Known limitations* for why).

### Covariance ordering

`PoseWithCovarianceStamped.pose.covariance` is a 6×6 in **ROS** order `[tx ty tz rx ry rz]`.
GTSAM internally uses the **opposite** order `[rx ry rz tx ty tz]` on the Pose3 tangent
space. The node applies a 6×6 block-swap permutation `P · C · Pᵀ` in both directions; a
unit test (`test/test_ros_gtsam_conversions.cpp`) enforces correctness in both directions
with a non-symmetric covariance block.

---

## TF requirements

The node requires the static transform `base_link <- imu_link` (or whatever frames are
configured under `frames.*`) to be available on `/tf_static` at startup. It is looked up
once (with retry up to `imu.static_tf_lookup_timeout_sec`) and the rotation is applied to
every IMU sample before integration.

If the lookup times out, the node falls back to identity and logs a `WARN`; if the imu
frame is actually rotated relative to base_link, the resulting state estimate will be
biased.

The translation component of the IMU mount offset is treated as small enough to ignore at
typical vehicle speeds (this is also what the autoware_core EKF assumes). If your IMU mount
is far from `base_link`, model the lever arm explicitly outside this node.

---

## Factors

### CombinedImuFactor

Reference: Forster et al., *On-Manifold Preintegration for Real-Time Visual-Inertial Odometry*
(2017). Encodes the relative motion between two keyframes (`ΔR, Δv, Δp`) plus a random-walk
prior on the biases. The factor connects `X(i), V(i), X(j), V(j), B(i), B(j)`.

Tuned via `imu.*` parameters — accelerometer & gyroscope continuous-time noise densities,
bias random-walk standard deviations, and an integration covariance for unmodeled errors.

### PriorFactor<Pose3> — NDT and GNSS

The simplest absolute-pose factor: residual is `Logmap(measured^-1 * X)`. The 6×6 noise
model is constructed from the incoming covariance (after the ROS→GTSAM block reorder).

In Phase B, this factor is wrapped in a `Robust::Create(Huber, base)` noise model. The
Huber kernel switches from quadratic to linear penalty above its threshold `k` (default
1.345 standard deviations) so a single outlier observation can no longer dominate the
optimization.

### PriorFactor<Vector3> — twist linear velocity

Residual is `V(j) - v_measured`. Uses the 3×3 linear-velocity block of the incoming
TwistWithCovariance. Angular velocity is not used because the IMU preintegration already
constrains it via the gyro.

### GnssLeverArmFactor (Phase C)

`NoiseModelFactor2<Pose3, Point3>` with residual

> `e = T_world_base.transformFrom(lever_arm) - z_gnss`

The lever-arm `L(0)` is a single Point3 variable shared across all GNSS observations;
a loose `PriorFactor<Point3>` (default σ = 5 cm) seeds it from the URDF value, and the
optimizer refines it from data. Analytic Jacobians from `Pose3::transformFrom` are used.

Note that the lever arm is only fully observable during yawing motion — a vehicle driving
in a straight line cannot resolve all three components. A typical operational policy is to
perform an explicit calibration drive with turns and then freeze the estimate.

### ScaledTwistFactor (Phase C)

`NoiseModelFactor2<Vector3, double>` with residual

> `e = z_twist - s * V(j)`

A single scale variable `S(0)` (default prior 1.0 ± 0.05) absorbs slow scale errors in the
wheel odometry — for example after a passenger weight change alters tire rolling radius.

### SwitchablePose3PriorFactor (Phase C)

DCS-style switchable constraint: residual is `switch * Logmap(measured^-1 * X)` with a
per-observation `switch` variable in [0, 1]. A unit prior on `switch` keeps it near 1; the
optimizer is free to drive it toward 0 if a measurement is an outlier. Enable by setting
`robust.mode: switchable`.

Reference: Sünderhauf & Protzel, *Switchable Constraints for Robust Pose Graph SLAM* (IROS 2012).

---

## Three phases of capability

All three phases are implemented; switch between them via parameters at launch. No
recompilation needed.

### Phase A — Baseline

- IMU preintegration
- NDT and GNSS as `PriorFactor<Pose3>`
- Twist linear velocity as `PriorFactor<Vector3>`
- iSAM2 incremental optimization
- Simple translation gate on incoming aiding factors (`keyframe.ndt_position_gate_m`,
  `keyframe.gnss_position_gate_m`)

To run Phase A only: set `robust.mode: none`, `gnss.use_lever_arm_factor: false`,
`twist.use_scale_variable: false`, `isam2.use_fixed_lag_smoother: false`. (These are the
defaults.)

### Phase B — Robustness

- `robust.mode: huber` (or `cauchy`) wraps NDT/GNSS in a robust kernel
- Mahalanobis-distance prefilter before factor admission; threshold from `robust.mahalanobis_chi2_p`
- `max_consecutive_rejects` admits persistent disagreement with inflated cov to recover
- NDT covariance eigen-inflation along degenerate axes
  (`ndt.degeneracy_condition_threshold`, `ndt.degeneracy_inflation_factor`)
- Diagnostics publishing of chi², biases, divergence

### Phase C — Advanced

- `gnss.use_lever_arm_factor: true` enables `GnssLeverArmFactor`
- `twist.use_scale_variable: true` enables `ScaledTwistFactor`
- `robust.mode: switchable` enables `SwitchablePose3PriorFactor`
- `isam2.use_fixed_lag_smoother: true` swaps iSAM2 for `IncrementalFixedLagSmoother`
  (bounded compute via marginalization beyond `isam2.smoother_lag_sec`)

---

## Build

### Dependencies

System-level (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install libgtsam-dev libeigen3-dev
```

If `libgtsam-dev` is not resolved by `rosdep` in your environment, add a local override
file under `/etc/ros/rosdep/sources.list.d/` mapping the key to the package name:

```yaml
# /etc/ros/rosdep/sources.list.d/30-gtsam.yaml
libgtsam-dev:
  ubuntu: [libgtsam-dev]
  debian: [libgtsam-dev]
```

Then `rosdep update`.

GTSAM version compatibility:

| GTSAM version | Status                                                                                              |
|---------------|-----------------------------------------------------------------------------------------------------|
| 4.2.x         | **Recommended.** Uses `NoiseModelFactorN` and `OptionalMatrixType = Matrix*`.                       |
| 4.1.x         | Should work; identical API surface for the features used here.                                      |
| 4.0.x         | The custom factors use the 4.2 `NoiseModelFactorN` API; adapt to `NoiseModelFactor2`/3 if needed.  |

### Workspace build

```bash
cd <your-autoware-ws>
colcon build --packages-select autoware_gtsam_fusion_localizer --symlink-install
source install/setup.bash
```

The package links against both `gtsam` and `gtsam_unstable` (the latter for
`IncrementalFixedLagSmoother` used in Phase C).

---

## Run

### Standalone launch

```bash
ros2 launch autoware_gtsam_fusion_localizer gtsam_fusion_localizer.launch.xml
```

### With custom input topics

```bash
ros2 launch autoware_gtsam_fusion_localizer gtsam_fusion_localizer.launch.xml \
  input_imu_topic:=/my/imu \
  input_ndt_pose_topic:=/my/ndt \
  input_gnss_pose_topic:=/my/gnss \
  input_gnss_fix_topic:=/my/gnss/fix \
  input_twist_topic:=/my/twist
```

### Swapping in for the autoware_core EKF

In your localization launch graph, replace the autoware_core EKF include with the equivalent
include of this package's launch file. Because the output topics are identical, no other
launch file needs to change.

---

## Parameter reference

All parameters live under `/**/ros__parameters/<group>` and have a JSON schema at
`schema/gtsam_fusion_localizer.schema.json`.

### `frames`

| Key         | Type   | Default      | Meaning                                      |
|-------------|--------|--------------|----------------------------------------------|
| `map`       | string | `map`        | World frame for output pose                  |
| `base_link` | string | `base_link`  | Body frame; output TF child                  |
| `imu_link`  | string | `imu_link`   | IMU sensor frame for static-TF rotation      |

### `imu`

| Key                              | Type   | Default | Meaning                                                  |
|----------------------------------|--------|---------|----------------------------------------------------------|
| `gravity`                        | number | 9.80665 | m/s², used by `PreintegrationCombinedParams::MakeSharedU`|
| `accel_noise_sigma`              | number | 0.05    | m/s²/√Hz, continuous-time accel noise density            |
| `gyro_noise_sigma`               | number | 0.005   | rad/s/√Hz, continuous-time gyro noise density            |
| `accel_bias_rw_sigma`            | number | 1e-4    | m/s³/√Hz, accelerometer bias random walk                  |
| `gyro_bias_rw_sigma`             | number | 1e-5    | rad/s²/√Hz, gyroscope bias random walk                   |
| `integration_cov`                | number | 1e-8    | Variance added per IMU step for unmodeled errors          |
| `expected_rate_hz`               | number | 100.0   | Used to clamp pathological dt after stalls                |
| `static_tf_lookup_timeout_sec`   | number | 5.0     | How long to retry `base_link<-imu_link` lookup at start   |

### `keyframe`

| Key                              | Type   | Default        | Meaning                                                   |
|----------------------------------|--------|----------------|-----------------------------------------------------------|
| `trigger`                        | string | `ndt_or_gnss`  | Either trigger creates a keyframe                          |
| `min_dt_sec`                     | number | 0.05           | Debounce — don't add a keyframe more often than this      |
| `max_dt_sec`                     | number | 1.0            | Force a keyframe even without aiding after this elapsed   |
| `ndt_position_gate_m`            | number | 5.0            | Reject NDT priors >this from predicted pose               |
| `gnss_position_gate_m`           | number | 5.0            | Reject GNSS priors >this from predicted pose              |
| `gnss_fix_match_tolerance_sec`   | number | 0.05           | Stamp-matching tolerance between pose and NavSatFix       |

### `init`

| Key                              | Type   | Default          | Meaning                                                  |
|----------------------------------|--------|------------------|----------------------------------------------------------|
| `source`                         | string | `ndt_then_gnss`  | Preferred initialization source                          |
| `wait_timeout_sec`               | number | 30.0             | How long to wait for the first fix                       |
| `initial_velocity_sigma`         | number | 0.5              | Sigma of the initial-velocity prior                      |
| `initial_bias_accel_sigma`       | number | 0.1              | Sigma of the initial accel-bias prior                    |
| `initial_bias_gyro_sigma`        | number | 0.01             | Sigma of the initial gyro-bias prior                     |

### `twist`

| Key                  | Type    | Default | Meaning                                              |
|----------------------|---------|---------|------------------------------------------------------|
| `enable`             | boolean | true    | If false, no velocity factor is added                |
| `use_scale_variable` | boolean | false   | Phase C — adds ScaledTwistFactor with scale S(0)     |
| `scale_prior_mean`   | number  | 1.0     | Prior mean for the scale variable                    |
| `scale_prior_sigma`  | number  | 0.05    | Prior σ; tight enough to anchor but permits drift    |

### `gnss`

| Key                      | Type     | Default          | Meaning                                                       |
|--------------------------|----------|------------------|---------------------------------------------------------------|
| `max_position_sigma`     | number   | 0.5              | Drop GNSS pose if sqrt(σx²+σy²+σz²) exceeds √3 × this         |
| `accept_fix_statuses`    | int list | [0, 1, 2]        | NavSatFix.status values to accept                              |
| `use_lever_arm_factor`   | boolean  | false            | Phase C — use GnssLeverArmFactor instead of PriorFactor       |
| `lever_arm_xyz`          | num list | [1.2, 0.0, 1.5]  | Initial lever-arm (m) in base_link                            |
| `lever_arm_prior_sigma`  | number   | 0.05             | Prior σ on lever arm; loose enough to absorb cal error        |

### `robust` (Phase B)

| Key                       | Type    | Default | Meaning                                                  |
|---------------------------|---------|---------|----------------------------------------------------------|
| `mode`                    | string  | `none`  | `none` \| `huber` \| `cauchy` \| `switchable`            |
| `huber_k`                 | number  | 1.345   | Huber/Cauchy threshold in sigmas                          |
| `mahalanobis_chi2_p`      | number  | 0.99    | Confidence for the Mahalanobis prefilter (χ²₆)            |
| `max_consecutive_rejects` | int     | 5       | After this many rejects, admit with inflated cov          |

### `ndt` (Phase B)

| Key                                | Type   | Default | Meaning                                                  |
|------------------------------------|--------|---------|----------------------------------------------------------|
| `degeneracy_condition_threshold`   | number | 100.0   | Inflate all eigenvalues if max/min ratio exceeds this     |
| `degeneracy_threshold`             | number | 0.5     | Inflate an axis whose eigenvalue exceeds this             |
| `degeneracy_inflation_factor`      | number | 100.0   | Multiplier applied to flagged eigenvalues                 |

### `isam2`

| Key                          | Type    | Default    | Meaning                                                  |
|------------------------------|---------|------------|----------------------------------------------------------|
| `relinearize_threshold`      | number  | 0.01       | iSAM2 relinearization threshold                          |
| `relinearize_skip`           | int     | 1          | Number of updates between relinearization checks         |
| `cache_linearized_factors`   | boolean | true       | iSAM2 perf tunable                                       |
| `factorization`              | string  | `CHOLESKY` | `CHOLESKY` or `QR`                                       |
| `use_fixed_lag_smoother`     | boolean | false      | Phase C — use IncrementalFixedLagSmoother                |
| `smoother_lag_sec`           | number  | 30.0       | Smoother window size                                     |

### `output`

| Key                 | Type    | Default | Meaning                                          |
|---------------------|---------|---------|--------------------------------------------------|
| `publish_tf`        | boolean | true    | Broadcast `map -> base_link`                     |
| `publish_rate_hz`   | number  | 50.0    | Rate of the publisher timer                      |

### `diagnostics`

| Key                              | Type    | Default | Meaning                                          |
|----------------------------------|---------|---------|--------------------------------------------------|
| `enable`                         | boolean | true    | Publish `/diagnostics`                           |
| `rate_hz`                        | number  | 1.0     | Rate of `/diagnostics` publication               |
| `gnss_ndt_divergence_warn_m`     | number  | 0.5     | WARN threshold on GNSS-NDT divergence            |
| `gnss_ndt_divergence_error_m`    | number  | 2.0     | ERROR threshold on GNSS-NDT divergence           |

---

## Verification & troubleshooting

### Build verification

```bash
colcon build --packages-select autoware_gtsam_fusion_localizer --symlink-install
colcon test --packages-select autoware_gtsam_fusion_localizer
colcon test-result --verbose
```

The test suite covers:

- `test_ros_gtsam_conversions` — covariance reorder round-trip in both directions, pose
  message ↔ Pose3 round-trip, array ↔ matrix round-trip.
- `test_factor_graph_manager` — initialize-then-keyframe smoke test with stationary IMU and
  NDT prior at origin, verifying the state converges to zero pose and zero velocity.
- `test_factors` — error-zero-at-truth and analytic-vs-numerical Jacobian checks for
  `GnssLeverArmFactor`, `ScaledTwistFactor`, and `SwitchablePose3PriorFactor`.

### Runtime smoke test (rosbag)

```bash
ros2 launch autoware_gtsam_fusion_localizer gtsam_fusion_localizer.launch.xml &
ros2 bag play your_bag.db3 --clock
```

In a separate terminal:

```bash
ros2 topic hz /localization/pose_twist_fusion_filter/pose_with_covariance
# expect ~50 Hz (output.publish_rate_hz)

ros2 topic echo --once /localization/pose_twist_fusion_filter/pose_with_covariance
# inspect covariance ordering: [xx yy zz rr pp yy] across diagonal

ros2 run tf2_ros tf2_echo map base_link
# expect a stable transform after first fix

ros2 topic echo /diagnostics
# initialized -> OK after first fix; WARN before with "waiting for first NDT or GNSS fix"
```

### A/B against autoware_core EKF

Record both estimators' `pose_with_covariance` on the same bag with different remaps:

```bash
ros2 bag record \
  /localization/pose_twist_fusion_filter/pose_with_covariance:=/gtsam/pose \
  /localization/pose_twist_fusion_filter/pose_with_covariance:=/ekf/pose
```

Plot them in RViz or PlotJuggler. Expected behavior: comparable absolute pose in nominal
conditions, smoother trajectory during NDT dropouts and stationary periods.

### Common issues

**Q: The node never publishes a pose.**
A: Check `/diagnostics`. If it says "waiting for first NDT or GNSS fix", verify the input
topics are flowing. `ros2 topic hz <input>` to each subscription.

**Q: `dpkg: package libgtsam-dev not installed`.**
A: Install it: `sudo apt install libgtsam-dev`. On distros without a package, build GTSAM
from source and set `GTSAM_DIR` in your environment.

**Q: `linker error: undefined reference to gtsam::IncrementalFixedLagSmoother::...`**
A: Make sure `gtsam_unstable` is linked. The CMakeLists.txt does this; if you're using a
custom GTSAM build, ensure both `libgtsam` and `libgtsam_unstable` are built and installed.

**Q: Output pose covariance looks reversed (translation variances in rotation slots).**
A: The covariance reorder is broken. Run the unit test (`test_ros_gtsam_conversions`) —
if it fails, file a bug.

**Q: Pose drifts noticeably while parked.**
A: Either (1) twist is not subscribed, (2) twist covariance is too large, or (3) IMU bias
estimate is wrong. Echo `/diagnostics` to inspect `bias_accel_*` and `bias_gyro_*`. If
biases drift while parked, lower `imu.{accel,gyro}_bias_rw_sigma`.

**Q: After a GNSS multipath event, pose jumps and recovers slowly.**
A: Enable Phase B: set `robust.mode: huber`. This caps the influence of any single
observation. Also consider lowering `gnss.max_position_sigma` to reject obvious bad fixes
upstream.

**Q: Pose locks to NDT inside a tunnel, even though NDT should be unreliable there.**
A: Verify NDT is actually emitting larger covariance along the tunnel axis. If not, fix
NDT first (enable Hessian-based covariance estimation). Then enable
`ndt.degeneracy_inflation_factor` in Phase B.

---

## Tuning guide

### IMU noise densities

The defaults are conservative; consult your IMU datasheet. Typical values:

- Industrial-grade MEMS (Xsens MTi-10): accel ≈ 0.06 m/s²/√Hz, gyro ≈ 0.01 rad/s/√Hz
- Tactical-grade FOG (KVH 1750): accel ≈ 0.03, gyro ≈ 0.0002
- Consumer MEMS (BMI088): accel ≈ 0.2, gyro ≈ 0.05

Bias random walks are usually 1–2 orders of magnitude smaller than the white-noise
densities; start with `accel_bias_rw_sigma = 1e-4` and `gyro_bias_rw_sigma = 1e-5` and
adjust upward if the estimator under-tracks slow bias changes.

### Keyframe rate

Default `min_dt_sec = 0.05` gives a 20 Hz upper bound on the keyframe rate, which roughly
matches typical NDT/GNSS arrival rates. Lowering it (e.g. 0.02) creates more keyframes at
higher iSAM2 cost; raising it can cause numerical issues in CombinedImuFactor over very
long preintegration windows. Stick within [0.02, 0.2].

### Robust kernel

`huber` is conservative — outliers are linearly penalized but not zeroed. `cauchy` is more
aggressive (penalty actually decreases beyond the threshold). `switchable` does the
strictest job but adds an extra variable per observation, which costs compute. Start with
`huber` and only escalate if outliers persist.

### Mahalanobis threshold

`robust.mahalanobis_chi2_p = 0.99` means 99% of well-modeled observations should pass. If
you see legitimate observations being rejected, lower to 0.95. If too many bad fixes leak
through, raise to 0.999.

---

## Diagnostics

The `/diagnostics` topic carries a single `DiagnosticStatus` named `gtsam_fusion_localizer`
with the following key-value pairs:

| Key                        | Meaning                                              |
|----------------------------|------------------------------------------------------|
| `graph_size`               | Number of keyframes currently in the graph           |
| `isam_update_sec`          | Wall-clock time of the last iSAM2 update             |
| `chi2_imu`                 | Sum of CombinedImuFactor errors in last batch        |
| `chi2_ndt`                 | Sum of NDT-related pose-prior errors in last batch   |
| `chi2_gnss`                | Sum of GNSS-related pose-prior errors in last batch  |
| `chi2_twist`               | Sum of velocity-prior errors in last batch           |
| `bias_accel_{x,y,z}`       | Current estimated accelerometer bias (m/s²)          |
| `bias_gyro_{x,y,z}`        | Current estimated gyroscope bias (rad/s)             |
| `gnss_ndt_divergence_m`    | Latest \|p_ndt - p_gnss\| in meters                  |
| `lever_arm_{x,y,z}`        | (Phase C) Current lever-arm estimate (m)             |
| `twist_scale`              | (Phase C) Current twist scale estimate               |

Status level rules:

- `OK` — initialized and within divergence thresholds
- `WARN` — uninitialized, or GNSS-NDT divergence above `diagnostics.gnss_ndt_divergence_warn_m`
- `ERROR` — GNSS-NDT divergence above `diagnostics.gnss_ndt_divergence_error_m`

---

## Code layout

```
autoware_gtsam_fusion_localizer/
├── CMakeLists.txt
├── package.xml
├── README.md                     # this file
├── config/
│   └── gtsam_fusion_localizer.param.yaml
├── launch/
│   └── gtsam_fusion_localizer.launch.xml
├── schema/
│   └── gtsam_fusion_localizer.schema.json
├── include/autoware/gtsam_fusion_localizer/
│   ├── parameters.hpp                       # POD parameter struct
│   ├── ros_gtsam_conversions.hpp            # Pose & covariance ↔ GTSAM
│   ├── sensor_buffer.hpp                    # thread-safe per-sensor queues
│   ├── imu_preintegrator.hpp                # CombinedImuFactor accumulation
│   ├── factor_graph_manager.hpp             # iSAM2 / FLS owner
│   ├── diagnostics_publisher.hpp
│   ├── gtsam_fusion_localizer_node.hpp
│   └── factors/
│       ├── gnss_lever_arm_factor.hpp        # Phase C
│       ├── scaled_twist_factor.hpp          # Phase C
│       └── switchable_constraint_factor.hpp # Phase C
├── src/
│   ├── ros_gtsam_conversions.cpp
│   ├── sensor_buffer.cpp
│   ├── imu_preintegrator.cpp
│   ├── factor_graph_manager.cpp
│   ├── diagnostics_publisher.cpp
│   ├── gtsam_fusion_localizer_node.cpp
│   └── factors/
│       ├── gnss_lever_arm_factor.cpp
│       ├── scaled_twist_factor.cpp
│       └── switchable_constraint_factor.cpp
└── test/
    ├── test_ros_gtsam_conversions.cpp
    ├── test_factor_graph_manager.cpp
    └── test_factors.cpp
```

---

## Threading model

- **Executor thread(s)** — Run subscription callbacks. Each callback is short: validate
  message, push into `SensorBuffer`, notify the condition variable, return.
- **Worker thread** — Owned by the node. Blocks on `SensorBuffer::wait_for_keyframe()`.
  On wake-up: drains IMU into the preintegrator, builds a `KeyframeInputs`, calls
  `FactorGraphManager::add_keyframe()` (which internally runs iSAM2), then resets the
  preintegrator with the new bias.
- **Publisher timer thread** — Runs at `output.publish_rate_hz` (default 50 Hz). Reads
  the latest `NavState` under a short mutex, applies the covariance reorder, and
  publishes pose/twist/TF.

The worker thread holds no lock during the iSAM2 update itself. The publisher takes a
small lock only to read the latest state. iSAM2 is single-threaded internally — there is
no contention.

`SensorBuffer` is a thin layer over four deques guarded by a single `std::mutex` plus a
`std::condition_variable`. Subscription callbacks never block on the optimizer.

---

## Known limitations

- **`biased_pose` is published equal to `pose`.** The autoware_core EKF distinguishes
  biased vs unbiased pose to support a separate yaw-bias state. In this model, the gyro
  bias inside `imuBias::ConstantBias` absorbs that role, so the two outputs are identical.
  Kept in the interface for downstream-consumer compatibility.

- **Lever-arm Phase C disabled by default.** When `gnss.use_lever_arm_factor: false`
  (the default), the node trusts the GNSS driver to have applied the antenna→base_link
  offset upstream. If your GNSS driver does not, enable the lever-arm factor.

- **IMU translation offset is not modeled.** Only the rotation of `base_link <- imu_link`
  is consumed. If your IMU is far from base_link, you'll see a small lever-arm error
  proportional to angular acceleration.

- **First-keyframe IMU history is discarded.** Any IMU samples that arrive before the
  first NDT/GNSS fix are integrated against bias=0 with a wrong attitude. After
  initialization the preintegrator is reset, so this only affects the latency of the
  first keyframe.

- **No outlier reasoning across sensors.** If both GNSS and NDT happen to be biased in
  the same direction (e.g. both reference a wrong map origin), the graph cannot detect
  the disagreement. The diagnostics divergence metric helps surface this case.

---

## References

1. Forster, Carlone, Dellaert, Scaramuzza. *On-Manifold Preintegration for Real-Time
   Visual-Inertial Odometry*. T-RO 2017.
2. Kaess, Johannsson, Roberts, Ila, Leonard, Dellaert. *iSAM2: Incremental Smoothing
   and Mapping using the Bayes Tree*. IJRR 2012.
3. Sünderhauf, Protzel. *Switchable Constraints for Robust Pose Graph SLAM*. IROS 2012.
4. Dellaert, Kaess. *Factor Graphs for Robot Perception*. Foundations and Trends in
   Robotics, 2017.
5. GTSAM library: https://gtsam.org/
6. Autoware EKF localizer (autoware_core): the package this node replaces.

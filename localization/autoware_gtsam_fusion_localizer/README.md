# autoware_gtsam_fusion_localizer

A GTSAM iSAM2 factor-graph fusion localizer designed as a drop-in replacement for the
`autoware_ekf_localizer` package in autoware_core. Fuses IMU, vehicle twist, NDT pose, and
dual-antenna RTK GNSS pose into a globally consistent map-frame pose+twist estimate.

## Architecture

The node maintains a sliding factor graph of vehicle states (`Pose3`, `Vector3` velocity,
`imuBias::ConstantBias`) at "keyframe" timestamps triggered by NDT or GNSS arrivals. Between
keyframes, IMU samples are preintegrated into a `CombinedImuFactor`. NDT, GNSS, and twist
contribute prior-style factors on pose and velocity. The graph is solved incrementally with
iSAM2; the latest `NavState` is forward-propagated to publish at 50 Hz.

See `/root/.claude/plans/please-create-a-new-replicated-aho.md` (or the upstream design
document) for the full chapter-by-chapter spec.

## Topic interface

### Subscriptions

| Remap key            | Default                                                                | Type                                       |
| -------------------- | ---------------------------------------------------------------------- | ------------------------------------------ |
| `~/input/imu`        | `/sensing/imu/imu_data`                                                | `sensor_msgs/Imu`                          |
| `~/input/twist`      | `/localization/twist_estimator/twist_with_covariance`                  | `geometry_msgs/TwistWithCovarianceStamped` |
| `~/input/ndt_pose`   | `/localization/pose_estimator/pose_with_covariance`                    | `geometry_msgs/PoseWithCovarianceStamped`  |
| `~/input/gnss_pose`  | `/sensing/gnss/pose_with_covariance`                                   | `geometry_msgs/PoseWithCovarianceStamped`  |
| `~/input/gnss_fix`   | `/sensing/gnss/fix`                                                    | `sensor_msgs/NavSatFix`                    |

### Publishers (match the `autoware_core` EKF output namespace)

| Remap key                                | Default                                                                 |
| ---------------------------------------- | ----------------------------------------------------------------------- |
| `~/output/pose`                          | `/localization/pose_twist_fusion_filter/pose`                           |
| `~/output/pose_with_covariance`          | `/localization/pose_twist_fusion_filter/pose_with_covariance`           |
| `~/output/biased_pose`                   | `/localization/pose_twist_fusion_filter/biased_pose`                    |
| `~/output/biased_pose_with_covariance`   | `/localization/pose_twist_fusion_filter/biased_pose_with_covariance`    |
| `~/output/twist`                         | `/localization/pose_twist_fusion_filter/twist`                          |
| `~/output/twist_with_covariance`         | `/localization/pose_twist_fusion_filter/twist_with_covariance`          |
| `~/output/kinematic_state`               | `/localization/kinematic_state`                                         |

Plus a `map -> base_link` TF broadcast and `/diagnostics`.

## Build

GTSAM must be installed at the system level:

```bash
sudo apt-get install libgtsam-dev
```

If `libgtsam-dev` is not resolved by `rosdep`, add a local override under
`/etc/ros/rosdep/sources.list.d/`.

```bash
colcon build --packages-select autoware_gtsam_fusion_localizer --symlink-install
```

## Run

```bash
ros2 launch autoware_gtsam_fusion_localizer gtsam_fusion_localizer.launch.xml
```

## Notes on `biased_pose`

The autoware_core EKF distinguishes biased vs unbiased pose for yaw-bias estimation. In this
factor-graph model there is no explicit yaw-bias variable - the gyro bias inside
`imuBias::ConstantBias` plays the equivalent role. `biased_pose` is therefore published equal
to `pose` for interface compatibility.

## Phases

- **Phase A** - Minimal viable: IMU preintegration + NDT/GNSS/twist priors + iSAM2.
- **Phase B** - Robustness: Huber kernels, Mahalanobis prefilter, NDT degenerate-axis
  covariance inflation, diagnostics.
- **Phase C** - Advanced: lever-arm-aware GNSS factor, twist scale variable, fixed-lag
  smoothing, switchable constraints / DCS.

All three phases are implemented in this package and selectable via parameters.

// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#ifndef AUTOWARE__GTSAM_FUSION_LOCALIZER__FACTOR_GRAPH_MANAGER_HPP_
#define AUTOWARE__GTSAM_FUSION_LOCALIZER__FACTOR_GRAPH_MANAGER_HPP_

#include "autoware/gtsam_fusion_localizer/parameters.hpp"

#include <Eigen/Core>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <rclcpp/time.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <variant>

namespace autoware::gtsam_fusion_localizer
{

struct KeyframeInputs
{
  rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
  std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> imu_preint;

  // ROS-ordered 6x6 covariances are reordered inside the manager before noise-model construction.
  std::optional<std::pair<gtsam::Pose3, Eigen::Matrix<double, 6, 6>>> ndt_prior_ros_cov;
  std::optional<std::pair<gtsam::Pose3, Eigen::Matrix<double, 6, 6>>> gnss_prior_ros_cov;
  std::optional<std::pair<gtsam::Vector3, Eigen::Matrix3d>> twist_prior;
};

struct ChiSquaredReport
{
  double imu{0.0};
  double ndt{0.0};
  double gnss{0.0};
  double twist{0.0};
  int ndt_count{0};
  int gnss_count{0};
};

class FactorGraphManager
{
public:
  explicit FactorGraphManager(const Parameters & params);

  // Add initialization variables and prior factors. Returns false if already initialized.
  bool initialize(
    const rclcpp::Time & stamp, const gtsam::Pose3 & initial_pose,
    const gtsam::Vector3 & initial_velocity, const gtsam::imuBias::ConstantBias & initial_bias);

  // Append a new keyframe (X, V, B) connected to the previous keyframe via the IMU
  // preintegration factor, plus aiding priors as available.
  // Returns true on success, false if not yet initialized or if dropped by gating.
  bool add_keyframe(const KeyframeInputs & inputs);

  // Force-run the optimizer on staged factors/values.
  void update();

  bool initialized() const { return initialized_; }
  std::uint64_t key_index() const { return key_idx_; }

  gtsam::NavState latest_state() const;
  gtsam::imuBias::ConstantBias latest_bias() const;
  Eigen::Matrix<double, 6, 6> latest_pose_cov_gtsam_order() const;
  Eigen::Matrix3d latest_velocity_cov() const;
  rclcpp::Time latest_stamp() const { return latest_stamp_; }
  std::optional<gtsam::Point3> latest_lever_arm() const;
  std::optional<double> latest_twist_scale() const;

  const ChiSquaredReport & last_chi2() const { return last_chi2_; }
  double last_update_duration_sec() const { return last_update_duration_sec_; }

  // Used by Mahalanobis prefilter: predicted Pose3 covariance after IMU preintegration.
  Eigen::Matrix<double, 6, 6> predicted_pose_cov(
    const gtsam::PreintegratedCombinedMeasurements & preint) const;

private:
  void apply_isam2_params();
  bool gate_ndt(
    const gtsam::Pose3 & predicted, const gtsam::Pose3 & measured,
    const Eigen::Matrix<double, 6, 6> & combined_cov_gtsam);
  bool gate_gnss(
    const gtsam::Pose3 & predicted, const gtsam::Pose3 & measured,
    const Eigen::Matrix<double, 6, 6> & combined_cov_gtsam);
  Eigen::Matrix<double, 6, 6> inflate_ndt_cov(const Eigen::Matrix<double, 6, 6> & ros_cov) const;
  gtsam::SharedNoiseModel wrap_robust(const gtsam::SharedNoiseModel & base) const;

  Parameters params_;
  std::shared_ptr<gtsam::ISAM2> isam_;
  std::shared_ptr<gtsam::IncrementalFixedLagSmoother> smoother_;
  gtsam::NonlinearFactorGraph new_factors_;
  gtsam::Values new_values_;
  std::map<gtsam::Key, double> new_timestamps_;
  std::uint64_t key_idx_{0};
  bool initialized_{false};

  rclcpp::Time latest_stamp_{0, 0, RCL_ROS_TIME};
  gtsam::NavState latest_state_;
  gtsam::imuBias::ConstantBias latest_bias_;
  Eigen::Matrix<double, 6, 6> latest_pose_cov_gtsam_{Eigen::Matrix<double, 6, 6>::Identity()};
  Eigen::Matrix3d latest_velocity_cov_{Eigen::Matrix3d::Identity()};
  std::optional<gtsam::Point3> latest_lever_arm_;
  std::optional<double> latest_twist_scale_;

  // Mahalanobis prefilter state.
  int consecutive_ndt_rejects_{0};
  int consecutive_gnss_rejects_{0};

  ChiSquaredReport last_chi2_{};
  double last_update_duration_sec_{0.0};

  // Phase C key for the (single) GNSS lever-arm variable and twist scale.
  static constexpr std::uint64_t kLeverArmKeyIdx = 0;
  static constexpr std::uint64_t kTwistScaleKeyIdx = 0;
  static constexpr std::uint64_t kSwitchKeyBase = 0;
};

}  // namespace autoware::gtsam_fusion_localizer

#endif  // AUTOWARE__GTSAM_FUSION_LOCALIZER__FACTOR_GRAPH_MANAGER_HPP_

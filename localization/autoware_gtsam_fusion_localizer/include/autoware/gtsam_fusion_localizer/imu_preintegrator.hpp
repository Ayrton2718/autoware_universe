// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#ifndef AUTOWARE__GTSAM_FUSION_LOCALIZER__IMU_PREINTEGRATOR_HPP_
#define AUTOWARE__GTSAM_FUSION_LOCALIZER__IMU_PREINTEGRATOR_HPP_

#include "autoware/gtsam_fusion_localizer/parameters.hpp"

#include <Eigen/Geometry>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <memory>

namespace autoware::gtsam_fusion_localizer
{

// Wraps gtsam::PreintegratedCombinedMeasurements. IMU samples are pre-rotated into base_link
// before integration (the static base_link <- imu_link rotation is provided at startup).
class ImuPreintegrator
{
public:
  ImuPreintegrator(const ImuParams & params, const Eigen::Quaterniond & R_base_imu);

  void reset(const gtsam::imuBias::ConstantBias & bias);

  // Integrate a sample. dt is computed from the timestamp delta against the previous sample;
  // the first call after reset() seeds the previous timestamp without integrating.
  void integrate(const sensor_msgs::msg::Imu & msg);

  // Returns the accumulated preintegration object (caller takes a shared copy).
  std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> take();

  double accumulated_dt() const { return accumulated_dt_; }
  bool empty() const { return accumulated_dt_ <= 0.0; }

  const gtsam::PreintegrationCombinedParams & params() const { return *params_; }

private:
  ImuParams imu_params_;
  Eigen::Quaterniond R_base_imu_;
  std::shared_ptr<gtsam::PreintegrationCombinedParams> params_;
  std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> preint_;
  bool have_prev_stamp_{false};
  rclcpp::Time prev_stamp_{0, 0, RCL_ROS_TIME};
  double accumulated_dt_{0.0};
};

}  // namespace autoware::gtsam_fusion_localizer

#endif  // AUTOWARE__GTSAM_FUSION_LOCALIZER__IMU_PREINTEGRATOR_HPP_

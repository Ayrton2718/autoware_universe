// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "autoware/gtsam_fusion_localizer/imu_preintegrator.hpp"

#include <gtsam/base/Matrix.h>

namespace autoware::gtsam_fusion_localizer
{

namespace
{
std::shared_ptr<gtsam::PreintegrationCombinedParams> make_params(const ImuParams & p)
{
  auto params = gtsam::PreintegrationCombinedParams::MakeSharedU(p.gravity);
  const double a = p.accel_noise_sigma * p.accel_noise_sigma;
  const double g = p.gyro_noise_sigma * p.gyro_noise_sigma;
  const double ba = p.accel_bias_rw_sigma * p.accel_bias_rw_sigma;
  const double bg = p.gyro_bias_rw_sigma * p.gyro_bias_rw_sigma;
  params->setAccelerometerCovariance(gtsam::I_3x3 * a);
  params->setGyroscopeCovariance(gtsam::I_3x3 * g);
  params->setIntegrationCovariance(gtsam::I_3x3 * p.integration_cov);
  params->setBiasAccCovariance(gtsam::I_3x3 * ba);
  params->setBiasOmegaCovariance(gtsam::I_3x3 * bg);
  params->setBiasAccOmegaInit(gtsam::I_6x6 * 1.0e-5);
  return params;
}
}  // namespace

ImuPreintegrator::ImuPreintegrator(const ImuParams & params, const Eigen::Quaterniond & R_base_imu)
: imu_params_(params), R_base_imu_(R_base_imu.normalized()), params_(make_params(params))
{
  preint_ =
    std::make_shared<gtsam::PreintegratedCombinedMeasurements>(params_, gtsam::imuBias::ConstantBias());
}

void ImuPreintegrator::reset(const gtsam::imuBias::ConstantBias & bias)
{
  preint_ = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(params_, bias);
  have_prev_stamp_ = false;
  accumulated_dt_ = 0.0;
}

void ImuPreintegrator::integrate(const sensor_msgs::msg::Imu & msg)
{
  const rclcpp::Time t(msg.header.stamp, RCL_ROS_TIME);
  if (!have_prev_stamp_) {
    prev_stamp_ = t;
    have_prev_stamp_ = true;
    return;
  }
  double dt = (t - prev_stamp_).seconds();
  if (dt <= 0.0) {
    // Drop out-of-order or duplicate samples.
    return;
  }
  // Clamp pathological dt (e.g. after long stalls) to a reasonable upper bound.
  const double max_dt = 5.0 / std::max(1.0, imu_params_.expected_rate_hz);
  if (dt > max_dt) {
    dt = max_dt;
  }
  prev_stamp_ = t;

  const Eigen::Vector3d acc_imu(
    msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z);
  const Eigen::Vector3d gyro_imu(
    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z);
  const Eigen::Vector3d acc = R_base_imu_ * acc_imu;
  const Eigen::Vector3d gyro = R_base_imu_ * gyro_imu;

  preint_->integrateMeasurement(acc, gyro, dt);
  accumulated_dt_ += dt;
}

std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> ImuPreintegrator::take()
{
  return preint_;
}

}  // namespace autoware::gtsam_fusion_localizer

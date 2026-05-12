// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "autoware/gtsam_fusion_localizer/factor_graph_manager.hpp"
#include "autoware/gtsam_fusion_localizer/imu_preintegrator.hpp"

#include <gtest/gtest.h>

namespace ag = autoware::gtsam_fusion_localizer;

namespace
{
ag::Parameters make_params()
{
  ag::Parameters p;
  p.isam2.relinearize_threshold = 0.01;
  p.isam2.relinearize_skip = 1;
  p.imu.gravity = 9.80665;
  return p;
}

sensor_msgs::msg::Imu make_imu(double t, double az = 9.80665)
{
  sensor_msgs::msg::Imu m;
  rclcpp::Time stamp(static_cast<int64_t>(t * 1.0e9), RCL_ROS_TIME);
  m.header.stamp = stamp;
  m.linear_acceleration.x = 0.0;
  m.linear_acceleration.y = 0.0;
  m.linear_acceleration.z = az;  // counter-gravity = stationary
  m.angular_velocity.x = 0.0;
  m.angular_velocity.y = 0.0;
  m.angular_velocity.z = 0.0;
  return m;
}
}  // namespace

TEST(FactorGraphManager, InitializesAndAddsKeyframe)
{
  const auto params = make_params();
  ag::FactorGraphManager graph(params);

  const rclcpp::Time t0(static_cast<int64_t>(0 * 1.0e9), RCL_ROS_TIME);
  ASSERT_TRUE(graph.initialize(
    t0, gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0, 0, 0)), gtsam::Vector3::Zero(),
    gtsam::imuBias::ConstantBias()));
  EXPECT_TRUE(graph.initialized());

  // Preintegrate 100 samples at 100 Hz of stationary IMU.
  ag::ImuPreintegrator preint(params.imu, Eigen::Quaterniond::Identity());
  preint.reset(graph.latest_bias());
  for (int i = 0; i <= 100; ++i) {
    preint.integrate(make_imu(i * 0.01));
  }
  ag::KeyframeInputs kf;
  kf.stamp = rclcpp::Time(static_cast<int64_t>(1.0 * 1.0e9), RCL_ROS_TIME);
  kf.imu_preint = preint.take();

  // Aiding NDT at the origin (stationary).
  Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
  kf.ndt_prior_ros_cov = std::make_pair(
    gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0, 0, 0)), cov);

  ASSERT_TRUE(graph.add_keyframe(kf));
  const auto state = graph.latest_state();
  EXPECT_NEAR(state.pose().x(), 0.0, 0.05);
  EXPECT_NEAR(state.pose().y(), 0.0, 0.05);
  EXPECT_NEAR(state.pose().z(), 0.0, 0.05);
  EXPECT_NEAR(state.velocity().norm(), 0.0, 0.05);
}

TEST(FactorGraphManager, RejectsAddBeforeInit)
{
  const auto params = make_params();
  ag::FactorGraphManager graph(params);
  ag::KeyframeInputs kf;
  kf.stamp = rclcpp::Time(0, 0, RCL_ROS_TIME);
  EXPECT_FALSE(graph.add_keyframe(kf));
}

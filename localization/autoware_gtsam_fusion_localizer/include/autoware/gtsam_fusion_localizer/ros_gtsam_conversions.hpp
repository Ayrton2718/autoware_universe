// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#ifndef AUTOWARE__GTSAM_FUSION_LOCALIZER__ROS_GTSAM_CONVERSIONS_HPP_
#define AUTOWARE__GTSAM_FUSION_LOCALIZER__ROS_GTSAM_CONVERSIONS_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>

#include <array>

namespace autoware::gtsam_fusion_localizer::conversions
{

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using RowMajorMatrix6 = std::array<double, 36>;

// GTSAM 6-DOF tangent ordering is [rot(3); trans(3)].
// ROS PoseWithCovariance ordering is [trans(3); rot(3)].
// reorder_cov_gtsam_to_ros() and reorder_cov_ros_to_gtsam() apply the appropriate
// 6x6 permutation: P * C * P^T (with P self-inverse for this swap).
Matrix6d reorder_cov_gtsam_to_ros(const Matrix6d & gtsam_cov);
Matrix6d reorder_cov_ros_to_gtsam(const Matrix6d & ros_cov);

// ROS row-major covariance (size 36) <-> Eigen 6x6.
Matrix6d cov_array_to_matrix(const RowMajorMatrix6 & arr);
RowMajorMatrix6 matrix_to_cov_array(const Matrix6d & mat);

// Pose conversions.
gtsam::Pose3 pose_msg_to_gtsam(const geometry_msgs::msg::Pose & msg);
geometry_msgs::msg::Pose pose_gtsam_to_msg(const gtsam::Pose3 & pose);

// Build a Gaussian noise model from a ROS-ordered 6x6 covariance (reorders internally).
gtsam::SharedNoiseModel pose_noise_from_ros_cov(const Matrix6d & ros_cov);

}  // namespace autoware::gtsam_fusion_localizer::conversions

#endif  // AUTOWARE__GTSAM_FUSION_LOCALIZER__ROS_GTSAM_CONVERSIONS_HPP_

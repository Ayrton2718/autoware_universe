// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "autoware/gtsam_fusion_localizer/ros_gtsam_conversions.hpp"

#include <gtsam/geometry/Rot3.h>

namespace autoware::gtsam_fusion_localizer::conversions
{

namespace
{
Matrix6d block_swap_permutation()
{
  // Maps [trans(3); rot(3)] <-> [rot(3); trans(3)].
  Matrix6d p = Matrix6d::Zero();
  // ROS index 0..2 (trans) -> GTSAM index 3..5
  p(3, 0) = 1.0;
  p(4, 1) = 1.0;
  p(5, 2) = 1.0;
  // ROS index 3..5 (rot) -> GTSAM index 0..2
  p(0, 3) = 1.0;
  p(1, 4) = 1.0;
  p(2, 5) = 1.0;
  return p;
}
}  // namespace

Matrix6d reorder_cov_ros_to_gtsam(const Matrix6d & ros_cov)
{
  const Matrix6d p = block_swap_permutation();
  return p * ros_cov * p.transpose();
}

Matrix6d reorder_cov_gtsam_to_ros(const Matrix6d & gtsam_cov)
{
  // The permutation is its own inverse (swaps two 3-blocks), so the same matrix works.
  const Matrix6d p = block_swap_permutation();
  return p * gtsam_cov * p.transpose();
}

Matrix6d cov_array_to_matrix(const RowMajorMatrix6 & arr)
{
  Matrix6d m;
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      m(r, c) = arr[r * 6 + c];
    }
  }
  return m;
}

RowMajorMatrix6 matrix_to_cov_array(const Matrix6d & mat)
{
  RowMajorMatrix6 out{};
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      out[r * 6 + c] = mat(r, c);
    }
  }
  return out;
}

gtsam::Pose3 pose_msg_to_gtsam(const geometry_msgs::msg::Pose & msg)
{
  const gtsam::Rot3 rot = gtsam::Rot3::Quaternion(
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z);
  const gtsam::Point3 t(msg.position.x, msg.position.y, msg.position.z);
  return gtsam::Pose3(rot, t);
}

geometry_msgs::msg::Pose pose_gtsam_to_msg(const gtsam::Pose3 & pose)
{
  geometry_msgs::msg::Pose msg;
  msg.position.x = pose.x();
  msg.position.y = pose.y();
  msg.position.z = pose.z();
  const auto q = pose.rotation().toQuaternion();
  msg.orientation.w = q.w();
  msg.orientation.x = q.x();
  msg.orientation.y = q.y();
  msg.orientation.z = q.z();
  return msg;
}

gtsam::SharedNoiseModel pose_noise_from_ros_cov(const Matrix6d & ros_cov)
{
  const Matrix6d gtsam_cov = reorder_cov_ros_to_gtsam(ros_cov);
  return gtsam::noiseModel::Gaussian::Covariance(gtsam_cov);
}

}  // namespace autoware::gtsam_fusion_localizer::conversions

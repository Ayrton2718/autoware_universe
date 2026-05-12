// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "autoware/gtsam_fusion_localizer/ros_gtsam_conversions.hpp"

#include <gtest/gtest.h>

namespace ag = autoware::gtsam_fusion_localizer;
using ag::conversions::Matrix6d;

TEST(RosGtsamConversions, ReorderRoundTrip)
{
  // Build a non-symmetric structured 6x6 with clearly different rotation and translation blocks.
  Matrix6d ros = Matrix6d::Zero();
  // ROS ordering is [tx ty tz | rx ry rz]
  ros(0, 0) = 1.0; ros(1, 1) = 4.0; ros(2, 2) = 9.0;       // trans variances
  ros(3, 3) = 16.0; ros(4, 4) = 25.0; ros(5, 5) = 36.0;     // rot variances
  ros(0, 3) = 0.5; ros(3, 0) = 0.5;                          // trans-rot cross
  ros(1, 4) = -0.25; ros(4, 1) = -0.25;

  Matrix6d gtsam_order = ag::conversions::reorder_cov_ros_to_gtsam(ros);

  // GTSAM ordering is [rx ry rz | tx ty tz]
  EXPECT_DOUBLE_EQ(gtsam_order(0, 0), 16.0);
  EXPECT_DOUBLE_EQ(gtsam_order(1, 1), 25.0);
  EXPECT_DOUBLE_EQ(gtsam_order(2, 2), 36.0);
  EXPECT_DOUBLE_EQ(gtsam_order(3, 3), 1.0);
  EXPECT_DOUBLE_EQ(gtsam_order(4, 4), 4.0);
  EXPECT_DOUBLE_EQ(gtsam_order(5, 5), 9.0);
  // Off-diagonal trans-rot should appear in (rot, trans) block (rows 0-2, cols 3-5).
  EXPECT_DOUBLE_EQ(gtsam_order(0, 3), 0.5);
  EXPECT_DOUBLE_EQ(gtsam_order(3, 0), 0.5);

  Matrix6d back = ag::conversions::reorder_cov_gtsam_to_ros(gtsam_order);
  EXPECT_TRUE(back.isApprox(ros, 1.0e-12));
}

TEST(RosGtsamConversions, ArrayMatrixRoundTrip)
{
  Matrix6d m;
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      m(r, c) = static_cast<double>(r * 6 + c);
    }
  }
  const auto arr = ag::conversions::matrix_to_cov_array(m);
  const auto m2 = ag::conversions::cov_array_to_matrix(arr);
  EXPECT_TRUE(m.isApprox(m2, 1.0e-12));
}

TEST(RosGtsamConversions, PoseRoundTrip)
{
  geometry_msgs::msg::Pose p;
  p.position.x = 1.0; p.position.y = 2.0; p.position.z = 3.0;
  p.orientation.w = std::cos(0.5 * 0.7);
  p.orientation.x = 0.0;
  p.orientation.y = 0.0;
  p.orientation.z = std::sin(0.5 * 0.7);
  const auto g = ag::conversions::pose_msg_to_gtsam(p);
  const auto p2 = ag::conversions::pose_gtsam_to_msg(g);
  EXPECT_NEAR(p.position.x, p2.position.x, 1.0e-12);
  EXPECT_NEAR(p.position.y, p2.position.y, 1.0e-12);
  EXPECT_NEAR(p.position.z, p2.position.z, 1.0e-12);
  EXPECT_NEAR(p.orientation.w, p2.orientation.w, 1.0e-9);
  EXPECT_NEAR(p.orientation.z, p2.orientation.z, 1.0e-9);
}

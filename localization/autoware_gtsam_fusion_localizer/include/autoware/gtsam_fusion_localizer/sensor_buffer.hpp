// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#ifndef AUTOWARE__GTSAM_FUSION_LOCALIZER__SENSOR_BUFFER_HPP_
#define AUTOWARE__GTSAM_FUSION_LOCALIZER__SENSOR_BUFFER_HPP_

#include "autoware/gtsam_fusion_localizer/parameters.hpp"

#include <rclcpp/time.hpp>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

namespace autoware::gtsam_fusion_localizer
{

struct GnssObservation
{
  geometry_msgs::msg::PoseWithCovarianceStamped pose;
  sensor_msgs::msg::NavSatFix fix;
};

// Reason the worker thread woke up.
enum class KeyframeTrigger
{
  None,
  Ndt,
  Gnss,
  MaxDt,
  Shutdown
};

struct KeyframeSlice
{
  rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
  KeyframeTrigger trigger{KeyframeTrigger::None};
  std::vector<sensor_msgs::msg::Imu> imu_samples;
  std::optional<geometry_msgs::msg::PoseWithCovarianceStamped> ndt;
  std::optional<GnssObservation> gnss;
  std::optional<geometry_msgs::msg::TwistWithCovarianceStamped> twist;
};

class SensorBuffer
{
public:
  explicit SensorBuffer(const KeyframeParams & params);

  void push_imu(const sensor_msgs::msg::Imu & msg);
  void push_twist(const geometry_msgs::msg::TwistWithCovarianceStamped & msg);
  void push_ndt(const geometry_msgs::msg::PoseWithCovarianceStamped & msg);
  void push_gnss_pose(const geometry_msgs::msg::PoseWithCovarianceStamped & msg);
  void push_gnss_fix(const sensor_msgs::msg::NavSatFix & msg);

  // Block until a keyframe trigger occurs, or shutdown(); returns the slice (may be empty
  // on shutdown).
  KeyframeSlice wait_for_keyframe(const rclcpp::Time & last_keyframe_stamp);

  void shutdown();

  // For the publisher thread: peek the most recent IMU sample (or nullopt if buffer empty).
  std::optional<sensor_msgs::msg::Imu> latest_imu() const;

private:
  void try_match_gnss_locked();
  bool earliest_aiding_stamp_locked(rclcpp::Time & out) const;

  KeyframeParams params_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool shutdown_{false};

  std::deque<sensor_msgs::msg::Imu> imu_;
  std::deque<geometry_msgs::msg::TwistWithCovarianceStamped> twist_;
  std::deque<geometry_msgs::msg::PoseWithCovarianceStamped> ndt_;
  std::deque<geometry_msgs::msg::PoseWithCovarianceStamped> gnss_pose_pending_;
  std::deque<sensor_msgs::msg::NavSatFix> gnss_fix_pending_;
  std::deque<GnssObservation> gnss_matched_;
};

}  // namespace autoware::gtsam_fusion_localizer

#endif  // AUTOWARE__GTSAM_FUSION_LOCALIZER__SENSOR_BUFFER_HPP_

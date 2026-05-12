// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "autoware/gtsam_fusion_localizer/sensor_buffer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace autoware::gtsam_fusion_localizer
{

namespace
{
rclcpp::Time stamp_of(const std_msgs::msg::Header & h)
{
  return rclcpp::Time(h.stamp, RCL_ROS_TIME);
}
}  // namespace

SensorBuffer::SensorBuffer(const KeyframeParams & params) : params_(params) {}

void SensorBuffer::push_imu(const sensor_msgs::msg::Imu & msg)
{
  {
    std::lock_guard<std::mutex> lk(mutex_);
    imu_.push_back(msg);
  }
  cv_.notify_all();
}

void SensorBuffer::push_twist(const geometry_msgs::msg::TwistWithCovarianceStamped & msg)
{
  {
    std::lock_guard<std::mutex> lk(mutex_);
    twist_.push_back(msg);
    // Bound the queue - we only ever need the most recent.
    while (twist_.size() > 32) {
      twist_.pop_front();
    }
  }
  cv_.notify_all();
}

void SensorBuffer::push_ndt(const geometry_msgs::msg::PoseWithCovarianceStamped & msg)
{
  {
    std::lock_guard<std::mutex> lk(mutex_);
    ndt_.push_back(msg);
  }
  cv_.notify_all();
}

void SensorBuffer::push_gnss_pose(const geometry_msgs::msg::PoseWithCovarianceStamped & msg)
{
  {
    std::lock_guard<std::mutex> lk(mutex_);
    gnss_pose_pending_.push_back(msg);
    try_match_gnss_locked();
  }
  cv_.notify_all();
}

void SensorBuffer::push_gnss_fix(const sensor_msgs::msg::NavSatFix & msg)
{
  {
    std::lock_guard<std::mutex> lk(mutex_);
    gnss_fix_pending_.push_back(msg);
    try_match_gnss_locked();
  }
  cv_.notify_all();
}

void SensorBuffer::try_match_gnss_locked()
{
  const double tol = params_.gnss_fix_match_tolerance_sec;
  // Greedy stamp-matching: for each pose, find the nearest fix within tolerance and pop both.
  for (auto pit = gnss_pose_pending_.begin(); pit != gnss_pose_pending_.end();) {
    const rclcpp::Time pose_t = stamp_of(pit->header);
    auto best = gnss_fix_pending_.end();
    double best_dt = tol;
    for (auto fit = gnss_fix_pending_.begin(); fit != gnss_fix_pending_.end(); ++fit) {
      const rclcpp::Time fix_t = stamp_of(fit->header);
      const double dt = std::abs((pose_t - fix_t).seconds());
      if (dt <= best_dt) {
        best = fit;
        best_dt = dt;
      }
    }
    if (best != gnss_fix_pending_.end()) {
      GnssObservation obs;
      obs.pose = *pit;
      obs.fix = *best;
      gnss_matched_.push_back(std::move(obs));
      gnss_fix_pending_.erase(best);
      pit = gnss_pose_pending_.erase(pit);
    } else {
      ++pit;
    }
  }
  // Drop very stale pending entries (older than 1 s) to avoid unbounded growth.
  const auto drop_stale = [&](auto & q) {
    while (q.size() > 1) {
      const rclcpp::Time newest = stamp_of(q.back().header);
      const rclcpp::Time oldest = stamp_of(q.front().header);
      if ((newest - oldest).seconds() > 1.0) {
        q.pop_front();
      } else {
        break;
      }
    }
  };
  drop_stale(gnss_pose_pending_);
  drop_stale(gnss_fix_pending_);
}

bool SensorBuffer::earliest_aiding_stamp_locked(rclcpp::Time & out) const
{
  std::optional<rclcpp::Time> best;
  if (!ndt_.empty()) {
    best = stamp_of(ndt_.front().header);
  }
  if (!gnss_matched_.empty()) {
    const rclcpp::Time g = stamp_of(gnss_matched_.front().pose.header);
    if (!best || g < *best) best = g;
  }
  if (best) {
    out = *best;
    return true;
  }
  return false;
}

KeyframeSlice SensorBuffer::wait_for_keyframe(const rclcpp::Time & last_keyframe_stamp)
{
  std::unique_lock<std::mutex> lk(mutex_);
  cv_.wait(lk, [&] {
    if (shutdown_) return true;
    rclcpp::Time aiding;
    const bool have_aiding = earliest_aiding_stamp_locked(aiding);
    if (have_aiding) {
      const double dt =
        (last_keyframe_stamp.nanoseconds() == 0) ? 1.0 : (aiding - last_keyframe_stamp).seconds();
      if (dt >= params_.min_dt_sec) return true;
    }
    // Force a keyframe based on IMU time-of-day if max_dt_sec exceeded since last keyframe.
    if (!imu_.empty() && last_keyframe_stamp.nanoseconds() != 0) {
      const rclcpp::Time latest_imu_t = stamp_of(imu_.back().header);
      if ((latest_imu_t - last_keyframe_stamp).seconds() >= params_.max_dt_sec) return true;
    }
    return false;
  });

  KeyframeSlice slice;
  if (shutdown_) {
    slice.trigger = KeyframeTrigger::Shutdown;
    return slice;
  }

  // Decide trigger & stamp.
  rclcpp::Time aiding;
  const bool have_aiding = earliest_aiding_stamp_locked(aiding);
  if (have_aiding) {
    const bool ndt_first =
      !ndt_.empty() &&
      (gnss_matched_.empty() || stamp_of(ndt_.front().header) <=
                                  stamp_of(gnss_matched_.front().pose.header));
    if (ndt_first) {
      slice.trigger = KeyframeTrigger::Ndt;
      slice.ndt = ndt_.front();
      ndt_.pop_front();
      slice.stamp = stamp_of(slice.ndt->header);
      // Opportunistically attach a same-keyframe GNSS if it's close.
      if (!gnss_matched_.empty()) {
        const rclcpp::Time gt = stamp_of(gnss_matched_.front().pose.header);
        if (std::abs((gt - slice.stamp).seconds()) < params_.min_dt_sec) {
          slice.gnss = gnss_matched_.front();
          gnss_matched_.pop_front();
        }
      }
    } else {
      slice.trigger = KeyframeTrigger::Gnss;
      slice.gnss = gnss_matched_.front();
      gnss_matched_.pop_front();
      slice.stamp = stamp_of(slice.gnss->pose.header);
    }
  } else {
    // No aiding: max_dt elapsed, take current IMU stamp.
    slice.trigger = KeyframeTrigger::MaxDt;
    slice.stamp = imu_.empty() ? rclcpp::Time(0, 0, RCL_ROS_TIME) : stamp_of(imu_.back().header);
  }

  // Drain IMU up to and including the keyframe stamp.
  while (!imu_.empty() && stamp_of(imu_.front().header) <= slice.stamp) {
    slice.imu_samples.push_back(imu_.front());
    imu_.pop_front();
  }

  // Use the most recent twist no older than the keyframe stamp.
  if (!twist_.empty()) {
    const auto match_it = std::max_element(
      twist_.begin(), twist_.end(), [&](const auto & a, const auto & b) {
        return stamp_of(a.header) < stamp_of(b.header);
      });
    slice.twist = *match_it;
  }

  return slice;
}

void SensorBuffer::shutdown()
{
  {
    std::lock_guard<std::mutex> lk(mutex_);
    shutdown_ = true;
  }
  cv_.notify_all();
}

std::optional<sensor_msgs::msg::Imu> SensorBuffer::latest_imu() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  if (imu_.empty()) return std::nullopt;
  return imu_.back();
}

}  // namespace autoware::gtsam_fusion_localizer

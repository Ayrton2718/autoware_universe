// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#ifndef AUTOWARE__GTSAM_FUSION_LOCALIZER__DIAGNOSTICS_PUBLISHER_HPP_
#define AUTOWARE__GTSAM_FUSION_LOCALIZER__DIAGNOSTICS_PUBLISHER_HPP_

#include "autoware/gtsam_fusion_localizer/parameters.hpp"

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/time.hpp>

#include <mutex>
#include <optional>
#include <string>

namespace autoware::gtsam_fusion_localizer
{

struct DiagnosticsSnapshot
{
  rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
  bool initialized{false};
  std::string init_reason;
  std::uint64_t graph_size{0};
  double last_update_duration_sec{0.0};
  double chi2_imu{0.0};
  double chi2_ndt{0.0};
  double chi2_gnss{0.0};
  double chi2_twist{0.0};
  std::array<double, 3> bias_accel{0.0, 0.0, 0.0};
  std::array<double, 3> bias_gyro{0.0, 0.0, 0.0};
  std::optional<std::array<double, 3>> lever_arm;
  std::optional<double> twist_scale;
  double gnss_ndt_divergence_m{0.0};
  bool has_gnss_ndt_divergence{false};
};

class DiagnosticsPublisher
{
public:
  DiagnosticsPublisher(
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pub,
    const DiagnosticsParams & params);

  void post(const DiagnosticsSnapshot & snap);
  void publish_latest(const rclcpp::Time & now);

private:
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pub_;
  DiagnosticsParams params_;
  std::mutex mutex_;
  std::optional<DiagnosticsSnapshot> latest_;
};

}  // namespace autoware::gtsam_fusion_localizer

#endif  // AUTOWARE__GTSAM_FUSION_LOCALIZER__DIAGNOSTICS_PUBLISHER_HPP_

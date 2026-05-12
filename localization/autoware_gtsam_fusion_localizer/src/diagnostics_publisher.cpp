// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "autoware/gtsam_fusion_localizer/diagnostics_publisher.hpp"

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>

namespace autoware::gtsam_fusion_localizer
{

namespace
{
diagnostic_msgs::msg::KeyValue kv(const std::string & k, const std::string & v)
{
  diagnostic_msgs::msg::KeyValue out;
  out.key = k;
  out.value = v;
  return out;
}
}  // namespace

DiagnosticsPublisher::DiagnosticsPublisher(
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pub,
  const DiagnosticsParams & params)
: pub_(std::move(pub)), params_(params)
{
}

void DiagnosticsPublisher::post(const DiagnosticsSnapshot & snap)
{
  std::lock_guard<std::mutex> lk(mutex_);
  latest_ = snap;
}

void DiagnosticsPublisher::publish_latest(const rclcpp::Time & now)
{
  if (!params_.enable || !pub_) return;
  DiagnosticsSnapshot snap;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!latest_) return;
    snap = *latest_;
  }

  diagnostic_msgs::msg::DiagnosticArray array;
  array.header.stamp = now;

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "gtsam_fusion_localizer";
  status.hardware_id = "";
  if (!snap.initialized) {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
    status.message = snap.init_reason.empty() ? "uninitialized" : snap.init_reason;
  } else if (
    snap.has_gnss_ndt_divergence &&
    snap.gnss_ndt_divergence_m > params_.gnss_ndt_divergence_error_m) {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    status.message = "GNSS/NDT divergence exceeds error threshold";
  } else if (
    snap.has_gnss_ndt_divergence &&
    snap.gnss_ndt_divergence_m > params_.gnss_ndt_divergence_warn_m) {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
    status.message = "GNSS/NDT divergence elevated";
  } else {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
    status.message = "OK";
  }

  status.values.push_back(kv("graph_size", std::to_string(snap.graph_size)));
  status.values.push_back(kv("isam_update_sec", std::to_string(snap.last_update_duration_sec)));
  status.values.push_back(kv("chi2_imu", std::to_string(snap.chi2_imu)));
  status.values.push_back(kv("chi2_ndt", std::to_string(snap.chi2_ndt)));
  status.values.push_back(kv("chi2_gnss", std::to_string(snap.chi2_gnss)));
  status.values.push_back(kv("chi2_twist", std::to_string(snap.chi2_twist)));
  status.values.push_back(kv("bias_accel_x", std::to_string(snap.bias_accel[0])));
  status.values.push_back(kv("bias_accel_y", std::to_string(snap.bias_accel[1])));
  status.values.push_back(kv("bias_accel_z", std::to_string(snap.bias_accel[2])));
  status.values.push_back(kv("bias_gyro_x", std::to_string(snap.bias_gyro[0])));
  status.values.push_back(kv("bias_gyro_y", std::to_string(snap.bias_gyro[1])));
  status.values.push_back(kv("bias_gyro_z", std::to_string(snap.bias_gyro[2])));
  if (snap.has_gnss_ndt_divergence) {
    status.values.push_back(kv("gnss_ndt_divergence_m", std::to_string(snap.gnss_ndt_divergence_m)));
  }
  if (snap.lever_arm) {
    status.values.push_back(kv("lever_arm_x", std::to_string((*snap.lever_arm)[0])));
    status.values.push_back(kv("lever_arm_y", std::to_string((*snap.lever_arm)[1])));
    status.values.push_back(kv("lever_arm_z", std::to_string((*snap.lever_arm)[2])));
  }
  if (snap.twist_scale) {
    status.values.push_back(kv("twist_scale", std::to_string(*snap.twist_scale)));
  }

  array.status.push_back(status);
  pub_->publish(array);
}

}  // namespace autoware::gtsam_fusion_localizer

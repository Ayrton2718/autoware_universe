// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#ifndef AUTOWARE__GTSAM_FUSION_LOCALIZER__GTSAM_FUSION_LOCALIZER_NODE_HPP_
#define AUTOWARE__GTSAM_FUSION_LOCALIZER__GTSAM_FUSION_LOCALIZER_NODE_HPP_

#include "autoware/gtsam_fusion_localizer/diagnostics_publisher.hpp"
#include "autoware/gtsam_fusion_localizer/factor_graph_manager.hpp"
#include "autoware/gtsam_fusion_localizer/imu_preintegrator.hpp"
#include "autoware/gtsam_fusion_localizer/parameters.hpp"
#include "autoware/gtsam_fusion_localizer/sensor_buffer.hpp"

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

namespace autoware::gtsam_fusion_localizer
{

class GtsamFusionLocalizerNode : public rclcpp::Node
{
public:
  explicit GtsamFusionLocalizerNode(const rclcpp::NodeOptions & options);
  ~GtsamFusionLocalizerNode() override;

private:
  void load_parameters();
  bool wait_for_imu_extrinsics(Eigen::Quaterniond & R_base_imu);

  void on_imu(const sensor_msgs::msg::Imu::ConstSharedPtr msg);
  void on_twist(const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr msg);
  void on_ndt(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg);
  void on_gnss_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg);
  void on_gnss_fix(const sensor_msgs::msg::NavSatFix::ConstSharedPtr msg);

  void worker_loop();
  void publish_timer_tick();

  bool gnss_status_accepted(const sensor_msgs::msg::NavSatFix & fix) const;
  bool gnss_sigma_acceptable(const geometry_msgs::msg::PoseWithCovarianceStamped & p) const;

  Parameters params_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::unique_ptr<SensorBuffer> buffer_;
  std::unique_ptr<ImuPreintegrator> preint_;
  std::unique_ptr<FactorGraphManager> graph_;
  std::unique_ptr<DiagnosticsPublisher> diag_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr sub_twist_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_ndt_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_gnss_pose_;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_gnss_fix_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_pose_cov_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_biased_pose_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_biased_pose_cov_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pub_twist_;
  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr pub_twist_cov_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_kinematic_state_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pub_diag_;

  rclcpp::TimerBase::SharedPtr publish_timer_;
  rclcpp::TimerBase::SharedPtr diagnostics_timer_;

  std::thread worker_;
  std::atomic_bool stop_{false};

  mutable std::mutex latest_mutex_;
  rclcpp::Time last_keyframe_stamp_{0, 0, RCL_ROS_TIME};
  std::optional<geometry_msgs::msg::PoseWithCovarianceStamped> last_ndt_for_divergence_;
  std::optional<geometry_msgs::msg::PoseWithCovarianceStamped> last_gnss_for_divergence_;
};

}  // namespace autoware::gtsam_fusion_localizer

#endif  // AUTOWARE__GTSAM_FUSION_LOCALIZER__GTSAM_FUSION_LOCALIZER_NODE_HPP_

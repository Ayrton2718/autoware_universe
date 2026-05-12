// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "autoware/gtsam_fusion_localizer/gtsam_fusion_localizer_node.hpp"

#include "autoware/gtsam_fusion_localizer/ros_gtsam_conversions.hpp"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <algorithm>
#include <chrono>
#include <utility>

namespace autoware::gtsam_fusion_localizer
{

namespace
{
using std::placeholders::_1;
using namespace std::chrono_literals;
}  // namespace

GtsamFusionLocalizerNode::GtsamFusionLocalizerNode(const rclcpp::NodeOptions & options)
: Node("gtsam_fusion_localizer", options)
{
  load_parameters();

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  Eigen::Quaterniond R_base_imu = Eigen::Quaterniond::Identity();
  if (!wait_for_imu_extrinsics(R_base_imu)) {
    RCLCPP_WARN(
      get_logger(),
      "base_link <- imu_link static transform not found within %.1f s. "
      "Proceeding with identity rotation.",
      params_.imu.static_tf_lookup_timeout_sec);
  }

  buffer_ = std::make_unique<SensorBuffer>(params_.keyframe);
  preint_ = std::make_unique<ImuPreintegrator>(params_.imu, R_base_imu);
  graph_ = std::make_unique<FactorGraphManager>(params_);

  // Publishers.
  const auto qos_pub = rclcpp::QoS(10);
  pub_pose_ = create_publisher<geometry_msgs::msg::PoseStamped>("~/output/pose", qos_pub);
  pub_pose_cov_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "~/output/pose_with_covariance", qos_pub);
  pub_biased_pose_ = create_publisher<geometry_msgs::msg::PoseStamped>(
    "~/output/biased_pose", qos_pub);
  pub_biased_pose_cov_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "~/output/biased_pose_with_covariance", qos_pub);
  pub_twist_ = create_publisher<geometry_msgs::msg::TwistStamped>("~/output/twist", qos_pub);
  pub_twist_cov_ = create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
    "~/output/twist_with_covariance", qos_pub);
  pub_kinematic_state_ =
    create_publisher<nav_msgs::msg::Odometry>("~/output/kinematic_state", qos_pub);
  pub_diag_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "/diagnostics", rclcpp::QoS(1).transient_local());

  diag_ = std::make_unique<DiagnosticsPublisher>(pub_diag_, params_.diagnostics);

  // Subscriptions.
  const auto qos_imu = rclcpp::SensorDataQoS();
  const auto qos_pose = rclcpp::QoS(10);
  sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
    "~/input/imu", qos_imu, std::bind(&GtsamFusionLocalizerNode::on_imu, this, _1));
  sub_twist_ = create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
    "~/input/twist", qos_pose, std::bind(&GtsamFusionLocalizerNode::on_twist, this, _1));
  sub_ndt_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "~/input/ndt_pose", qos_pose, std::bind(&GtsamFusionLocalizerNode::on_ndt, this, _1));
  sub_gnss_pose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "~/input/gnss_pose", qos_pose,
    std::bind(&GtsamFusionLocalizerNode::on_gnss_pose, this, _1));
  sub_gnss_fix_ = create_subscription<sensor_msgs::msg::NavSatFix>(
    "~/input/gnss_fix", qos_pose, std::bind(&GtsamFusionLocalizerNode::on_gnss_fix, this, _1));

  // Publish timer.
  const auto period_ns = std::chrono::nanoseconds(
    static_cast<int64_t>(1.0e9 / std::max(1.0, params_.output.publish_rate_hz)));
  publish_timer_ = create_wall_timer(
    period_ns, std::bind(&GtsamFusionLocalizerNode::publish_timer_tick, this));

  if (params_.diagnostics.enable) {
    const auto diag_period = std::chrono::nanoseconds(
      static_cast<int64_t>(1.0e9 / std::max(1.0, params_.diagnostics.rate_hz)));
    diagnostics_timer_ = create_wall_timer(diag_period, [this]() {
      diag_->publish_latest(now());
    });
  }

  worker_ = std::thread(&GtsamFusionLocalizerNode::worker_loop, this);

  RCLCPP_INFO(get_logger(), "gtsam_fusion_localizer started");
}

GtsamFusionLocalizerNode::~GtsamFusionLocalizerNode()
{
  stop_ = true;
  if (buffer_) buffer_->shutdown();
  if (worker_.joinable()) worker_.join();
}

void GtsamFusionLocalizerNode::load_parameters()
{
  auto declare_or_get = [this](const std::string & name, auto & target) {
    target = declare_parameter(name, target);
  };
  declare_or_get("frames.map", params_.frames.map);
  declare_or_get("frames.base_link", params_.frames.base_link);
  declare_or_get("frames.imu_link", params_.frames.imu_link);

  declare_or_get("imu.gravity", params_.imu.gravity);
  declare_or_get("imu.accel_noise_sigma", params_.imu.accel_noise_sigma);
  declare_or_get("imu.gyro_noise_sigma", params_.imu.gyro_noise_sigma);
  declare_or_get("imu.accel_bias_rw_sigma", params_.imu.accel_bias_rw_sigma);
  declare_or_get("imu.gyro_bias_rw_sigma", params_.imu.gyro_bias_rw_sigma);
  declare_or_get("imu.integration_cov", params_.imu.integration_cov);
  declare_or_get("imu.expected_rate_hz", params_.imu.expected_rate_hz);
  declare_or_get("imu.static_tf_lookup_timeout_sec", params_.imu.static_tf_lookup_timeout_sec);

  declare_or_get("keyframe.trigger", params_.keyframe.trigger);
  declare_or_get("keyframe.min_dt_sec", params_.keyframe.min_dt_sec);
  declare_or_get("keyframe.max_dt_sec", params_.keyframe.max_dt_sec);
  declare_or_get("keyframe.ndt_position_gate_m", params_.keyframe.ndt_position_gate_m);
  declare_or_get("keyframe.gnss_position_gate_m", params_.keyframe.gnss_position_gate_m);
  declare_or_get(
    "keyframe.gnss_fix_match_tolerance_sec", params_.keyframe.gnss_fix_match_tolerance_sec);

  declare_or_get("init.source", params_.init.source);
  declare_or_get("init.wait_timeout_sec", params_.init.wait_timeout_sec);
  declare_or_get("init.initial_velocity_sigma", params_.init.initial_velocity_sigma);
  declare_or_get("init.initial_bias_accel_sigma", params_.init.initial_bias_accel_sigma);
  declare_or_get("init.initial_bias_gyro_sigma", params_.init.initial_bias_gyro_sigma);

  declare_or_get("twist.enable", params_.twist.enable);
  declare_or_get("twist.use_scale_variable", params_.twist.use_scale_variable);
  declare_or_get("twist.scale_prior_mean", params_.twist.scale_prior_mean);
  declare_or_get("twist.scale_prior_sigma", params_.twist.scale_prior_sigma);

  declare_or_get("gnss.max_position_sigma", params_.gnss.max_position_sigma);
  declare_or_get("gnss.accept_fix_statuses", params_.gnss.accept_fix_statuses);
  declare_or_get("gnss.use_lever_arm_factor", params_.gnss.use_lever_arm_factor);
  declare_or_get("gnss.lever_arm_xyz", params_.gnss.lever_arm_xyz);
  declare_or_get("gnss.lever_arm_prior_sigma", params_.gnss.lever_arm_prior_sigma);

  declare_or_get("robust.mode", params_.robust.mode);
  declare_or_get("robust.huber_k", params_.robust.huber_k);
  declare_or_get("robust.mahalanobis_chi2_p", params_.robust.mahalanobis_chi2_p);
  declare_or_get("robust.max_consecutive_rejects", params_.robust.max_consecutive_rejects);

  declare_or_get("ndt.degeneracy_condition_threshold", params_.ndt.degeneracy_condition_threshold);
  declare_or_get("ndt.degeneracy_threshold", params_.ndt.degeneracy_threshold);
  declare_or_get("ndt.degeneracy_inflation_factor", params_.ndt.degeneracy_inflation_factor);

  declare_or_get("isam2.relinearize_threshold", params_.isam2.relinearize_threshold);
  declare_or_get("isam2.relinearize_skip", params_.isam2.relinearize_skip);
  declare_or_get("isam2.cache_linearized_factors", params_.isam2.cache_linearized_factors);
  declare_or_get("isam2.factorization", params_.isam2.factorization);
  declare_or_get("isam2.use_fixed_lag_smoother", params_.isam2.use_fixed_lag_smoother);
  declare_or_get("isam2.smoother_lag_sec", params_.isam2.smoother_lag_sec);

  declare_or_get("output.publish_tf", params_.output.publish_tf);
  declare_or_get("output.publish_rate_hz", params_.output.publish_rate_hz);

  declare_or_get("diagnostics.enable", params_.diagnostics.enable);
  declare_or_get("diagnostics.rate_hz", params_.diagnostics.rate_hz);
  declare_or_get(
    "diagnostics.gnss_ndt_divergence_warn_m", params_.diagnostics.gnss_ndt_divergence_warn_m);
  declare_or_get(
    "diagnostics.gnss_ndt_divergence_error_m", params_.diagnostics.gnss_ndt_divergence_error_m);
}

bool GtsamFusionLocalizerNode::wait_for_imu_extrinsics(Eigen::Quaterniond & R_base_imu)
{
  const auto deadline =
    now() + rclcpp::Duration::from_seconds(params_.imu.static_tf_lookup_timeout_sec);
  while (rclcpp::ok() && now() < deadline) {
    try {
      const auto tf = tf_buffer_->lookupTransform(
        params_.frames.base_link, params_.frames.imu_link, tf2::TimePointZero);
      R_base_imu = Eigen::Quaterniond(
        tf.transform.rotation.w, tf.transform.rotation.x, tf.transform.rotation.y,
        tf.transform.rotation.z);
      return true;
    } catch (const tf2::TransformException &) {
      std::this_thread::sleep_for(100ms);
    }
  }
  return false;
}

void GtsamFusionLocalizerNode::on_imu(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
  buffer_->push_imu(*msg);
}

void GtsamFusionLocalizerNode::on_twist(
  const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr msg)
{
  buffer_->push_twist(*msg);
}

void GtsamFusionLocalizerNode::on_ndt(
  const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg)
{
  {
    std::lock_guard<std::mutex> lk(latest_mutex_);
    last_ndt_for_divergence_ = *msg;
  }
  buffer_->push_ndt(*msg);
}

void GtsamFusionLocalizerNode::on_gnss_pose(
  const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg)
{
  if (!gnss_sigma_acceptable(*msg)) {
    return;
  }
  {
    std::lock_guard<std::mutex> lk(latest_mutex_);
    last_gnss_for_divergence_ = *msg;
  }
  buffer_->push_gnss_pose(*msg);
}

void GtsamFusionLocalizerNode::on_gnss_fix(const sensor_msgs::msg::NavSatFix::ConstSharedPtr msg)
{
  if (!gnss_status_accepted(*msg)) {
    return;
  }
  buffer_->push_gnss_fix(*msg);
}

bool GtsamFusionLocalizerNode::gnss_status_accepted(const sensor_msgs::msg::NavSatFix & fix) const
{
  const auto & ok = params_.gnss.accept_fix_statuses;
  return std::find(ok.begin(), ok.end(), static_cast<int64_t>(fix.status.status)) != ok.end();
}

bool GtsamFusionLocalizerNode::gnss_sigma_acceptable(
  const geometry_msgs::msg::PoseWithCovarianceStamped & p) const
{
  const double sx = std::sqrt(std::max(0.0, p.pose.covariance[0]));
  const double sy = std::sqrt(std::max(0.0, p.pose.covariance[7]));
  const double sz = std::sqrt(std::max(0.0, p.pose.covariance[14]));
  const double s = std::sqrt(sx * sx + sy * sy + sz * sz);
  return s <= params_.gnss.max_position_sigma * std::sqrt(3.0);
}

void GtsamFusionLocalizerNode::worker_loop()
{
  while (!stop_) {
    rclcpp::Time last;
    {
      std::lock_guard<std::mutex> lk(latest_mutex_);
      last = last_keyframe_stamp_;
    }
    auto slice = buffer_->wait_for_keyframe(last);
    if (slice.trigger == KeyframeTrigger::Shutdown) break;

    // Integrate IMU samples up to the keyframe stamp.
    for (const auto & imu : slice.imu_samples) {
      preint_->integrate(imu);
    }

    if (!graph_->initialized()) {
      gtsam::Pose3 init_pose;
      bool have_init = false;
      if (slice.ndt) {
        init_pose = conversions::pose_msg_to_gtsam(slice.ndt->pose.pose);
        have_init = true;
      } else if (slice.gnss) {
        init_pose = conversions::pose_msg_to_gtsam(slice.gnss->pose.pose.pose);
        have_init = true;
      }
      if (have_init) {
        gtsam::Vector3 init_v = gtsam::Vector3::Zero();
        if (slice.twist) {
          init_v << slice.twist->twist.twist.linear.x, slice.twist->twist.twist.linear.y,
            slice.twist->twist.twist.linear.z;
        }
        graph_->initialize(slice.stamp, init_pose, init_v, gtsam::imuBias::ConstantBias());
        preint_->reset(graph_->latest_bias());
        {
          std::lock_guard<std::mutex> lk(latest_mutex_);
          last_keyframe_stamp_ = slice.stamp;
        }
      }
      continue;
    }

    KeyframeInputs kf;
    kf.stamp = slice.stamp;
    kf.imu_preint = preint_->take();
    if (slice.ndt) {
      auto cov = conversions::cov_array_to_matrix(slice.ndt->pose.covariance);
      kf.ndt_prior_ros_cov = std::make_pair(
        conversions::pose_msg_to_gtsam(slice.ndt->pose.pose), cov);
    }
    if (slice.gnss) {
      auto cov = conversions::cov_array_to_matrix(slice.gnss->pose.pose.covariance);
      kf.gnss_prior_ros_cov = std::make_pair(
        conversions::pose_msg_to_gtsam(slice.gnss->pose.pose.pose), cov);
    }
    if (slice.twist && params_.twist.enable) {
      const auto & t = slice.twist->twist;
      gtsam::Vector3 v;
      v << t.twist.linear.x, t.twist.linear.y, t.twist.linear.z;
      Eigen::Matrix3d c;
      for (int r = 0; r < 3; ++r) {
        for (int cc = 0; cc < 3; ++cc) {
          c(r, cc) = t.covariance[r * 6 + cc];
        }
      }
      kf.twist_prior = std::make_pair(v, c);
    }

    graph_->add_keyframe(kf);
    preint_->reset(graph_->latest_bias());

    {
      std::lock_guard<std::mutex> lk(latest_mutex_);
      last_keyframe_stamp_ = slice.stamp;
    }

    if (params_.diagnostics.enable) {
      DiagnosticsSnapshot snap;
      snap.stamp = slice.stamp;
      snap.initialized = true;
      snap.graph_size = graph_->key_index() + 1;
      snap.last_update_duration_sec = graph_->last_update_duration_sec();
      snap.chi2_imu = graph_->last_chi2().imu;
      snap.chi2_ndt = graph_->last_chi2().ndt;
      snap.chi2_gnss = graph_->last_chi2().gnss;
      snap.chi2_twist = graph_->last_chi2().twist;
      const auto bias = graph_->latest_bias();
      snap.bias_accel = {bias.accelerometer().x(), bias.accelerometer().y(),
                         bias.accelerometer().z()};
      snap.bias_gyro = {bias.gyroscope().x(), bias.gyroscope().y(), bias.gyroscope().z()};
      if (auto la = graph_->latest_lever_arm()) {
        snap.lever_arm = std::array<double, 3>{la->x(), la->y(), la->z()};
      }
      if (auto sc = graph_->latest_twist_scale()) {
        snap.twist_scale = *sc;
      }
      {
        std::lock_guard<std::mutex> lk(latest_mutex_);
        if (last_ndt_for_divergence_ && last_gnss_for_divergence_) {
          const auto & a = last_ndt_for_divergence_->pose.pose.position;
          const auto & b = last_gnss_for_divergence_->pose.pose.position;
          const double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
          snap.gnss_ndt_divergence_m = std::sqrt(dx * dx + dy * dy + dz * dz);
          snap.has_gnss_ndt_divergence = true;
        }
      }
      diag_->post(snap);
    }
  }
}

void GtsamFusionLocalizerNode::publish_timer_tick()
{
  if (!graph_->initialized()) {
    if (params_.diagnostics.enable) {
      DiagnosticsSnapshot snap;
      snap.stamp = now();
      snap.initialized = false;
      snap.init_reason = "waiting for first NDT or GNSS fix";
      diag_->post(snap);
    }
    return;
  }

  // Forward-propagate using the most recent IMU sample if available.
  gtsam::NavState state;
  gtsam::imuBias::ConstantBias bias;
  Eigen::Matrix<double, 6, 6> pose_cov_gtsam;
  Eigen::Matrix3d vel_cov;
  rclcpp::Time stamp;
  {
    state = graph_->latest_state();
    bias = graph_->latest_bias();
    pose_cov_gtsam = graph_->latest_pose_cov_gtsam_order();
    vel_cov = graph_->latest_velocity_cov();
    stamp = graph_->latest_stamp();
  }

  // Naive forward propagation: rotate latest velocity by accumulated IMU rotation.
  // For Phase A we publish the latest keyframe state directly with `now()` stamp.
  const auto latest_imu = buffer_->latest_imu();
  const rclcpp::Time publish_stamp =
    latest_imu ? rclcpp::Time(latest_imu->header.stamp, RCL_ROS_TIME) : now();

  geometry_msgs::msg::PoseWithCovarianceStamped pose_cov_msg;
  pose_cov_msg.header.stamp = publish_stamp;
  pose_cov_msg.header.frame_id = params_.frames.map;
  pose_cov_msg.pose.pose = conversions::pose_gtsam_to_msg(state.pose());
  const Eigen::Matrix<double, 6, 6> pose_cov_ros =
    conversions::reorder_cov_gtsam_to_ros(pose_cov_gtsam);
  pose_cov_msg.pose.covariance = conversions::matrix_to_cov_array(pose_cov_ros);

  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header = pose_cov_msg.header;
  pose_msg.pose = pose_cov_msg.pose.pose;

  geometry_msgs::msg::TwistWithCovarianceStamped twist_cov_msg;
  twist_cov_msg.header.stamp = publish_stamp;
  twist_cov_msg.header.frame_id = params_.frames.base_link;
  // Velocity in the body frame: rotate world-frame velocity by R^-1.
  const Eigen::Vector3d v_world = state.velocity();
  const Eigen::Matrix3d R_world_base = state.pose().rotation().matrix();
  const Eigen::Vector3d v_body = R_world_base.transpose() * v_world;
  twist_cov_msg.twist.twist.linear.x = v_body.x();
  twist_cov_msg.twist.twist.linear.y = v_body.y();
  twist_cov_msg.twist.twist.linear.z = v_body.z();
  // Angular velocity = latest IMU gyro - estimated bias.
  if (latest_imu) {
    const auto & g = latest_imu->angular_velocity;
    const auto & bg = bias.gyroscope();
    twist_cov_msg.twist.twist.angular.x = g.x - bg.x();
    twist_cov_msg.twist.twist.angular.y = g.y - bg.y();
    twist_cov_msg.twist.twist.angular.z = g.z - bg.z();
  }
  // Body-frame velocity covariance: R^T C R.
  const Eigen::Matrix3d v_body_cov = R_world_base.transpose() * vel_cov * R_world_base;
  for (int r = 0; r < 3; ++r) {
    for (int cc = 0; cc < 3; ++cc) {
      twist_cov_msg.twist.covariance[r * 6 + cc] = v_body_cov(r, cc);
    }
  }

  geometry_msgs::msg::TwistStamped twist_msg;
  twist_msg.header = twist_cov_msg.header;
  twist_msg.twist = twist_cov_msg.twist.twist;

  nav_msgs::msg::Odometry odom;
  odom.header.stamp = publish_stamp;
  odom.header.frame_id = params_.frames.map;
  odom.child_frame_id = params_.frames.base_link;
  odom.pose = pose_cov_msg.pose;
  odom.twist = twist_cov_msg.twist;

  pub_pose_->publish(pose_msg);
  pub_pose_cov_->publish(pose_cov_msg);
  pub_biased_pose_->publish(pose_msg);
  pub_biased_pose_cov_->publish(pose_cov_msg);
  pub_twist_->publish(twist_msg);
  pub_twist_cov_->publish(twist_cov_msg);
  pub_kinematic_state_->publish(odom);

  if (params_.output.publish_tf) {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = publish_stamp;
    t.header.frame_id = params_.frames.map;
    t.child_frame_id = params_.frames.base_link;
    t.transform.translation.x = pose_msg.pose.position.x;
    t.transform.translation.y = pose_msg.pose.position.y;
    t.transform.translation.z = pose_msg.pose.position.z;
    t.transform.rotation = pose_msg.pose.orientation;
    tf_broadcaster_->sendTransform(t);
  }

  (void)stamp;
}

}  // namespace autoware::gtsam_fusion_localizer

RCLCPP_COMPONENTS_REGISTER_NODE(autoware::gtsam_fusion_localizer::GtsamFusionLocalizerNode)

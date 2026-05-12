// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "autoware/gtsam_fusion_localizer/factor_graph_manager.hpp"

#include "autoware/gtsam_fusion_localizer/factors/gnss_lever_arm_factor.hpp"
#include "autoware/gtsam_fusion_localizer/factors/scaled_twist_factor.hpp"
#include "autoware/gtsam_fusion_localizer/factors/switchable_constraint_factor.hpp"
#include "autoware/gtsam_fusion_localizer/ros_gtsam_conversions.hpp"

#include <Eigen/Eigenvalues>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/slam/PriorFactor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

namespace autoware::gtsam_fusion_localizer
{

namespace
{
using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::L;  // lever arm
using gtsam::symbol_shorthand::S;  // twist scale
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::W;  // switchable-constraint switch variable
using gtsam::symbol_shorthand::X;

// Mahalanobis distance squared.
double mahalanobis_sq(const Eigen::Matrix<double, 6, 1> & e, const Eigen::Matrix<double, 6, 6> & C)
{
  Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(C);
  if (ldlt.info() != Eigen::Success) {
    return std::numeric_limits<double>::infinity();
  }
  const Eigen::Matrix<double, 6, 1> y = ldlt.solve(e);
  return e.transpose() * y;
}

// chi-squared inverse CDF approximation for dof=6 and a few common p values.
double chi2_inv_dof6(double p)
{
  if (p >= 0.999) return 22.458;
  if (p >= 0.99) return 16.812;
  if (p >= 0.975) return 14.449;
  if (p >= 0.95) return 12.592;
  if (p >= 0.90) return 10.645;
  return 10.645;
}

}  // namespace

FactorGraphManager::FactorGraphManager(const Parameters & params) : params_(params)
{
  apply_isam2_params();
}

void FactorGraphManager::apply_isam2_params()
{
  gtsam::ISAM2Params isam_params;
  isam_params.relinearizeThreshold = params_.isam2.relinearize_threshold;
  isam_params.relinearizeSkip = params_.isam2.relinearize_skip;
  isam_params.cacheLinearizedFactors = params_.isam2.cache_linearized_factors;
  if (params_.isam2.factorization == "QR") {
    isam_params.factorization = gtsam::ISAM2Params::QR;
  } else {
    isam_params.factorization = gtsam::ISAM2Params::CHOLESKY;
  }

  if (params_.isam2.use_fixed_lag_smoother) {
    smoother_ = std::make_shared<gtsam::IncrementalFixedLagSmoother>(
      params_.isam2.smoother_lag_sec, isam_params);
    isam_.reset();
  } else {
    isam_ = std::make_shared<gtsam::ISAM2>(isam_params);
    smoother_.reset();
  }
}

bool FactorGraphManager::initialize(
  const rclcpp::Time & stamp, const gtsam::Pose3 & initial_pose,
  const gtsam::Vector3 & initial_velocity, const gtsam::imuBias::ConstantBias & initial_bias)
{
  if (initialized_) return false;

  // Priors. Pose prior is loose to allow the first aiding to drive it; the noise here is
  // intentionally generous in rotation (5 deg) and translation (1 m) when there's no better
  // ground truth.
  gtsam::Vector6 pose_sigma;
  pose_sigma << 0.1, 0.1, 0.1, 1.0, 1.0, 1.0;  // [rot(3); trans(3)]
  auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas(pose_sigma);
  auto velocity_noise =
    gtsam::noiseModel::Isotropic::Sigma(3, params_.init.initial_velocity_sigma);
  gtsam::Vector6 bias_sigma;
  bias_sigma.head<3>().setConstant(params_.init.initial_bias_accel_sigma);
  bias_sigma.tail<3>().setConstant(params_.init.initial_bias_gyro_sigma);
  auto bias_noise = gtsam::noiseModel::Diagonal::Sigmas(bias_sigma);

  new_factors_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), initial_pose, pose_noise));
  new_factors_.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), initial_velocity, velocity_noise));
  new_factors_.add(
    gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), initial_bias, bias_noise));

  new_values_.insert(X(0), initial_pose);
  new_values_.insert(V(0), initial_velocity);
  new_values_.insert(B(0), initial_bias);

  const double t = stamp.seconds();
  new_timestamps_[X(0)] = t;
  new_timestamps_[V(0)] = t;
  new_timestamps_[B(0)] = t;

  if (params_.gnss.use_lever_arm_factor) {
    const auto & xyz = params_.gnss.lever_arm_xyz;
    const gtsam::Point3 lever(xyz.size() > 0 ? xyz[0] : 0.0, xyz.size() > 1 ? xyz[1] : 0.0,
                              xyz.size() > 2 ? xyz[2] : 0.0);
    auto lever_noise =
      gtsam::noiseModel::Isotropic::Sigma(3, params_.gnss.lever_arm_prior_sigma);
    new_factors_.add(
      gtsam::PriorFactor<gtsam::Point3>(L(kLeverArmKeyIdx), lever, lever_noise));
    new_values_.insert(L(kLeverArmKeyIdx), lever);
    new_timestamps_[L(kLeverArmKeyIdx)] = t;
    latest_lever_arm_ = lever;
  }

  if (params_.twist.use_scale_variable) {
    auto scale_noise =
      gtsam::noiseModel::Isotropic::Sigma(1, params_.twist.scale_prior_sigma);
    new_factors_.add(gtsam::PriorFactor<double>(
      S(kTwistScaleKeyIdx), params_.twist.scale_prior_mean, scale_noise));
    new_values_.insert<double>(S(kTwistScaleKeyIdx), params_.twist.scale_prior_mean);
    new_timestamps_[S(kTwistScaleKeyIdx)] = t;
    latest_twist_scale_ = params_.twist.scale_prior_mean;
  }

  key_idx_ = 0;
  latest_stamp_ = stamp;
  latest_state_ = gtsam::NavState(initial_pose, initial_velocity);
  latest_bias_ = initial_bias;
  initialized_ = true;
  update();
  return true;
}

gtsam::SharedNoiseModel FactorGraphManager::wrap_robust(const gtsam::SharedNoiseModel & base) const
{
  if (params_.robust.mode == "huber") {
    return gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(params_.robust.huber_k), base);
  }
  if (params_.robust.mode == "cauchy") {
    return gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Cauchy::Create(params_.robust.huber_k), base);
  }
  return base;
}

Eigen::Matrix<double, 6, 6> FactorGraphManager::inflate_ndt_cov(
  const Eigen::Matrix<double, 6, 6> & ros_cov) const
{
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(ros_cov);
  if (solver.info() != Eigen::Success) return ros_cov;
  Eigen::Matrix<double, 6, 1> eig = solver.eigenvalues();
  const Eigen::Matrix<double, 6, 6> & V = solver.eigenvectors();
  const double max_e = eig.maxCoeff();
  const double min_e = std::max(eig.minCoeff(), 1.0e-12);
  const bool ill_conditioned = (max_e / min_e) > params_.ndt.degeneracy_condition_threshold;
  for (int i = 0; i < 6; ++i) {
    const bool axis_degenerate = eig(i) > params_.ndt.degeneracy_threshold;
    if (axis_degenerate || ill_conditioned) {
      eig(i) *= params_.ndt.degeneracy_inflation_factor;
    }
  }
  return V * eig.asDiagonal() * V.transpose();
}

bool FactorGraphManager::gate_ndt(
  const gtsam::Pose3 & predicted, const gtsam::Pose3 & measured,
  const Eigen::Matrix<double, 6, 6> & combined_cov_gtsam)
{
  const gtsam::Vector6 err = gtsam::Pose3::Logmap(predicted.inverse() * measured);
  // Simple translation gate first (cheap), then Mahalanobis.
  const double trans_norm = err.tail<3>().norm();
  if (trans_norm > params_.keyframe.ndt_position_gate_m) {
    ++consecutive_ndt_rejects_;
    if (consecutive_ndt_rejects_ >= params_.robust.max_consecutive_rejects) {
      // Admit with inflated covariance to catch up.
      consecutive_ndt_rejects_ = 0;
      return true;
    }
    return false;
  }
  const double thresh = chi2_inv_dof6(params_.robust.mahalanobis_chi2_p);
  const double d2 = mahalanobis_sq(err, combined_cov_gtsam);
  if (d2 > thresh) {
    ++consecutive_ndt_rejects_;
    if (consecutive_ndt_rejects_ >= params_.robust.max_consecutive_rejects) {
      consecutive_ndt_rejects_ = 0;
      return true;
    }
    return false;
  }
  consecutive_ndt_rejects_ = 0;
  return true;
}

bool FactorGraphManager::gate_gnss(
  const gtsam::Pose3 & predicted, const gtsam::Pose3 & measured,
  const Eigen::Matrix<double, 6, 6> & combined_cov_gtsam)
{
  const gtsam::Vector6 err = gtsam::Pose3::Logmap(predicted.inverse() * measured);
  const double trans_norm = err.tail<3>().norm();
  if (trans_norm > params_.keyframe.gnss_position_gate_m) {
    ++consecutive_gnss_rejects_;
    if (consecutive_gnss_rejects_ >= params_.robust.max_consecutive_rejects) {
      consecutive_gnss_rejects_ = 0;
      return true;
    }
    return false;
  }
  const double thresh = chi2_inv_dof6(params_.robust.mahalanobis_chi2_p);
  const double d2 = mahalanobis_sq(err, combined_cov_gtsam);
  if (d2 > thresh) {
    ++consecutive_gnss_rejects_;
    if (consecutive_gnss_rejects_ >= params_.robust.max_consecutive_rejects) {
      consecutive_gnss_rejects_ = 0;
      return true;
    }
    return false;
  }
  consecutive_gnss_rejects_ = 0;
  return true;
}

Eigen::Matrix<double, 6, 6> FactorGraphManager::predicted_pose_cov(
  const gtsam::PreintegratedCombinedMeasurements & preint) const
{
  // Coarse approximation: latest pose marginal plus the preintegrated 6x6 pose block.
  const auto pim_cov = preint.preintMeasCov();
  Eigen::Matrix<double, 6, 6> pose_block = Eigen::Matrix<double, 6, 6>::Zero();
  pose_block.block<3, 3>(0, 0) = pim_cov.block<3, 3>(0, 0);  // rotation
  pose_block.block<3, 3>(3, 3) = pim_cov.block<3, 3>(3, 3);  // translation
  return latest_pose_cov_gtsam_ + pose_block;
}

bool FactorGraphManager::add_keyframe(const KeyframeInputs & inputs)
{
  if (!initialized_) return false;
  if (!inputs.imu_preint) return false;

  const std::uint64_t i = key_idx_;
  const std::uint64_t j = i + 1;
  const double t = inputs.stamp.seconds();

  // Predict next state from IMU preintegration.
  const gtsam::NavState predicted = inputs.imu_preint->predict(latest_state_, latest_bias_);
  new_values_.insert(X(j), predicted.pose());
  new_values_.insert(V(j), predicted.velocity());
  new_values_.insert(B(j), latest_bias_);
  new_timestamps_[X(j)] = t;
  new_timestamps_[V(j)] = t;
  new_timestamps_[B(j)] = t;

  // IMU factor between i and j.
  new_factors_.add(gtsam::CombinedImuFactor(
    X(i), V(i), X(j), V(j), B(i), B(j), *inputs.imu_preint));

  // NDT prior on X(j).
  if (inputs.ndt_prior_ros_cov) {
    auto ndt_cov_ros = inputs.ndt_prior_ros_cov->second;
    if (params_.robust.mode != "none") {
      ndt_cov_ros = inflate_ndt_cov(ndt_cov_ros);
    }
    const auto ndt_cov_gtsam = conversions::reorder_cov_ros_to_gtsam(ndt_cov_ros);
    const auto pred_cov_gtsam = predicted_pose_cov(*inputs.imu_preint);
    const auto combined = ndt_cov_gtsam + pred_cov_gtsam;
    const bool admit = (params_.robust.mode == "none")
                         ? true
                         : gate_ndt(predicted.pose(), inputs.ndt_prior_ros_cov->first, combined);
    if (admit) {
      auto base = gtsam::noiseModel::Gaussian::Covariance(ndt_cov_gtsam);
      auto noise = wrap_robust(base);
      new_factors_.add(
        gtsam::PriorFactor<gtsam::Pose3>(X(j), inputs.ndt_prior_ros_cov->first, noise));
      ++last_chi2_.ndt_count;
    }
  }

  // GNSS prior on X(j) (with lever-arm factor in Phase C).
  if (inputs.gnss_prior_ros_cov) {
    const auto gnss_cov_ros = inputs.gnss_prior_ros_cov->second;
    const auto gnss_cov_gtsam = conversions::reorder_cov_ros_to_gtsam(gnss_cov_ros);
    const auto pred_cov_gtsam = predicted_pose_cov(*inputs.imu_preint);
    const auto combined = gnss_cov_gtsam + pred_cov_gtsam;
    const bool admit = (params_.robust.mode == "none")
                         ? true
                         : gate_gnss(predicted.pose(), inputs.gnss_prior_ros_cov->first, combined);
    if (admit) {
      if (params_.gnss.use_lever_arm_factor) {
        // Use only the translation block as the antenna position observation; we approximate
        // the noise as a 3x3 isotropic from the trace of the position block.
        const Eigen::Matrix3d t_cov_gtsam = gnss_cov_gtsam.block<3, 3>(3, 3);
        auto pos_noise = gtsam::noiseModel::Gaussian::Covariance(t_cov_gtsam);
        auto noise = wrap_robust(pos_noise);
        new_factors_.add(factors::GnssLeverArmFactor(
          X(j), L(kLeverArmKeyIdx), inputs.gnss_prior_ros_cov->first.translation(), noise));
      } else if (params_.robust.mode == "switchable") {
        // Add a switch variable W(j) per GNSS factor.
        auto base = gtsam::noiseModel::Gaussian::Covariance(gnss_cov_gtsam);
        auto pose_noise = wrap_robust(base);
        new_factors_.add(factors::SwitchablePose3PriorFactor(
          X(j), W(j), inputs.gnss_prior_ros_cov->first, pose_noise));
        auto switch_prior = gtsam::noiseModel::Isotropic::Sigma(1, 1.0);
        new_factors_.add(gtsam::PriorFactor<double>(W(j), 1.0, switch_prior));
        new_values_.insert<double>(W(j), 1.0);
        new_timestamps_[W(j)] = t;
      } else {
        auto base = gtsam::noiseModel::Gaussian::Covariance(gnss_cov_gtsam);
        auto noise = wrap_robust(base);
        new_factors_.add(
          gtsam::PriorFactor<gtsam::Pose3>(X(j), inputs.gnss_prior_ros_cov->first, noise));
      }
      ++last_chi2_.gnss_count;
    }
  }

  // Twist prior on V(j) (linear velocity only).
  if (inputs.twist_prior && params_.twist.enable) {
    auto noise = gtsam::noiseModel::Gaussian::Covariance(inputs.twist_prior->second);
    if (params_.twist.use_scale_variable) {
      new_factors_.add(factors::ScaledTwistFactor(
        V(j), S(kTwistScaleKeyIdx), inputs.twist_prior->first, noise));
    } else {
      new_factors_.add(
        gtsam::PriorFactor<gtsam::Vector3>(V(j), inputs.twist_prior->first, noise));
    }
  }

  key_idx_ = j;
  latest_stamp_ = inputs.stamp;
  update();
  return true;
}

void FactorGraphManager::update()
{
  if (new_factors_.empty() && new_values_.empty()) return;
  const auto t_start = std::chrono::steady_clock::now();

  gtsam::Values estimate;
  if (smoother_) {
    gtsam::FixedLagSmoother::KeyTimestampMap stamps(new_timestamps_.begin(), new_timestamps_.end());
    smoother_->update(new_factors_, new_values_, stamps);
    estimate = smoother_->calculateEstimate();
  } else {
    isam_->update(new_factors_, new_values_);
    estimate = isam_->calculateEstimate();
  }

  // Compute chi-squared for the most recent batch (before clearing).
  last_chi2_.imu = 0.0;
  last_chi2_.ndt = 0.0;
  last_chi2_.gnss = 0.0;
  last_chi2_.twist = 0.0;
  for (const auto & f : new_factors_) {
    if (!f) continue;
    const double e = f->error(estimate);
    if (dynamic_cast<const gtsam::CombinedImuFactor *>(f.get())) {
      last_chi2_.imu += e;
    } else if (
      dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3> *>(f.get()) ||
      dynamic_cast<const factors::SwitchablePose3PriorFactor *>(f.get()) ||
      dynamic_cast<const factors::GnssLeverArmFactor *>(f.get())) {
      last_chi2_.ndt += e;
    } else if (
      dynamic_cast<const gtsam::PriorFactor<gtsam::Vector3> *>(f.get()) ||
      dynamic_cast<const factors::ScaledTwistFactor *>(f.get())) {
      last_chi2_.twist += e;
    }
  }

  new_factors_.resize(0);
  new_values_.clear();
  new_timestamps_.clear();

  const gtsam::Pose3 pose_j = estimate.at<gtsam::Pose3>(X(key_idx_));
  const gtsam::Vector3 vel_j = estimate.at<gtsam::Vector3>(V(key_idx_));
  const gtsam::imuBias::ConstantBias bias_j =
    estimate.at<gtsam::imuBias::ConstantBias>(B(key_idx_));
  latest_state_ = gtsam::NavState(pose_j, vel_j);
  latest_bias_ = bias_j;

  // Pose covariance for the current keyframe.
  try {
    if (smoother_) {
      const gtsam::Matrix M = smoother_->marginalCovariance(X(key_idx_));
      latest_pose_cov_gtsam_ = M;
      latest_velocity_cov_ = smoother_->marginalCovariance(V(key_idx_));
    } else {
      const gtsam::Matrix M = isam_->marginalCovariance(X(key_idx_));
      latest_pose_cov_gtsam_ = M;
      latest_velocity_cov_ = isam_->marginalCovariance(V(key_idx_));
    }
  } catch (const std::exception &) {
    // Marginal computation can fail early; keep the previous covariance.
  }

  if (params_.gnss.use_lever_arm_factor && estimate.exists(L(kLeverArmKeyIdx))) {
    latest_lever_arm_ = estimate.at<gtsam::Point3>(L(kLeverArmKeyIdx));
  }
  if (params_.twist.use_scale_variable && estimate.exists(S(kTwistScaleKeyIdx))) {
    latest_twist_scale_ = estimate.at<double>(S(kTwistScaleKeyIdx));
  }

  const auto t_end = std::chrono::steady_clock::now();
  last_update_duration_sec_ =
    std::chrono::duration<double>(t_end - t_start).count();
}

gtsam::NavState FactorGraphManager::latest_state() const { return latest_state_; }
gtsam::imuBias::ConstantBias FactorGraphManager::latest_bias() const { return latest_bias_; }
Eigen::Matrix<double, 6, 6> FactorGraphManager::latest_pose_cov_gtsam_order() const
{
  return latest_pose_cov_gtsam_;
}
Eigen::Matrix3d FactorGraphManager::latest_velocity_cov() const { return latest_velocity_cov_; }
std::optional<gtsam::Point3> FactorGraphManager::latest_lever_arm() const
{
  return latest_lever_arm_;
}
std::optional<double> FactorGraphManager::latest_twist_scale() const
{
  return latest_twist_scale_;
}

}  // namespace autoware::gtsam_fusion_localizer

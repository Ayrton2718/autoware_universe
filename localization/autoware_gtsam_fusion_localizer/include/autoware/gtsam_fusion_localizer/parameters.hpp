// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#ifndef AUTOWARE__GTSAM_FUSION_LOCALIZER__PARAMETERS_HPP_
#define AUTOWARE__GTSAM_FUSION_LOCALIZER__PARAMETERS_HPP_

#include <cstdint>
#include <string>
#include <vector>

namespace autoware::gtsam_fusion_localizer
{

struct ImuParams
{
  double gravity{9.80665};
  double accel_noise_sigma{0.05};
  double gyro_noise_sigma{0.005};
  double accel_bias_rw_sigma{1.0e-4};
  double gyro_bias_rw_sigma{1.0e-5};
  double integration_cov{1.0e-8};
  double expected_rate_hz{100.0};
  double static_tf_lookup_timeout_sec{5.0};
};

struct KeyframeParams
{
  std::string trigger{"ndt_or_gnss"};
  double min_dt_sec{0.05};
  double max_dt_sec{1.0};
  double ndt_position_gate_m{5.0};
  double gnss_position_gate_m{5.0};
  double gnss_fix_match_tolerance_sec{0.05};
};

struct InitParams
{
  std::string source{"ndt_then_gnss"};
  double wait_timeout_sec{30.0};
  double initial_velocity_sigma{0.5};
  double initial_bias_accel_sigma{0.1};
  double initial_bias_gyro_sigma{0.01};
};

struct TwistParams
{
  bool enable{true};
  bool use_scale_variable{false};
  double scale_prior_mean{1.0};
  double scale_prior_sigma{0.05};
};

struct GnssParams
{
  double max_position_sigma{0.5};
  std::vector<int64_t> accept_fix_statuses{0, 1, 2};
  bool use_lever_arm_factor{false};
  std::vector<double> lever_arm_xyz{1.2, 0.0, 1.5};
  double lever_arm_prior_sigma{0.05};
};

struct RobustParams
{
  std::string mode{"none"};  // none | huber | cauchy | switchable
  double huber_k{1.345};
  double mahalanobis_chi2_p{0.99};
  int max_consecutive_rejects{5};
};

struct NdtParams
{
  double degeneracy_condition_threshold{100.0};
  double degeneracy_threshold{0.5};
  double degeneracy_inflation_factor{100.0};
};

struct Isam2Params
{
  double relinearize_threshold{0.01};
  int relinearize_skip{1};
  bool cache_linearized_factors{true};
  std::string factorization{"CHOLESKY"};
  bool use_fixed_lag_smoother{false};
  double smoother_lag_sec{30.0};
};

struct OutputParams
{
  bool publish_tf{true};
  double publish_rate_hz{50.0};
};

struct DiagnosticsParams
{
  bool enable{true};
  double rate_hz{1.0};
  double gnss_ndt_divergence_warn_m{0.5};
  double gnss_ndt_divergence_error_m{2.0};
};

struct FramesParams
{
  std::string map{"map"};
  std::string base_link{"base_link"};
  std::string imu_link{"imu_link"};
};

struct Parameters
{
  FramesParams frames;
  ImuParams imu;
  KeyframeParams keyframe;
  InitParams init;
  TwistParams twist;
  GnssParams gnss;
  RobustParams robust;
  NdtParams ndt;
  Isam2Params isam2;
  OutputParams output;
  DiagnosticsParams diagnostics;
};

}  // namespace autoware::gtsam_fusion_localizer

#endif  // AUTOWARE__GTSAM_FUSION_LOCALIZER__PARAMETERS_HPP_

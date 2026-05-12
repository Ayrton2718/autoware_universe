// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#ifndef AUTOWARE__GTSAM_FUSION_LOCALIZER__FACTORS__SWITCHABLE_CONSTRAINT_FACTOR_HPP_
#define AUTOWARE__GTSAM_FUSION_LOCALIZER__FACTORS__SWITCHABLE_CONSTRAINT_FACTOR_HPP_

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace autoware::gtsam_fusion_localizer::factors
{

// A Pose3 prior factor with a continuous trustworthiness scalar switch in [0, 1].
// Residual = switch * Logmap(measured^-1 * estimated). The optimizer is free to drive
// `switch` toward 0 for outliers (a unit prior on `switch` keeps it near 1 by default).
// Reference: Suenderhauf et al., "Switchable Constraints for Robust Pose Graph SLAM" (IROS 2012).
class SwitchablePose3PriorFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, double>
{
public:
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, double>;

  SwitchablePose3PriorFactor(
    gtsam::Key pose_key, gtsam::Key switch_key, const gtsam::Pose3 & measurement,
    const gtsam::SharedNoiseModel & noise)
  : Base(noise, pose_key, switch_key), measurement_(measurement)
  {
  }

  gtsam::Vector evaluateError(
    const gtsam::Pose3 & T, const double & s, gtsam::OptionalMatrixType H1 = nullptr,
    gtsam::OptionalMatrixType H2 = nullptr) const override;

private:
  gtsam::Pose3 measurement_;
};

}  // namespace autoware::gtsam_fusion_localizer::factors

#endif  // AUTOWARE__GTSAM_FUSION_LOCALIZER__FACTORS__SWITCHABLE_CONSTRAINT_FACTOR_HPP_

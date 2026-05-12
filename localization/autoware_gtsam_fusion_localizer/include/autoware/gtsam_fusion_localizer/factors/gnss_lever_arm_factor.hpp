// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#ifndef AUTOWARE__GTSAM_FUSION_LOCALIZER__FACTORS__GNSS_LEVER_ARM_FACTOR_HPP_
#define AUTOWARE__GTSAM_FUSION_LOCALIZER__FACTORS__GNSS_LEVER_ARM_FACTOR_HPP_

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace autoware::gtsam_fusion_localizer::factors
{

// Predicts antenna position in the map frame as p_world = T_world_base * lever_arm.
// The factor residual is (predicted - measured).
class GnssLeverArmFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Point3>
{
public:
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Point3>;

  GnssLeverArmFactor(
    gtsam::Key pose_key, gtsam::Key lever_key, const gtsam::Point3 & measurement,
    const gtsam::SharedNoiseModel & noise)
  : Base(noise, pose_key, lever_key), measurement_(measurement)
  {
  }

  gtsam::Vector evaluateError(
    const gtsam::Pose3 & T_world_base, const gtsam::Point3 & lever_arm,
    gtsam::OptionalMatrixType H1 = nullptr,
    gtsam::OptionalMatrixType H2 = nullptr) const override;

  const gtsam::Point3 & measurement() const { return measurement_; }

private:
  gtsam::Point3 measurement_;
};

}  // namespace autoware::gtsam_fusion_localizer::factors

#endif  // AUTOWARE__GTSAM_FUSION_LOCALIZER__FACTORS__GNSS_LEVER_ARM_FACTOR_HPP_

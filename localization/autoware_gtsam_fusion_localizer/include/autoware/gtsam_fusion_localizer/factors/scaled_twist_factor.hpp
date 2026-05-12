// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#ifndef AUTOWARE__GTSAM_FUSION_LOCALIZER__FACTORS__SCALED_TWIST_FACTOR_HPP_
#define AUTOWARE__GTSAM_FUSION_LOCALIZER__FACTORS__SCALED_TWIST_FACTOR_HPP_

#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace autoware::gtsam_fusion_localizer::factors
{

// Models v_measured = s * v_true. Error = measured - s * v.
class ScaledTwistFactor : public gtsam::NoiseModelFactorN<gtsam::Vector3, double>
{
public:
  using Base = gtsam::NoiseModelFactorN<gtsam::Vector3, double>;

  ScaledTwistFactor(
    gtsam::Key velocity_key, gtsam::Key scale_key, const gtsam::Vector3 & measurement,
    const gtsam::SharedNoiseModel & noise)
  : Base(noise, velocity_key, scale_key), measurement_(measurement)
  {
  }

  gtsam::Vector evaluateError(
    const gtsam::Vector3 & v, const double & s, gtsam::OptionalMatrixType H1 = nullptr,
    gtsam::OptionalMatrixType H2 = nullptr) const override;

  const gtsam::Vector3 & measurement() const { return measurement_; }

private:
  gtsam::Vector3 measurement_;
};

}  // namespace autoware::gtsam_fusion_localizer::factors

#endif  // AUTOWARE__GTSAM_FUSION_LOCALIZER__FACTORS__SCALED_TWIST_FACTOR_HPP_

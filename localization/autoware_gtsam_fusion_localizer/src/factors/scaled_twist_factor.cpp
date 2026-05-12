// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "autoware/gtsam_fusion_localizer/factors/scaled_twist_factor.hpp"

namespace autoware::gtsam_fusion_localizer::factors
{

gtsam::Vector ScaledTwistFactor::evaluateError(
  const gtsam::Vector3 & v, const double & s, gtsam::OptionalMatrixType H1,
  gtsam::OptionalMatrixType H2) const
{
  // residual = measurement - s * v
  if (H1) {
    *H1 = -s * gtsam::Matrix3::Identity();
  }
  if (H2) {
    Eigen::Matrix<double, 3, 1> J;
    J.col(0) = -v;
    *H2 = J;
  }
  return measurement_ - s * v;
}

}  // namespace autoware::gtsam_fusion_localizer::factors

// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "autoware/gtsam_fusion_localizer/factors/switchable_constraint_factor.hpp"

#include <algorithm>

namespace autoware::gtsam_fusion_localizer::factors
{

gtsam::Vector SwitchablePose3PriorFactor::evaluateError(
  const gtsam::Pose3 & T, const double & s, gtsam::OptionalMatrixType H1,
  gtsam::OptionalMatrixType H2) const
{
  gtsam::Matrix6 D_err_T;
  // log(measured^-1 * T): zero when they coincide.
  const gtsam::Pose3 between = measurement_.inverse().compose(T);
  const gtsam::Vector6 raw_err = gtsam::Pose3::Logmap(between, H1 ? &D_err_T : nullptr);

  // The switch is clamped to [0, 1] for residual purposes; the variable itself stays unbounded
  // so the optimizer can move freely.
  const double s_clamped = std::clamp(s, 0.0, 1.0);

  if (H1) {
    *H1 = s_clamped * D_err_T;
  }
  if (H2) {
    gtsam::Matrix J(6, 1);
    if (s >= 0.0 && s <= 1.0) {
      J.col(0) = raw_err;
    } else {
      J.setZero();
    }
    *H2 = J;
  }
  return s_clamped * raw_err;
}

}  // namespace autoware::gtsam_fusion_localizer::factors

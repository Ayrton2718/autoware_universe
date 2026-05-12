// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "autoware/gtsam_fusion_localizer/factors/gnss_lever_arm_factor.hpp"

namespace autoware::gtsam_fusion_localizer::factors
{

gtsam::Vector GnssLeverArmFactor::evaluateError(
  const gtsam::Pose3 & T_world_base, const gtsam::Point3 & lever_arm,
  gtsam::OptionalMatrixType H1, gtsam::OptionalMatrixType H2) const
{
  gtsam::Matrix36 D_pred_pose;
  gtsam::Matrix3 D_pred_lever;
  const gtsam::Point3 predicted = T_world_base.transformFrom(lever_arm, &D_pred_pose, &D_pred_lever);
  if (H1) *H1 = D_pred_pose;
  if (H2) *H2 = D_pred_lever;
  return predicted - measurement_;
}

}  // namespace autoware::gtsam_fusion_localizer::factors

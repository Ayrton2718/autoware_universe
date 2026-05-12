// Copyright 2026 Autoware contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "autoware/gtsam_fusion_localizer/factors/gnss_lever_arm_factor.hpp"
#include "autoware/gtsam_fusion_localizer/factors/scaled_twist_factor.hpp"
#include "autoware/gtsam_fusion_localizer/factors/switchable_constraint_factor.hpp"

#include <gtest/gtest.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>

namespace agf = autoware::gtsam_fusion_localizer::factors;

TEST(GnssLeverArmFactor, ErrorZeroAtMeasurement)
{
  using gtsam::symbol_shorthand::L;
  using gtsam::symbol_shorthand::X;
  const gtsam::Pose3 T(gtsam::Rot3::Yaw(0.3), gtsam::Point3(10.0, -2.0, 1.0));
  const gtsam::Point3 lever(1.2, 0.0, 1.5);
  const gtsam::Point3 measurement = T.transformFrom(lever);
  auto noise = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
  agf::GnssLeverArmFactor f(X(0), L(0), measurement, noise);
  const auto err = f.evaluateError(T, lever);
  EXPECT_LT(err.norm(), 1.0e-9);
}

TEST(GnssLeverArmFactor, JacobiansMatchNumerical)
{
  using gtsam::symbol_shorthand::L;
  using gtsam::symbol_shorthand::X;
  const gtsam::Pose3 T(gtsam::Rot3::Yaw(0.3), gtsam::Point3(10.0, -2.0, 1.0));
  const gtsam::Point3 lever(1.2, 0.1, 1.5);
  const gtsam::Point3 measurement(11.0, -1.5, 2.4);
  auto noise = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
  agf::GnssLeverArmFactor f(X(0), L(0), measurement, noise);

  gtsam::Matrix H1, H2;
  f.evaluateError(T, lever, &H1, &H2);

  const auto numH1 = gtsam::numericalDerivative21<gtsam::Vector3, gtsam::Pose3, gtsam::Point3>(
    [&](const gtsam::Pose3 & p, const gtsam::Point3 & l) { return f.evaluateError(p, l); }, T,
    lever);
  const auto numH2 = gtsam::numericalDerivative22<gtsam::Vector3, gtsam::Pose3, gtsam::Point3>(
    [&](const gtsam::Pose3 & p, const gtsam::Point3 & l) { return f.evaluateError(p, l); }, T,
    lever);
  EXPECT_TRUE(H1.isApprox(numH1, 1.0e-5));
  EXPECT_TRUE(H2.isApprox(numH2, 1.0e-5));
}

TEST(ScaledTwistFactor, ErrorZeroAtMeasurement)
{
  using gtsam::symbol_shorthand::S;
  using gtsam::symbol_shorthand::V;
  const gtsam::Vector3 v(3.0, 0.5, 0.0);
  const double s = 1.05;
  const gtsam::Vector3 measurement = s * v;
  auto noise = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
  agf::ScaledTwistFactor f(V(0), S(0), measurement, noise);
  const auto err = f.evaluateError(v, s);
  EXPECT_LT(err.norm(), 1.0e-12);
}

TEST(ScaledTwistFactor, JacobiansMatchNumerical)
{
  using gtsam::symbol_shorthand::S;
  using gtsam::symbol_shorthand::V;
  const gtsam::Vector3 v(2.0, -1.0, 0.3);
  const double s = 0.95;
  const gtsam::Vector3 measurement(2.5, -0.8, 0.4);
  auto noise = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
  agf::ScaledTwistFactor f(V(0), S(0), measurement, noise);
  gtsam::Matrix H1, H2;
  f.evaluateError(v, s, &H1, &H2);

  const auto numH1 = gtsam::numericalDerivative21<gtsam::Vector3, gtsam::Vector3, double>(
    [&](const gtsam::Vector3 & vv, const double & ss) { return f.evaluateError(vv, ss); }, v, s);
  const auto numH2 = gtsam::numericalDerivative22<gtsam::Vector3, gtsam::Vector3, double>(
    [&](const gtsam::Vector3 & vv, const double & ss) { return f.evaluateError(vv, ss); }, v, s);
  EXPECT_TRUE(H1.isApprox(numH1, 1.0e-6));
  EXPECT_TRUE(H2.isApprox(numH2, 1.0e-6));
}

TEST(SwitchablePose3PriorFactor, ZeroErrorAtMeasurement)
{
  using gtsam::symbol_shorthand::W;
  using gtsam::symbol_shorthand::X;
  const gtsam::Pose3 T(gtsam::Rot3::Yaw(0.1), gtsam::Point3(1, 2, 3));
  auto noise = gtsam::noiseModel::Isotropic::Sigma(6, 0.1);
  agf::SwitchablePose3PriorFactor f(X(0), W(0), T, noise);
  const auto err = f.evaluateError(T, 1.0);
  EXPECT_LT(err.norm(), 1.0e-12);
}

TEST(SwitchablePose3PriorFactor, ZeroSwitchSuppressesError)
{
  using gtsam::symbol_shorthand::W;
  using gtsam::symbol_shorthand::X;
  const gtsam::Pose3 measured(gtsam::Rot3::Yaw(0.1), gtsam::Point3(0, 0, 0));
  const gtsam::Pose3 estimated(gtsam::Rot3::Yaw(0.5), gtsam::Point3(5, 5, 5));
  auto noise = gtsam::noiseModel::Isotropic::Sigma(6, 0.1);
  agf::SwitchablePose3PriorFactor f(X(0), W(0), measured, noise);
  const auto err_open = f.evaluateError(estimated, 1.0);
  const auto err_off = f.evaluateError(estimated, 0.0);
  EXPECT_GT(err_open.norm(), 0.1);
  EXPECT_LT(err_off.norm(), 1.0e-12);
}

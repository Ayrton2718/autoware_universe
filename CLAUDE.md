# CLAUDE.md - Autoware Universe Codebase Guide

This document provides essential context for AI assistants working with the Autoware Universe repository.

## Project Overview

Autoware Universe is an open-source autonomous driving software stack built on **ROS 2** (Robot Operating System 2). It provides modular, safety-critical components for perception, planning, control, localization, and system management of autonomous vehicles.

- **License**: Apache License 2.0
- **ROS 2 Distributions Supported**: Humble, Jazzy
- **Architecture Support**: x86-64, ARM64
- **GPU Support**: CUDA/TensorRT-enabled builds
- **Current Version**: 0.50.0
- **Total Packages**: ~184 ROS 2 packages

## Repository Structure

```
autoware_universe/
├── common/          # Shared utilities and libraries (17 packages)
├── control/         # Vehicle control and trajectory following (21 packages)
├── e2e/             # End-to-end ML components (e.g., TensorRT VAD)
├── evaluator/       # Metrics and evaluation tools (7 packages)
├── examples/        # Diagnostic graph test examples
├── localization/    # Pose estimation and localization (6 packages)
├── map/             # Map handling and TF generation (1 package)
├── perception/      # Detection, tracking, recognition (51 packages)
├── planning/        # Motion planning and trajectory generation (20 packages)
├── sensing/         # Sensor preprocessing (13 packages)
├── simulator/       # Simulation tools (7 packages)
├── system/          # System monitoring, diagnostics, lifecycle (24 packages)
├── vehicle/         # Vehicle command conversion and calibration (4 packages)
└── visualization/   # RViz2 plugins and monitors (12 packages)
```

### Domain Breakdown

**common/** - Core shared libraries: `autoware_universe_utils` (geometry, math, ROS utilities), `autoware_component_interface_utils`, `autoware_traffic_light_utils`, `autoware_grid_map_utils`, `autoware_fake_test_node`, etc.

**perception/** - The largest domain (51 packages). Key packages:
- Detection: `autoware_lidar_centerpoint`, `autoware_bevfusion`, `autoware_lidar_transfusion`, `autoware_bytetrack`
- ML Inference: `autoware_tensorrt_yolox`, `autoware_lidar_frnet`, `autoware_ptv3`
- Tracking: `autoware_multi_object_tracker`, `autoware_radar_object_tracker`
- Prediction: `autoware_map_based_prediction`
- Segmentation: `autoware_ground_segmentation`, `autoware_euclidean_cluster`
- Traffic Lights: `autoware_traffic_light_classifier`, `autoware_crosswalk_traffic_light_estimator`

**planning/** - Motion planning stack:
- `autoware_behavior_path_planner` / `autoware_behavior_velocity_planner` (behavior modules)
- `autoware_path_optimizer`, `autoware_path_smoother` (trajectory generation)
- `autoware_motion_velocity_planner`, `autoware_external_velocity_limit_selector`
- `autoware_mission_planner_universe`, `autoware_freespace_planner`
- `autoware_diffusion_planner` (ML-based planner)

**control/** - Vehicle control:
- `autoware_mpc_lateral_controller`, `autoware_pid_longitudinal_controller`, `autoware_pure_pursuit`
- `autoware_autonomous_emergency_braking`, `autoware_lane_departure_checker`
- `autoware_control_command_gate`, `autoware_vehicle_cmd_gate`

**system/** - Infrastructure:
- `autoware_diagnostic_graph_aggregator`, `autoware_component_monitor`
- `autoware_command_mode_decider`, `autoware_command_mode_switcher`
- `autoware_default_adapi_universe`, `autoware_duplicated_node_checker`
- `autoware_bluetooth_monitor`, `autoware_mrm_*` (minimal risk maneuver handlers)

**localization/** - Pose estimation:
- `autoware_ndt_scan_matcher`, `autoware_ekf_localizer`
- `autoware_pose_estimator_arbiter`, `autoware_landmark_based_localizer`

## Build System

### Tools
- **Build tool**: `colcon` (ROS 2 standard)
- **Build system**: CMake with `ament_cmake_auto` and `autoware_cmake`
- **C++ standard**: C++14 minimum, C++17 allowed (e.g., `<filesystem>`)
- **Compiler flags**: `-Wall -Wextra -Wpedantic` (enforced)

### Package Structure

Every package contains:
- `package.xml` (ROS 2 format 3) - declares dependencies
- `CMakeLists.txt` - build configuration
- `include/<package_name>/` - public headers (`.hpp`)
- `src/` - source files (`.cpp`)
- `test/` - unit tests
- `docs/` - documentation (many packages)
- `config/` - parameter files (`.yaml`)
- `launch/` - launch files (`.launch.xml` or `.launch.py`)

### Typical CMakeLists.txt Pattern

```cmake
cmake_minimum_required(VERSION 3.5)
project(autoware_my_package)

find_package(autoware_cmake REQUIRED)
autoware_package()

ament_auto_find_build_dependencies()

ament_auto_add_library(autoware_my_package SHARED
  src/my_node.cpp
)

rclcpp_components_register_node(autoware_my_package
  PLUGIN "autoware::my_package::MyNode"
  EXECUTABLE my_node
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_my_package test/test_my_package.cpp)
  target_link_libraries(test_my_package autoware_my_package)
endif()

ament_auto_package(INSTALL_TO_SHARE config launch)
```

### Typical package.xml Pattern

```xml
<?xml version="1.0"?>
<package format="3">
  <name>autoware_my_package</name>
  <version>0.50.0</version>
  <description>Description here</description>
  <maintainer email="maintainer@example.com">Maintainer Name</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake_auto</buildtool_depend>
  <buildtool_depend>autoware_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclcpp_components</depend>
  <depend>autoware_utils</depend>
  <!-- other dependencies -->

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>autoware_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

## Code Conventions

### C++ Style

- **Line length**: 100 characters maximum
- **Formatting**: clang-format with Google style, customized (see `.clang-format`)
  - Brace wrapping for namespaces and classes
  - Pointer alignment: Middle (`int * ptr`)
  - Include category ordering: ROS/system headers, then local headers
- **Linting**: cpplint (see `CPPLINT.cfg`)
  - C++11 and C++17 features allowed
  - Non-const reference parameters are acceptable
  - Custom include order style enforced

### Naming Conventions

- **Packages**: `autoware_<domain>_<description>` (e.g., `autoware_lidar_centerpoint`)
- **Namespaces**: `autoware::<package_name>` (snake_case)
- **Classes**: PascalCase
- **Files**: snake_case `.hpp` / `.cpp`
- **ROS 2 plugins**: `autoware::<package_name>::<ClassName>`

### Node Architecture Pattern

All nodes **must** use the composable node (component) architecture:

```cpp
namespace autoware::my_package
{
class MyNode : public rclcpp::Node
{
public:
  explicit MyNode(const rclcpp::NodeOptions & options);
  // ...
};
}  // namespace autoware::my_package

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::my_package::MyNode)
```

### Python Style

- **Formatter**: black with 100 char line length
- **Import sorting**: isort with black profile, 100 char line length
- **Linting**: flake8 with ament_lint extensions
- **Max line length**: 100 characters

### YAML / Launch Files

- Launch files use either `.launch.xml` or `.launch.py`
- Parameters stored in `config/` as `.yaml` files
- YAML linting via yamllint (see `.yamllint.yaml`)

## Testing

### Framework

- **Unit tests**: GTest via `ament_cmake_gtest` or `ament_add_ros_isolated_gtest`
- **Test location**: `test/` subdirectory of each package
- **Lint tests**: Auto-configured via `ament_lint_auto` + `autoware_lint_common`

### Running Tests

```bash
# Build with tests
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Run all tests
colcon test

# Run tests for a specific package
colcon test --packages-select autoware_my_package

# View test results
colcon test-result --verbose
```

### Test Patterns

```cpp
#include <gtest/gtest.h>
#include "autoware_my_package/my_class.hpp"

TEST(TestMyClass, BasicTest)
{
  autoware::my_package::MyClass obj;
  EXPECT_EQ(obj.compute(), expected_value);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
```

## Pre-commit Hooks

The repository uses 34 pre-commit hooks. Run them before committing:

```bash
pre-commit run --all-files
```

Key hooks:
- **clang-format**: C++ formatting (auto-fix)
- **black**: Python formatting (auto-fix)
- **isort**: Python import sorting (auto-fix)
- **cpplint**: C++ linting (report only)
- **flake8**: Python linting (report only)
- **prettier**: YAML/JSON/XML formatting (auto-fix)
- **shellcheck + shfmt**: Shell script linting and formatting
- **markdownlint**: Markdown formatting
- **yamllint**: YAML linting
- **spell-check**: cspell-based spell checker
- **sort-package-xml**: Keeps package.xml dependencies sorted
- **check-package-depends**: Validates dependency consistency
- **ros-include-guard**: Enforces ROS-style header guards

## CI/CD Workflows

GitHub Actions workflows in `.github/workflows/`:

| Workflow | Trigger | Purpose |
|---|---|---|
| `build-and-test.yaml` | Push to main | Full build + test |
| `build-and-test-differential.yaml` | PR | Test only changed packages |
| `build-and-test-reusable.yaml` | Called by others | Matrix: humble/jazzy, x86/arm64 |
| `build-and-test-daily.yaml` | Scheduled | Daily full build |
| `clang-tidy-differential.yaml` | PR | Incremental clang-tidy |
| `cppcheck-differential.yaml` | PR | Static analysis |
| `pre-commit.yaml` | PR | Lint/format validation |
| `semantic-pull-request.yaml` | PR | Conventional commit validation |
| `deploy-docs.yaml` | Push to main | MKDocs to GitHub Pages |
| `check-build-depends.yaml` | PR | Dependency validation |

### Container Images

CI uses `ghcr.io/autowarefoundation/autoware:universe-devel` (and `-jazzy`, `-cuda` variants).

## Documentation

- **Framework**: MKDocs with Material theme
- **Site**: https://autowarefoundation.github.io/autoware_universe
- **Per-package docs**: In `docs/` subdirectory of each package
- **Build locally**: `mkdocs serve`

Most packages with algorithms or complex behavior have a `docs/` directory with:
- Algorithm explanations
- Parameter descriptions
- Architecture diagrams (in `media/` or `image/` subdirectories)

## Commit Message Conventions

Follow **Conventional Commits** (enforced by CI):

```
<type>(<scope>): <description>

Types: feat, fix, refactor, chore, docs, test, perf, ci, build, revert
Scope: package name (e.g., autoware_lidar_centerpoint)

Examples:
feat(autoware_lidar_centerpoint): add multi-format point cloud input
fix(autoware_pid_longitudinal_controller): fix test for ROS 2 Jazzy
chore(autoware_multi_object_tracker): convert array config to explicit keys
refactor(autoware_tensorrt_yolox): parameterize label remapping
```

## Key Dependencies

| Package | Purpose |
|---|---|
| `rclcpp`, `rclcpp_components` | ROS 2 C++ client library + composable nodes |
| `autoware_utils` | Core Autoware utilities (geometry, math, ROS helpers) |
| `autoware_universe_utils` | Extended universe utilities |
| `autoware_perception_msgs` | Perception message types |
| `autoware_planning_msgs` | Planning message types |
| `autoware_vehicle_msgs` | Vehicle command/state messages |
| `autoware_vehicle_info_utils` | Vehicle parameter management |
| `Eigen3` | Linear algebra |
| `Boost` | Various utilities |
| `PCL` | Point Cloud Library |
| `TensorRT` | NVIDIA inference engine (perception) |
| `CUDA` | GPU compute (perception, sensing) |
| `fmt` | C++ formatting library |

## Configuration Files

| File | Purpose |
|---|---|
| `.clang-format` | C++ code formatting rules |
| `.clang-tidy-ignore` | Clang-tidy suppressions |
| `.cppcheck_suppressions` | CPPCheck suppressions |
| `.pre-commit-config.yaml` | Pre-commit hook configuration |
| `.markdownlint.yaml` | Markdown linting rules |
| `.prettierrc.yaml` | Prettier formatter configuration |
| `.yamllint.yaml` | YAML linting rules |
| `.cspell.json` | Spell checker dictionary |
| `setup.cfg` | Python linting (flake8, isort) |
| `CPPLINT.cfg` | C++ cpplint configuration |
| `codecov.yaml` | Code coverage settings |
| `mkdocs.yaml` | Documentation site configuration |

## Development Workflow

### Setup

```bash
# Clone with dependencies
mkdir autoware && cd autoware
vcs import src < autoware.repos
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

### Adding a New Package

1. Create directory under the appropriate domain: `<domain>/autoware_<name>/`
2. Add `package.xml` with `format="3"`, Apache 2.0 license, correct dependencies
3. Add `CMakeLists.txt` using `autoware_cmake` pattern
4. Use `autoware::<package_name>` namespace in C++
5. Implement as composable node (rclcpp_components)
6. Add lint tests via `ament_lint_auto` + `autoware_lint_common`
7. Add unit tests in `test/` directory
8. Add documentation in `docs/` directory

### Modifying Existing Packages

1. Always read package code before modifying
2. Run pre-commit on changed files before committing
3. Ensure tests pass: `colcon test --packages-select <package>`
4. Follow conventional commit format for commit messages

### Common pitfalls

- **Include guards**: Use `#ifndef AUTOWARE_<PACKAGE>__<FILE>_HPP_` format (enforced by pre-commit)
- **Header extensions**: Use `.hpp` (not `.h`) for C++ headers
- **Namespace closing**: Add comment `// namespace autoware::<package_name>` after closing brace
- **Parameter validation**: Always validate at node construction, not inside callbacks
- **Thread safety**: Use `std::mutex` or ROS 2 callback groups for shared state
- **Message timestamps**: Use `node->now()` not `rclcpp::Clock().now()`

## Code Coverage

Coverage is tracked per domain on codecov.io:
- Common, Control, Evaluator, Localization, Map, Perception, Planning, Sensing, Simulator, System, Vehicle

## Useful Resources

- Autoware documentation: https://autowarefoundation.github.io/autoware-documentation/
- Contributing guidelines: https://autowarefoundation.github.io/autoware-documentation/main/contributing/
- ROS 2 documentation: https://docs.ros.org/en/humble/

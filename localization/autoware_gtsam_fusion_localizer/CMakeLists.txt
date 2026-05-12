cmake_minimum_required(VERSION 3.14)
project(autoware_gtsam_fusion_localizer)

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/ros_gtsam_conversions.cpp
  src/sensor_buffer.cpp
  src/imu_preintegrator.cpp
  src/factor_graph_manager.cpp
  src/diagnostics_publisher.cpp
  src/factors/gnss_lever_arm_factor.cpp
  src/factors/scaled_twist_factor.cpp
  src/factors/switchable_constraint_factor.cpp
  src/gtsam_fusion_localizer_node.cpp
)

target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
  ${GTSAM_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  gtsam
  gtsam_unstable
  Eigen3::Eigen
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "autoware::gtsam_fusion_localizer::GtsamFusionLocalizerNode"
  EXECUTABLE ${PROJECT_NAME}_node
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_${PROJECT_NAME}
    test/test_ros_gtsam_conversions.cpp
    test/test_factor_graph_manager.cpp
    test/test_factors.cpp
  )
  target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME})
endif()

ament_auto_package(INSTALL_TO_SHARE
  config
  launch
  schema
)

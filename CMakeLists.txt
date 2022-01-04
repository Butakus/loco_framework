cmake_minimum_required(VERSION 3.5)
project(loco_framework)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)


# Add vehicle detection msgs
rosidl_generate_interfaces(${PROJECT_NAME}
  msg/PlatoonDetection.msg
  msg/PlatoonDetectionArray.msg
  msg/DetectionArray.msg
  DEPENDENCIES std_msgs geometry_msgs
)

# Add OdometryBuffer target library
add_library(odometry_buffer SHARED
  src/loco_framework/odometry_buffer.cpp
)
target_include_directories(odometry_buffer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(odometry_buffer
  rclcpp
  nav_msgs
)

# Add Hungarian target library
add_library(hungarian SHARED
  src/loco_framework/hungarian.cpp
)
target_include_directories(hungarian PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)


# Add PlatoonEstimator target library
add_library(loco_platoon_estimator SHARED
  src/loco_framework/estimators/platoon_estimator.cpp
)
target_include_directories(loco_platoon_estimator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(loco_platoon_estimator
  rclcpp
  geometry_msgs
  nav_msgs
)
rosidl_target_interfaces(loco_platoon_estimator ${PROJECT_NAME} rosidl_typesupport_cpp)

# Add LocoPlatoon target library
add_library(loco_platoon_node SHARED
  src/loco_framework/loco_platoon_node.cpp
)
target_include_directories(loco_platoon_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(loco_platoon_node
  rclcpp
  rclcpp_components
  rcl_interfaces
  geometry_msgs
  nav_msgs
)
target_link_libraries(loco_platoon_node
  loco_platoon_estimator
  hungarian
  odometry_buffer
)
rosidl_target_interfaces(loco_platoon_node ${PROJECT_NAME} rosidl_typesupport_cpp)
rclcpp_components_register_nodes(loco_platoon_node "loco::LocoPlatoon")

# Add loco_platoon_node executable target
add_executable(loco_platoon
  src/loco_platoon.cpp
)
target_include_directories(loco_platoon PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(loco_platoon
  rclcpp
)
target_link_libraries(loco_platoon
  loco_platoon_node
)

# Add OdometryBuffer test target
add_executable(odometry_buffer_test
  test/odometry_buffer_test.cpp
)
target_include_directories(odometry_buffer_test PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(odometry_buffer_test
  rclcpp
  nav_msgs
)
target_link_libraries(odometry_buffer_test
  odometry_buffer
)

# Add Hungarian test target
add_executable(hungarian_test
  test/hungarian_test.cpp
)
target_include_directories(hungarian_test PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(hungarian_test
  hungarian
)

# Add PlatoonEstimator test target
add_executable(loco_platoon_estimator_test
  test/platoon_estimator_test.cpp
)
target_include_directories(loco_platoon_estimator_test PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(loco_platoon_estimator_test
  rclcpp
  geometry_msgs
  nav_msgs
)
target_link_libraries(loco_platoon_estimator_test
  loco_platoon_estimator
)

# Install library targets
install(TARGETS
  odometry_buffer
  hungarian
  loco_platoon_estimator
  loco_platoon_node
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install executable targets
install(TARGETS
  loco_platoon
  loco_platoon_estimator_test
  odometry_buffer_test
  hungarian_test
  DESTINATION lib/${PROJECT_NAME}
)

# Install include directory
install(
  DIRECTORY include/
  DESTINATION include
)

# Install Python scripts
# ament_python_install_package(scripts)
install(PROGRAMS
  scripts/introspection_test.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(
  include
)

ament_export_dependencies(
  rosidl_default_runtime
  rclcpp
  rclcpp_components
  rcl_interfaces
  geometry_msgs
  nav_msgs
)

ament_export_libraries(
  odometry_buffer
  loco_platoon_estimator
  loco_platoon_node
)

ament_package()

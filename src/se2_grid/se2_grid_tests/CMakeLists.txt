cmake_minimum_required(VERSION 3.5.1)
project(se2_grid_tests)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  se2_grid_core
  se2_grid_msgs
  se2_grid_ros
  se2_grid_rviz_plugin
  geometry_msgs
  sensor_msgs
)

# find_package(OpenCV REQUIRED
#   COMPONENTS
#   opencv_highgui
#   CONFIG
# )

catkin_package(
  # INCLUDE_DIRS include
#  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
#  DEPENDS system_lib
)

include_directories(
  # include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(move_test
  src/move_test_node.cpp
)

add_executable(iterators_test
  src/iterators_test_node.cpp
)

add_executable(interpolation_test
  src/interpolation_test_node.cpp
)

target_link_libraries(
  move_test
  ${catkin_LIBRARIES}
)

target_link_libraries(
  iterators_test
  ${catkin_LIBRARIES}
)

target_link_libraries(
  interpolation_test
  ${catkin_LIBRARIES}
)

# Mark executables and/or libraries for installation
install(
  TARGETS 
    move_test
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark other files for installation
install(
  DIRECTORY config data doc launch rviz scripts
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

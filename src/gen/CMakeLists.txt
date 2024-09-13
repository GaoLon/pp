cmake_minimum_required(VERSION 3.5.1)

execute_process(COMMAND whoami OUTPUT_VARIABLE CURRENT_USER OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_BUILD_TYPE "Release")

set(CUDA_TOOLKIT_ROOT_DIR "/home/xulong/cuda/cuda-11.7")
project(pp CXX CUDA)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++11")
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  std_msgs
  geometry_msgs
  pcl_conversions
)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math;)
set(CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75;)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

add_library(cuda_computer src/pp.cu)
target_link_libraries(cuda_computer 
  ${CUDA_LIBRARIES}
)

set(
  CATKIN_PACKAGE_DEPENDENCIES
    se2_grid_core
    se2_grid_ros
    se2_grid_msgs
    roscpp
)

find_package(catkin REQUIRED 
  COMPONENTS
    ${CATKIN_PACKAGE_DEPENDENCIES}
)

catkin_package(
  INCLUDE_DIRS
    ${Eigen_INCLUDE_DIRS}
  CATKIN_DEPENDS
    ${CATKIN_PACKAGE_DEPENDENCIES}
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_executable (pp src/pp.cpp )
target_link_libraries(pp
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  cuda_computer
)

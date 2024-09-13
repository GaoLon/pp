cmake_minimum_required(VERSION 3.5.1)
project(se2_grid_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  SE2GridInfo.msg
  SE2Grid.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
    CATKIN_DEPENDS
    geometry_msgs
    message_runtime
    std_msgs
)

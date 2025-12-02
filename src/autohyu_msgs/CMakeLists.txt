cmake_minimum_required(VERSION 3.0.2)
project(autohyu_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  message_generation
)

add_message_files(
  FILES
  DetectObjects3D.msg
  DetectObject3D.msg
  ObjectDimension.msg
  Object3DState.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs
)

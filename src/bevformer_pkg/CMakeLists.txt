cmake_minimum_required(VERSION 3.0.2)
project(bevformer_pkg)

## 1. roscpp와 cv_bridge를 find_package에 추가합니다.
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  message_filters
  sensor_msgs
  nav_msgs
  visualization_msgs
  std_msgs
  tf
  cv_bridge
  autohyu_msgs
)

## 2. catkin include 디렉토리를 명시적으로 추가합니다. (이것이 핵심 수정 사항)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## OpenCV 패키지 찾기
find_package(OpenCV REQUIRED)

## generate_messages (기존과 동일)
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

## catkin_package (기존과 동일)
catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs message_runtime cv_bridge autohyu_msgs
)

###########
## Build ##
###########

## 3. C++ 실행 파일 빌드 (파일 이름을 실제 파일명인 img_undistort.cpp로 수정)
add_executable(fast_undistort_node src/img_undistort.cpp)

## C++ 실행 파일에 라이브러리 링크 (기존과 동일)
target_link_libraries(fast_undistort_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)


#############
## Install ##
#############

## Python 노드 설치 (기존과 동일)
catkin_install_python(PROGRAMS
  src/bevformer_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## C++ 실행 파일 설치 (새로 추가)
install(TARGETS fast_undistort_node
  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

## launch 디렉토리 설치 (기존과 동일)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
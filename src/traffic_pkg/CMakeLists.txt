cmake_minimum_required(VERSION 3.0.2)
project(traffic_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

# 파이썬 노드/리소스 설치
install(PROGRAMS
  scripts/traffic_light_inference_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 가중치/체크포인트 등 리소스 동봉 시 (패키지 내부 models/ 디렉토리)
install(DIRECTORY
  models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

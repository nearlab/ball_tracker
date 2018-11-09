cmake_minimum_required(VERSION 3.0.0)
project(ball_tracker)

find_package(catkin REQUIRED COMPONENTS message_generation rospy pv_estimator std_msgs)

catkin_package( CATKIN_DEPENDS 
  message_runtime
  pv_estimator
  rospy
  std_msgs
)

catkin_install_python(PROGRAMS bin/ball_tracking.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
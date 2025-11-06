cmake_minimum_required(VERSION 3.0.2)
project(sponge_mpc)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/include)
find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  message_generation
  std_msgs
  roslib
)
find_package(CASADI REQUIRED)


add_service_files(
  FILES
  mpc_step.srv
)


generate_messages(
  DEPENDENCIES
  std_msgs
)
include_directories(
  include
  ${CASADI_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS 
  roscpp 
  rospy 
  std_msgs 
  message_runtime)

add_executable(mpc_node src/mpc_node.cpp)
target_link_libraries(mpc_node ${catkin_LIBRARIES} ${CASADI_LIBRARIES}) 
add_dependencies(mpc_node sponge_mpc_generate_messages_cpp)

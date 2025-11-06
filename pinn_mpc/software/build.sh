#!/bin/bash -e
source /opt/ros/noetic/setup.bash
source catkin_ws/devel/setup.bash
cd catkin_ws
catkin_make install

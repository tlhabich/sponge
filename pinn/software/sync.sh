#!/bin/bash -e
scp test_bench RTPC_ec:rtmdl/
ssh RTPC_ec 'mkdir -p ~/app_interface/ros_install'
cd catkin_ws
rsync -rltv --delete install scripts RTPC_ec:~/app_interface/ros_install

#!/bin/bash -e
# Copy to real-time computer (RTPC_ec is added to ~/.ssh/config)
scp test_bench RTPC_ec:rtmdl/
ssh RTPC_ec 'mkdir -p ~/app_interface/ros_install'
cd catkin_ws
rsync -rltv --delete install scripts RTPC_ec:~/app_interface/ros_install

cd ..
scp Extern/libMPC_BLOCK.so RTPC_ec:app_interface

# Send parameters for MPC to real-time computer
ssh RTPC_ec 'mkdir -p ~/app_interface/ros_install/include/models'
scp Extern/MPC_casadi_SFunction/include/models/sys_params.yaml RTPC_ec:~/app_interface/ros_install/include/models
scp Extern/MPC_casadi_SFunction/include/models/mpc_params.yaml RTPC_ec:~/app_interface/ros_install/include/models

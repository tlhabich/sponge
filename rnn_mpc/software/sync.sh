#!/bin/bash -e
# Beispiel auf Testrechner kopieren.
# Der Rechner RTPC_ec ist in der ~/.ssh/config eingetragen
# Dieses Skript muss aus dem Verzeichnis aufgerufen werden, in dem es liegt.

# Moritz Schappler, moritz.schappler@imes.uni-hannover.de, 2020-03  
# (C) Institut für Mechatronische Systeme, Leibniz Universität Hannover

# Simulink-Modell kopieren
scp test_bench RTPC_ec:rtmdl/

# ROS-Workspace (Installationsordner) kopieren
ssh RTPC_ec 'mkdir -p ~/app_interface/ros_install'
cd catkin_ws
rsync -rltv --delete install scripts RTPC_ec:~/app_interface/ros_install

cd ..
scp Extern/libMPC_BLOCK.so RTPC_ec:app_interface

# send parameters for MPC to real-time computer
ssh RTPC_ec 'mkdir -p ~/app_interface/ros_install/include/models'
scp Extern/MPC_casadi_SFunction/include/models/sys_params.yaml RTPC_ec:~/app_interface/ros_install/include/models
scp Extern/MPC_casadi_SFunction/include/models/mpc_params.yaml RTPC_ec:~/app_interface/ros_install/include/models

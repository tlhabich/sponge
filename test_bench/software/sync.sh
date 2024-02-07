#!/bin/bash -e
# Beispiel auf Testrechner kopieren.
# Der Rechner imessmartrtpc1_ec ist in der ~/.ssh/config eingetragen
# Dieses Skript muss aus dem Verzeichnis aufgerufen werden, in dem es liegt.

# Moritz Schappler, moritz.schappler@imes.uni-hannover.de, 2020-03  
# (C) Institut für Mechatronische Systeme, Leibniz Universität Hannover

# Simulink-Modell kopieren
scp appint_ros_example RTPC_ec:rtmdl/
# ROS-Workspace (Installationsordner) kopieren
ssh RTPC_ec 'mkdir -p ~/app_interface/ros_install'
cd catkin_ws
rsync -rltv --delete install scripts RTPC_ec:~/app_interface/ros_install

#!/bin/bash -e
# Copy to RT-PC

# Moritz Schappler, moritz.schappler@imes.uni-hannover.de, 2020-03  
# (C) Institut für Mechatronische Systeme, Leibniz Universität Hannover

scp test_bench_sponge RTPC_ec:rtmdl/
ssh RTPC_ec 'mkdir -p ~/app_interface/ros_install'
cd catkin_ws
rsync -rltv --delete install scripts RTPC_ec:~/app_interface/ros_install

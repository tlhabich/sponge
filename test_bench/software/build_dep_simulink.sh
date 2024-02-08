#!/bin/bash -e
# ROS-Real-Time-Interface for Simulink

# Moritz Schappler, moritz.schappler@imes.uni-hannover.de, 2020-03  
# (C) Institut für Mechatronische Systeme, Leibniz Universität Hannover

cd ros_rt_interface
rm -rf build
mkdir -p build
make
cd ..

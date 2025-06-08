#!/bin/bash -e
cd ros_rt_interface
rm -rf build
mkdir -p build
make
cd ..

#!/bin/bash -e

# Build configuration
BUILDDIR=out/build
TARGETDIR=mFiles/build/
PROJECT_LIBRARY_NAME="MPC_BLOCK"        # Project/Block name
PROJECT_LIBRARY_NAME_SO="lib$PROJECT_LIBRARY_NAME.so"   # .dylib for MacOS, .so for Ubuntu

# Usage:
# To build for soft-robot system: ./build.sh 1

# Clean and create build directory
rm -rf $BUILDDIR
mkdir -p $BUILDDIR 

# Run cmake from main directory
# Checks cmake versions, updates submodules in extern/ and runs cmake in src/
cmake -S . -B $BUILDDIR -DCMAKE_DUMMY=0 -DPROJECT_LIBRARY_NAME=$PROJECT_LIBRARY_NAME -DBUILD_LIB=1 -DCMAKE_MODULE_PATH=/usr/local/lib/cmake/casadi -DMODEL=$1

# This will create output on every line of cmake. Good for debugging of CMakeLists!
# cmake -S . -B $BUILDDIR -DCMAKE_DUMMY=0 -DPROJECT_LIBRARY_NAME=$PROJECT_LIBRARY_NAME --trace-source=CMakeLists.txt
echo "Installing Library!"

# Run make from build directory
cd $BUILDDIR && make
cd ../..

# Clean and create target directory
rm -rf $TARGETDIR
mkdir -p $TARGETDIR 

# Copy target library and IO-xml to corresponding folder
cp -v $BUILDDIR/src/$PROJECT_LIBRARY_NAME_SO $TARGETDIR
cp -v $BUILDDIR/src/$PROJECT_LIBRARY_NAME_SO ..
# cp -v sfunction_in_out.xml $TARGETDIR


# call run("cmake-sfunction-cpp-vorlage/mFiles/generate_sl_block.m") from matlab/simulink project folder containing this submodule. cmake-sfunction-cpp-vorlage is the name you gave this submodule while adding to your project.





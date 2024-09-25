#!/bin/bash -e
BUILDDIR=out/build
TARGETDIR=mFiles/build/
PROJECT_LIBRARY_NAME="MPC_BLOCK"		# Ändere diesen Wert zu deinem Projektnamen/Blocknamen
PROJECT_LIBRARY_NAME_SO="lib$PROJECT_LIBRARY_NAME.so"   # Unter MacOS .dylib; Unter Ubuntu .so


# usage:
# To build for soft-robot system      ./build.sh 1



# clean build directory
rm -rf $BUILDDIR
mkdir -p $BUILDDIR 

# run cmake from main (this) directory. Checks for cmake versions, updates submodules in extern/ and runs cmake in src/ folder. For any changes check CMAKELIST in /src folder
cmake -S . -B $BUILDDIR -DCMAKE_DUMMY=0 -DPROJECT_LIBRARY_NAME=$PROJECT_LIBRARY_NAME -DBUILD_LIB=1 -DCMAKE_MODULE_PATH=/usr/local/lib/cmake/casadi -DMODEL=$1

# This will create output on every line of cmake. Good for debugging of CMakeLists!
# cmake -S . -B $BUILDDIR -DCMAKE_DUMMY=0 -DPROJECT_LIBRARY_NAME=$PROJECT_LIBRARY_NAME --trace-source=CMakeLists.txt

echo "Installing Library!"

# run make from build directory
cd $BUILDDIR && make
cd ../..

# clean target directory
rm -rf $TARGETDIR
mkdir -p $TARGETDIR 

# copy target library and IO -xml to corresponding folder.
cp -v $BUILDDIR/src/$PROJECT_LIBRARY_NAME_SO $TARGETDIR
cp -v $BUILDDIR/src/$PROJECT_LIBRARY_NAME_SO ..
# cp -v sfunction_in_out.xml $TARGETDIR


# call run("cmake-sfunction-cpp-vorlage/mFiles/generate_sl_block.m") from matlab/simulink project folder containing this submodule. cmake-sfunction-cpp-vorlage is the name you gave this submodule while adding to your project.





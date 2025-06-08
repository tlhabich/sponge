#!/bin/bash
prefix=/home/ec/app_interface/ros_install/install
addenv () {
        if ! eval "echo \$$1" | /bin/grep -Eq "(^|:)$2($|:)" ; then
            export $1="$2:$(eval "echo \$$1")"
        fi
}
addenv ROS_PACKAGE_PATH $prefix/share
addenv LD_LIBRARY_PATH $prefix/lib
addenv PYTHONPATH $prefix/lib/python2.7/site-packages
addenv PKG_CONFIG_PATH $prefix/lib/pkgconfig
addenv CMAKE_PREFIX_PATH $prefix

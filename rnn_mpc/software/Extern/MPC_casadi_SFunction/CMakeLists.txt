# For a guide on this projects structure see https://cliutils.gitlab.io/modern-cmake/
# This is the top level CMakeLists for this project. 
cmake_minimum_required(VERSION 3.11...3.27)

project(
    smr_dyn 
    VERSION 0.1 
    LANGUAGES CXX
)

# support folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# set install paths for shared libraries and executables
set(CMAKE_INSTALL_RPATH "/usr/local/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# includes are placed here
include(CMakePrintHelpers)
include(FetchContent)
include(CTest)

# FetchContent_MakeAvailable was not added until CMake 3.14; use our shim
if(${CMAKE_VERSION} VERSION_LESS 3.14)
  include(cmake/add_FetchContent_MakeAvailable.cmake)
endif()

### Require out-of-source builds
# Avoids building in valid source directories
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

# Update all git submodules
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
endif()


add_subdirectory(src)

include(CMakePrintHelpers)

# Find Required Packages
set(CASADI_INCLUDE_DIR /usr/local/include/casadi)
find_package (Eigen3 REQUIRED NO_MODULE)
include_directories(${Eigen3_INCLUDE_DIR})
find_package(yaml-cpp REQUIRED)
find_package(CASADI REQUIRED)
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CASADI_INCLUDE_DIR}
)

set(CMAKE_CXX_STANDARD 14)

message("Casadi include dir: ${CASADI_INCLUDE_DIR}")

# To use with Simulink in S-Function Block use 'BUILD_LIB'
if(BUILD_LIB)

message("compile library")

if(CMAKE_DUMMY)
    message("Dummy is going to be compiled and linked! Afterwards compile S-Function and SL-Model.")
    add_library(${PROJECT_LIBRARY_NAME} SHARED sl_func_dummy.cpp)
    target_compile_definitions(${PROJECT_LIBRARY_NAME} PUBLIC DUMMY)
    target_include_directories(${PROJECT_LIBRARY_NAME} PUBLIC ../include/sl_function)
else()

message("Dynamic library ${PROJECT_LIBRARY_NAME} is being compiled and linked.")

    if(MODEL EQUAL 1)
    message("compiling MPC for Soft-Robot Model")     
    add_library(${PROJECT_LIBRARY_NAME} SHARED 
        sl_mpc.cpp
        sponge_mpc.cpp
        ModelBase.cpp
        models/RobotSystem.cpp     
    )
    endif() # MODEL

    # link third party libraries
    target_include_directories(${PROJECT_LIBRARY_NAME} PUBLIC ../include)
    target_link_libraries(
        ${PROJECT_LIBRARY_NAME}
        ${CMAKE_THREAD_LIBS_INIT}
        ${CASADI_LIBRARIES}
        Eigen3::Eigen
        yaml-cpp::yaml-cpp)    
    # install the library
    install(TARGETS ${PROJECT_LIBRARY_NAME} DESTINATION lib)

endif() # CMAKE_DUMMY

cmake_print_properties(
    TARGETS ${PROJECT_LIBRARY_NAME}
    PROPERTIES CXX_STANDARD COMPILE_DEFINITIONS INCLUDE_DIRECTORIES INSTALL_RPATH
)

else() # BUILD_LIB
endif() # BUILD_LIB

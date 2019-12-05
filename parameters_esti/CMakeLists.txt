cmake_minimum_required(VERSION 3.0)

project(parameters_esti)

SET(SRC_LIST parameters_debug.cpp)

find_package( OpenCV REQUIRED )
# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )

add_executable(parameters_esti ${SRC_LIST})
target_link_libraries( parameters_esti ${OpenCV_LIBS} )


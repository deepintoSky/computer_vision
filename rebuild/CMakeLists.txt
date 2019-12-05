cmake_minimum_required(VERSION 3.0)
add_compile_options(-std=c++11)

project(rebuild)
SET(SRC_LIST rebuild.cpp)

find_package( OpenCV REQUIRED )
# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )

add_executable(rebuild ${SRC_LIST})
target_link_libraries( rebuild ${OpenCV_LIBS} )


cmake_minimum_required(VERSION 3.0)
add_compile_options(-std=c++11)

project(lm)

# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )


SET(SRC_LIST lm.cpp )

add_executable(lm ${SRC_LIST})


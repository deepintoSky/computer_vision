cmake_minimum_required(VERSION 3.0)
add_compile_options(-std=c++11)

project(filter)
SET(SRC_LIST filter.cpp)

find_package( OpenCV REQUIRED )

add_executable(filter ${SRC_LIST})
target_link_libraries( filter ${OpenCV_LIBS} )


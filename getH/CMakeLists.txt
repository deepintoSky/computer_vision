cmake_minimum_required(VERSION 3.0)
project(getH)
SET(SRC_LIST getH.cpp)

find_package( OpenCV REQUIRED )

add_executable(getH ${SRC_LIST})
target_link_libraries( getH ${OpenCV_LIBS} )


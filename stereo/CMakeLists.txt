cmake_minimum_required(VERSION 3.0)
project(stereo)
SET(SRC_LIST stereo_simple.cpp)

find_package( OpenCV REQUIRED )

add_executable(stereo ${SRC_LIST})
target_link_libraries( stereo ${OpenCV_LIBS} )


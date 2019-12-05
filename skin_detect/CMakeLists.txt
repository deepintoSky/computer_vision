cmake_minimum_required(VERSION 3.0)
project(skin_detect)
SET(SRC_LIST skin_detect.cpp)
find_package( OpenCV REQUIRED )
add_executable(skin_detect ${SRC_LIST})
target_link_libraries( skin_detect ${OpenCV_LIBS} )


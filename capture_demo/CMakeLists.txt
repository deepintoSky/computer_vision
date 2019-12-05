cmake_minimum_required(VERSION 3.0)
project(capture_demo)
SET(SRC_LIST capture_demo.cpp)
find_package( OpenCV REQUIRED )
add_executable(capture_demo ${SRC_LIST})
target_link_libraries( capture_demo ${OpenCV_LIBS} )


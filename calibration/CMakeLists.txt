cmake_minimum_required(VERSION 3.0)
project(calibration)
SET(SRC_LIST calibration.cpp)

find_package( OpenCV REQUIRED )

add_executable(calibration ${SRC_LIST})
target_link_libraries( calibration ${OpenCV_LIBS} )


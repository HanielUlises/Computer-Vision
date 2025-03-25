cmake_minimum_required(VERSION 3.10)
project(frame_differencing)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(frame_differencing main.cpp)
target_link_libraries(frame_differencing ${OpenCV_LIBS})

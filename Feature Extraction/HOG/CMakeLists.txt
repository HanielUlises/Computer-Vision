cmake_minimum_required(VERSION 3.10)
project(HOGExtractor)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(HOGExtractor main.cpp HOGExtractor.cpp)

target_link_libraries(HOGExtractor ${OpenCV_LIBS})
cmake_minimum_required(VERSION 3.10)
project(contour_processing)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(contour_processing main.cpp)

target_link_libraries(contour_processing ${OpenCV_LIBS})
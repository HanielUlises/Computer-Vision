cmake_minimum_required(VERSION 3.10)
project(CudaOpenCVTest LANGUAGES CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cuda_test blur.cpp)
target_link_libraries(cuda_test ${OpenCV_LIBS})
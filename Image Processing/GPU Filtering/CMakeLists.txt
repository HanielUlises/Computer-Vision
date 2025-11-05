cmake_minimum_required(VERSION 3.18)
project(CudaOpenCVTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cuda_test blur.cpp)

target_link_libraries(cuda_test ${OpenCV_LIBS})

set_target_properties(cuda_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

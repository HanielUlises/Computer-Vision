cmake_minimum_required(VERSION 3.22)
project(hpcv_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -O3 -march=native>
  $<$<CONFIG:Debug>:-g -O0>
)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui dnn videoio)

find_package(CUDAToolkit REQUIRED)


set(TENSORRT_ROOT "" CACHE PATH "TensorRT install prefix (optional)")

if(TENSORRT_ROOT)
  set(TRT_INC "${TENSORRT_ROOT}/include")
  find_library(TRT_LIB       NAMES nvinfer        HINTS "${TENSORRT_ROOT}/lib" REQUIRED)
  find_library(TRT_ONNX_LIB  NAMES nvonnxparser   HINTS "${TENSORRT_ROOT}/lib" REQUIRED)
  find_library(TRT_PARSE_LIB NAMES nvparsers       HINTS "${TENSORRT_ROOT}/lib")
else()
  find_path(TRT_INC NAMES NvInfer.h
    PATHS /usr/local/include /usr/include
          /usr/local/tensorrt/include
    REQUIRED)
  find_library(TRT_LIB      NAMES nvinfer       PATHS /usr/lib /usr/local/lib REQUIRED)
  find_library(TRT_ONNX_LIB NAMES nvonnxparser  PATHS /usr/lib /usr/local/lib REQUIRED)
  find_library(TRT_PARSE_LIB NAMES nvparsers    PATHS /usr/lib /usr/local/lib)
endif()

message(STATUS "TRT include : ${TRT_INC}")
message(STATUS "TRT nvinfer : ${TRT_LIB}")
message(STATUS "TRT onnxparser : ${TRT_ONNX_LIB}")

add_library(hpcv_trt SHARED
  src/trt_inference_engine.cpp
  src/preprocessor.cpp
)

target_include_directories(hpcv_trt
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${TRT_INC}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(hpcv_trt
  PUBLIC
    ${TRT_LIB}
    ${TRT_ONNX_LIB}
    $<$<BOOL:${TRT_PARSE_LIB}>:${TRT_PARSE_LIB}>
    CUDA::cudart
    ${OpenCV_LIBS}
)

add_executable(trt_demo src/main.cpp)
target_link_libraries(trt_demo PRIVATE hpcv_trt)

option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(Catch2_FOUND)
    add_executable(trt_tests tests/test.cpp)
    target_link_libraries(trt_tests PRIVATE hpcv_trt Catch2::Catch2)
    include(CTest)
    include(Catch)
    catch_discover_tests(trt_tests)
  else()
    message(WARNING "Catch2 not found — tests skipped.")
  endif()
endif()

install(TARGETS hpcv_trt trt_demo
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include/hpcv)

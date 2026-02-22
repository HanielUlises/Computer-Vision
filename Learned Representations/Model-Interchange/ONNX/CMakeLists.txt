cmake_minimum_required(VERSION 3.22)
project(hpcv_onnx LANGUAGES CXX CUDA)

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

find_package(OpenCV REQUIRED
  COMPONENTS core imgproc highgui dnn videoio)

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime install prefix (optional)")

if(ONNXRUNTIME_ROOT)
  set(ORT_INC "${ONNXRUNTIME_ROOT}/include")
  find_library(ORT_LIB
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib"
    REQUIRED)
else()
  find_path(ORT_INC
    NAMES onnxruntime/core/session/onnxruntime_cxx_api.h
    PATHS /usr/local/include /usr/include
    REQUIRED)
  find_library(ORT_LIB
    NAMES onnxruntime
    PATHS /usr/local/lib /usr/lib
    REQUIRED)
endif()

message(STATUS "ORT include: ${ORT_INC}")
message(STATUS "ORT library: ${ORT_LIB}")

add_library(hpcv_onnx SHARED
  src/onnx_inference_engine.cpp
  src/preprocessor.cpp
)

target_include_directories(hpcv_onnx
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${ORT_INC}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(hpcv_onnx
  PUBLIC
    ${ORT_LIB}
    ${OpenCV_LIBS}
)

add_executable(onnx_demo src/main.cpp)
target_link_libraries(onnx_demo PRIVATE hpcv_onnx)

option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(Catch2_FOUND)
    add_executable(test tests/test.cpp)
    target_link_libraries(test PRIVATE hpcv_onnx Catch2::Catch2)
    include(CTest)
    include(Catch)
    catch_discover_tests(test)
  else()
    message(WARNING "Catch2 not found — tests skipped. "
                    "Install via: apt install catch2  OR  vcpkg install catch2")
  endif()
endif()

install(TARGETS hpcv_onnx onnx_demo
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include/hpcv)
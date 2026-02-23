cmake_minimum_required(VERSION 3.22)
project(hpcv_torch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── LibTorch ──────────────────────────────────────────────────────────────────
# Either set CMAKE_PREFIX_PATH to the LibTorch directory, or pass
# -DCMAKE_PREFIX_PATH=/path/to/libtorch on the command line.
#
# Download: https://pytorch.org/get-started/locally/
#   (select LibTorch, C++/Java, CUDA version)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui dnn videoio)

message(STATUS "Torch libraries : ${TORCH_LIBRARIES}")
message(STATUS "OpenCV  include : ${OpenCV_INCLUDE_DIRS}")

add_library(hpcv_torch SHARED
  src/torch_inference_engine.cpp
  src/preprocessor.cpp
)

target_include_directories(hpcv_torch
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(hpcv_torch
  PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

add_executable(torch_demo src/main.cpp)
target_link_libraries(torch_demo PRIVATE hpcv_torch)

set_target_properties(torch_demo PROPERTIES
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(Catch2_FOUND)
    add_executable(torch_tests tests/test.cpp)
    target_link_libraries(torch_tests PRIVATE hpcv_torch Catch2::Catch2)
    set_target_properties(torch_tests PROPERTIES
      BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
    include(CTest)
    include(Catch)
    catch_discover_tests(torch_tests)
  else()
    message(WARNING "Catch2 not found — tests skipped.")
  endif()
endif()

install(TARGETS hpcv_torch torch_demo
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include/hpcv)

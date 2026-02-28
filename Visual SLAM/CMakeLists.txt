cmake_minimum_required(VERSION 3.16)
project(VisualSLAM CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(OpenCV_DIR /usr/local/lib/cmake/opencv4)

find_package(OpenCV REQUIRED COMPONENTS
    core highgui imgproc features2d calib3d video viz)
find_package(Eigen3 REQUIRED)

set(SLAM_SOURCES
    src/feature_tracker.cpp
    src/pose_estimator.cpp
    src/map.cpp
    src/frontend.cpp
    src/visualizer.cpp
    src/slam_system.cpp
    src/main.cpp
)

add_executable(visual_slam ${SLAM_SOURCES})

target_include_directories(visual_slam PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(visual_slam
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_compile_options(visual_slam PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CXX_COMPILER_ID:GNU,Clang>:-ffast-math>
)
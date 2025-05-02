cmake_minimum_required(VERSION 3.10)
project(CameraPoseEstimation)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(camera_pose_estimation
    main.cpp
    camera_pose_estimator.cpp
)
target_link_libraries(camera_pose_estimation
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
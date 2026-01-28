cmake_minimum_required(VERSION 3.10)
project(SteerableFourierTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(
    steerable_test
    main.cpp
    SteerableFourier.cpp
)

target_include_directories(
    steerable_test
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    steerable_test
    PRIVATE
    ${OpenCV_LIBS}
)

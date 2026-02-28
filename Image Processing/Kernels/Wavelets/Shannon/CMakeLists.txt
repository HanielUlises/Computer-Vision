cmake_minimum_required(VERSION 3.10)
project(ShannonWavelet)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
)

add_executable(ShannonDemo
    main.cpp
    src/Shannon.cpp
)

target_link_libraries(ShannonDemo
    ${OpenCV_LIBS}
)
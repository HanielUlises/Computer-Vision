cmake_minimum_required(VERSION 3.10)
project(MexicanHatWavelet)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
)

add_executable(MexicanHatDemo
    main.cpp
    src/MexicanHat.cpp
)

target_link_libraries(MexicanHatDemo
    ${OpenCV_LIBS}
)
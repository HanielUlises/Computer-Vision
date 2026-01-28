cmake_minimum_required(VERSION 3.16)
project(gabor_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(gabor_demo
    main.cpp
    src/GaborEven.cpp
    src/GaborOdd.cpp
    src/GaborComplex.cpp
)

target_include_directories(gabor_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    include
)

target_link_libraries(gabor_demo PRIVATE
    ${OpenCV_LIBS}
)

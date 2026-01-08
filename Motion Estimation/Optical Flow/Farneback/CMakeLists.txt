cmake_minimum_required(VERSION 3.10)

project(FarnebackTracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(SOURCES
    main.cpp
    lk_tracker.cpp
    pyramidal_lk.cpp
)

set(HEADERS
    lk_tracker.hpp
    pyramidal_lk.hpp
)

add_executable(lk_tracker
    ${SOURCES}
    ${HEADERS}
)

target_include_directories(lk_tracker PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(lk_tracker
    ${OpenCV_LIBS}
)

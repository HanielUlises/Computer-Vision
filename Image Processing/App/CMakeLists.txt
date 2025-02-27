cmake_minimum_required(VERSION 3.10)

project(ImageViewer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

include_directories(include)

find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(ImageViewer src/main.cpp src/mainwindow.cpp)

target_link_libraries(ImageViewer Qt5::Widgets Eigen3::Eigen ${OpenCV_LIBS})



cmake_minimum_required(VERSION 3.16)
project(QtSvmApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 COMPONENTS Widgets REQUIRED QUIET)

if (NOT Qt6_FOUND)
    find_package(Qt5 COMPONENTS Widgets REQUIRED)
endif()


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

set(SOURCES
    src/main.cpp
    src/main_window.cpp
    src/SVM.cpp
)

set(HEADERS
    include/main_window.h
    include/SVM.h
)

set(UIS
    ui/main_window.ui
)

add_executable(QtSvmApp
    ${SOURCES}
    ${HEADERS}
    ${UIS}
)

if (Qt6_FOUND)
    target_link_libraries(QtSvmApp Qt6::Widgets ${OpenCV_LIBS})
else()
    target_link_libraries(QtSvmApp Qt5::Widgets ${OpenCV_LIBS})
endif()


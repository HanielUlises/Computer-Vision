cmake_minimum_required(VERSION 3.16)

project(ImgProcApp LANGUAGES CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt5 REQUIRED COMPONENTS Core Quick QuickControls2 Multimedia)
find_package(OpenCV REQUIRED)

qt5_add_resources(QML_RESOURCES qml.qrc)

add_executable(ImgProcApp
    src/main.cpp
    src/backend/file_capture.cpp
    src/backend/frame.cpp
    src/backend/thread_manager.cpp
    src/backend/file_capture.h
    src/backend/frame.h
    src/backend/thread_manager.h
    ${QML_RESOURCES}
)

target_include_directories(ImgProcApp PRIVATE
    src/backend
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ImgProcApp
    Qt5::Core
    Qt5::Quick
    Qt5::QuickControls2
    Qt5::Multimedia
    ${OpenCV_LIBS}
)

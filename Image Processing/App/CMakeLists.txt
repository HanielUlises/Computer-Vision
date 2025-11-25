cmake_minimum_required(VERSION 3.16)

project(ImgProcApp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 REQUIRED COMPONENTS Core Quick QuickControls2 Multimedia)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/backend/file_capture.cpp
    src/backend/frame.cpp
    src/backend/thread_manager.cpp
    src/backend/file_capture.h
    src/backend/frame.h
    src/backend/thread_manager.h
)

target_include_directories(${PROJECT_NAME} PRIVATE
    src/backend
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    Qt5::Core
    Qt5::Quick
    Qt5::QuickControls2
    Qt5::Multimedia
    ${OpenCV_LIBS}
)

set(QML_FILES
    qml/main.qml
    qml/home.qml
    qml/showImage.qml
    qml/showVideo.qml
)

qt5_add_resources(QML_RESOURCES qml.qrc)
target_sources(${PROJECT_NAME} PRIVATE ${QML_RESOURCES})

file(COPY ${QML_FILES} DESTINATION ${CMAKE_BINARY_DIR}/qml)